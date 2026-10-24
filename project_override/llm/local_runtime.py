import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional
from project_override.config import AgentConfig

logger = logging.getLogger(__name__)

def parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust parser supporting fenced ```python ... ``` blocks (Duck REPL style)
    and balanced JSON objects with self-repair.
    """
    if not text:
        return None
    py_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_match:
        code_body = py_match.group(1).strip()
        if code_body:
            thought_prefix = text[: py_match.start()].strip()
            return {
                "subgoal": "CODEACT",
                "code": code_body,
                "thought": thought_prefix[:240],
            }
    dec = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None

class LocalLLMRuntime:
    """
    Offline open-weight LLM loader supporting local SGLang/vLLM HTTP endpoints
    (for Intel/Qwen3.8-Flash-Next-W4A16-AutoRound + albucino speculative drafter MTP=3)
    and local HuggingFace checkpoints, with hard latency/token budgets and non-LLM fallback.
    """
    _INSTANCE: Optional["LocalLLMRuntime"] = None
    _LOCK = threading.Lock()

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.http_endpoint: Optional[str] = None
        self.http_model_name: str = "default"
        self.model_path: Optional[str] = self._resolve_model_path()
        self.drafter_path: Optional[str] = self._resolve_drafter_path()
        self.model: Any = None
        self.tokenizer: Any = None
        self.is_loaded: bool = False
        self.load_attempted: bool = False
        self.total_calls: int = 0
        self.fallback_count: int = 0

    @classmethod
    def get_instance(cls, config: Optional[AgentConfig] = None) -> "LocalLLMRuntime":
        if cls._INSTANCE is None:
            with cls._LOCK:
                if cls._INSTANCE is None:
                    cls._INSTANCE = cls(config=config)
        return cls._INSTANCE

    def _resolve_model_path(self) -> Optional[str]:
        env_p = os.environ.get("ARC_LOCAL_MODEL_PATH")
        if env_p and Path(env_p).exists():
            return env_p
        for p in self.config.candidate_model_paths:
            if Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                low = str(cfg).lower()
                if "drafter" in low:
                    continue
                if any(k in low for k in ("qwen", "gemma", "coder", "flash", "autoround")):
                    return str(cfg.parent)
        return None

    def _resolve_drafter_path(self) -> Optional[str]:
        env_d = os.environ.get("ARC_LOCAL_DRAFTER_PATH")
        if env_d and Path(env_d).exists():
            return env_d
        for p in self.config.candidate_drafter_paths:
            if Path(p).exists():
                return p
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            for cfg in kaggle_in.rglob("config.json"):
                if "drafter" in str(cfg).lower():
                    return str(cfg.parent)
        return None

    def _probe_local_http_server(self) -> bool:
        for base_url in self.config.vllm_base_urls:
            if not base_url.startswith(("http://127.0.0.1", "http://localhost")):
                continue
            try:
                req = urllib.request.Request(f"{base_url}/models", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("data", [])
                    if models:
                        self.http_model_name = models[0].get("id", "default")
                    self.http_endpoint = f"{base_url}/chat/completions"
                    self.is_loaded = True
                    return True
            except Exception:
                continue
        return False

    def ensure_loaded(self) -> bool:
        if not self.config.enable_llm:
            return False
        if self.is_loaded:
            return True
        if self.load_attempted:
            return False
        with self._LOCK:
            if self.is_loaded:
                return True
            self.load_attempted = True
            if self._probe_local_http_server():
                logger.info(f"Connected to local SGLang/vLLM server at {self.http_endpoint}")
                return True
            if self.model_path is None:
                return False
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True, local_files_only=True
                )
                max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())}
                max_mem["cpu"] = "24GiB"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    max_memory=max_mem,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception as exc:
                logger.warning(f"LocalLLMRuntime load failed; using World-Model/BFS fallback: {exc}")
                self.is_loaded = False
                return False

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[str]:
        self.total_calls += 1
        max_tok = max_new_tokens or self.config.llm_max_new_tokens
        limit_s = timeout_s or self.config.llm_timeout_s
        if not self.ensure_loaded():
            self.fallback_count += 1
            return None
        if self.http_endpoint is not None:
            try:
                payload = json.dumps(
                    {
                        "model": self.http_model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tok,
                        "temperature": 0.1,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    self.http_endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=limit_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.debug(f"HTTP LLM inference error: {exc}")
                self.fallback_count += 1
                return None
        try:
            import torch
            t0 = time.perf_counter()
            messages = [{"role": "user", "content": prompt}]
            text = (
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if hasattr(self.tokenizer, "apply_chat_template")
                else prompt
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            if (time.perf_counter() - t0) > limit_s:
                self.fallback_count += 1
                return None
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            logger.debug(f"Transformers LLM inference error: {exc}")
            self.fallback_count += 1
            return None

    def infer_decision(self, prompt: str) -> Optional[Dict[str, Any]]:
        raw = self.generate_text(prompt)
        if not raw:
            return None
        return parse_llm_response(raw)
