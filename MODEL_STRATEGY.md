# MODEL STRATEGY & COMPUTE FEASIBILITY: ARC PRIZE 2026 (ARC-AGI-3)

## 1. Executive Summary & Hardware Realities
ARC-AGI-3 on Kaggle requires an entirely self-contained, offline execution within a 9-hour runtime cap across 110 games ($\approx 294.5$ seconds per game, or $<5.59$ seconds per action for a 50-action budget).
Model selection is fundamentally bounded by **token generation throughput**, **inter-GPU communication latency**, and **memory footprint**.

---

## 2. Hardware Environments & Accelerator Physics

| Platform | Compute Architecture | Memory & Bandwidth | Interconnect | Native Precision | Kaggle VM Specs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local Development** | Apple Silicon M4 Air | 16 GB – 24 GB Unified RAM | On-die unified bus (~150 GB/s) | FP16, Metal / MLX INT4 | 8/10 CPU cores |
| **Kaggle Production (Dual T4)** | Turing (`sm_75`) | 32 GB (2x 16 GB GDDR6), 320 GB/s ea | PCIe 3.0 x16 (NO NVLink, ~16 GB/s) | FP16, INT8, INT4 (No native FP8) | 4 vCPUs, 30 GB RAM |
| **Kaggle Production (RTX 6000)** | Blackwell (`sm_100/120`) / Ada (`sm_89`) | **Up to 48 GB / 96 GB VRAM** | **Single Unified GPU** (Zero PCIe barrier) | FP16, BF16, **FP8**, INT4 | **48 vCPUs, 180 GB RAM** |

> [!IMPORTANT]
> **Key Hardware Discoveries:**
> 1. **Kaggle's Exclusive RTX 6000 Accelerator:** Kaggle provisioned Google Cloud `g4-standard-48` instances featuring NVIDIA RTX PRO 6000 specifically for ARC Prize 2026 (`"accelerator": "rtx6000"`). This eliminates the dual-T4 PCIe bottleneck entirely.
> 2. **The Dual-T4 PCIe Bottleneck:** Running tensor-parallel vLLM (`tp=2`) over PCIe without NVLink executes 128 `AllReduce` bus synchronizations per generated token for a 64-layer model (Qwen 32B), dropping throughput by 35–50%.
> 3. **llama.cpp Layer-Splitting Advantage:** On dual T4, `llama.cpp` using `LLAMA_SPLIT_MODE_LAYER` (layers 0–31 on GPU 0, layers 32–63 on GPU 1) passes activations across PCIe **only once per forward pass**, running **~30% faster** than tensor-parallel vLLM.

---

## 3. The 9-Hour Budget Equation & The CoT Trap

$$\text{Global Runtime} = 9\text{ hours} = 32,400\text{ seconds}$$
$$\text{Per-Game Budget} = \frac{32,400\text{ s}}{110\text{ games}} \approx 294.5\text{ seconds/game} \ (\approx 4.9\text{ minutes})$$
$$\text{Per-Action Budget (50 actions)} = \frac{294.5\text{ s}}{50\text{ actions}} \approx 5.89\text{ seconds/action}$$

### The Chain-of-Thought (CoT) Trap:
Verbose reasoning models (e.g. unconstrained DeepSeek-R1 generating 1,000+ token `<think>` blocks at 25 tok/s) consume $\approx 40$ seconds per action. An agent using unconstrained CoT will trigger a game timeout after only 7 actions.
**Engineering Mandate:**
- Interactive step-by-step actions must enforce strict completion token caps ($\le 80\text{--}120$ tokens).
- Heavy reasoning must be reserved for **high-level macro planning** and **hypothesis reflection**, while routine spatial movements are delegated to exact symbolic planners (A*, BFS).

---

## 4. Symbolic Representation vs. Multimodal Vision

For 64×64 discrete grid frames, pure vision-language models (e.g. Qwen 2.5-VL) suffer from:
1. ViT patch blurring across fine $1\times 1$ and $2\times 2$ pixel obstacles.
2. Spatial coordinate off-by-one errors on 64×64 grids.
3. High token generation overhead ($\approx 1,000$ image tokens per frame).

**Optimal Solution:** A deterministic computer vision frontend (CCL + diff extraction) converting frames into **compact symbolic text (ASCII / RLE / Object Entity Graphs)** feeding a code/reasoning LLM. This delivers superior accuracy, deterministic coordinates, and consumes $<300$ prompt tokens per turn.

---

## 5. Precise Memory Footprint & KV-Cache Sizing

### Formula:
$$\text{Total VRAM} = M_{\text{weights}} + M_{\text{KV}} + M_{\text{CUDA\_overhead}}$$
$$M_{\text{KV}} = 2 \times N_{\text{layers}} \times (N_{\text{kv\_heads}} \times D_{\text{head}}) \times 2\text{ bytes} \times \text{Context Length}$$

| Model Architecture | Quantization Format | Weights ($M_{\text{weights}}$) | KV Cache (4K Context) | KV Cache (8K Context) | Total VRAM (8K Context) | Dual T4 Feasibility | RTX 6000 Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 7B** | Q4_K_M / AWQ | **4.8 GB** | 0.23 GB | 0.47 GB | **6.1 GB** | **Fits Single T4** | Fits (6% of 96GB) |
| **Qwen 2.5 14B** | Q4_K_M / AWQ | **9.1 GB** | 0.79 GB | 1.57 GB | **11.5 GB** | **Fits Single T4** | Fits (12% of 96GB) |
| **Gemma 2 27B** | Q4_K_M / AWQ | **16.8 GB** | 1.47 GB | 2.94 GB | **20.5 GB** | Fits Dual T4 (~10.3G/ea) | Fits (21% of 96GB) |
| **Qwen 2.5 32B** | Q4_K_M / AWQ | **19.8 GB** | 1.05 GB | 2.10 GB | **22.7 GB** | **Fits Dual T4 (~11.4G/ea)** | Fits (24% of 96GB) |
| **Qwen 2.5 32B** | FP8 (Native) | 33.8 GB | 0.53 GB (FP8) | 1.05 GB (FP8) | 35.7 GB | **OOM on Dual T4** | **Fits (37% of 96GB)** |
| **Llama 3.3 70B** | Q4_K_M / AWQ | 41.5 GB | 1.34 GB | 2.68 GB | 45.0 GB | **OOM on Dual T4** | Fits (47% of 96GB) |

---

## 6. Inference Engine Benchmarks on Dual T4 vs. RTX 6000

| Engine & Configuration | Execution Topology | 14B Decode | 32B Decode | Memory Overhead | Cold Boot Time | Offline Packaging |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **llama.cpp (Layer-Split)** | `LLAMA_SPLIT_MODE_LAYER`, `-ts 0.5,0.5` | **36 tok/s** | **26 tok/s** | ~200 MB | **< 3s** | **Easiest (Static binary/wheel)** |
| **llama.cpp (Row-Split)** | `LLAMA_SPLIT_MODE_ROW`, `-ts 0.5,0.5` | 28 tok/s | 16 tok/s | ~200 MB | < 3s | Easiest |
| **vLLM (Dual T4 TP=2)** | Megatron Tensor Parallel (`tp=2`, AWQ) | 31 tok/s | 20 tok/s | 1.5 GB / GPU | ~35s | Medium (Pre-matched wheels) |
| **vLLM (Single T4 TP=1)** | Single GPU (`tp=1`, AWQ on GPU 0) | **38 tok/s** | N/A (OOM) | 1.5 GB | ~25s | Medium |
| **vLLM on RTX 6000** | Single Unified GPU (FP8 / AWQ) | **95 tok/s** | **55 tok/s** | 2.0 GB | ~20s | SOTA Production Baseline |

---

## 7. Authoritative Staged Model Deployment Recommendations

### 1. Local M4 Air Development Model:
- **Model:** **Qwen 2.5 14B Instruct (4-bit MLX or GGUF Q4_K_M via llama.cpp Metal)**.
- **Specs:** Consumes ~8.8 GB Unified RAM (fits within 16GB/24GB Macs), decodes at ~32 tok/s. Shares exact prompt structure and tool calling schemas with 32B Coder.

### 2. Kaggle Development Model (Fast Inner Loop on Dual T4):
- **Model:** **Qwen 2.5 14B Instruct (AWQ on single T4 via vLLM or Q4_K_M via llama.cpp)**.
- **Specs:** Runs on a single T4 card (`tp=1`), eliminating PCIe synchronization risks and multi-GPU deadlock issues. Decodes at ~38 tok/s. Total per-action latency is ~4.5s (225s per 50 actions).

### 3. Competition Champion Model (RTX 6000 Available):
- **Model:** **Qwen 2.5 Coder 32B Instruct (AWQ 4-bit or FP8 via vLLM)**.
- **Specs:** SOTA open coding model. In an interactive agent, the model acts as a programmer in a Python REPL loop (proven by Tufa Labs' Milestone 1 win). Single-GPU RTX 6000 delivers ~55 tok/s, completing 50 actions in ~110s.

### 4. Competition Champion Model (Dual T4 Fallback):
- **Model:** **Qwen 2.5 Coder 32B Instruct (GGUF Q4_K_M via llama.cpp layer-split)**, OR **Qwen 2.5 14B Instruct (AWQ on single T4)**.

### 5. Minimum Viable Safety Fallback:
- **Model:** **Qwen 2.5 7B Instruct (AWQ / Q4_K_M)**.
- **Specs:** Consumes <6 GB VRAM, decodes at 50+ tok/s, guarantees 0% chance of OOM or runtime timeout on any Kaggle hardware tier.

---

## 8. Offline Packaging Strategy for Kaggle Sandbox

1. **Zero-Copy mmap Loading:** Kaggle scratch disk (`/kaggle/working/`) is capped at ~20 GB. Copying model files results in `DiskQuotaExceeded` crashes. All engines must load directly from the read-only dataset mount point (`/kaggle/input/arc3-model/model.gguf`) using memory-mapped I/O (`mmap=True`).
2. **Pre-Compiled Wheels Dataset:** Create a private Kaggle dataset (`arc3-wheels`) with pre-downloaded wheels matching Kaggle's environment (`pip install --no-index --find-links=/kaggle/input/arc3-wheels/ llama-cpp-python`). Avoids burning 15+ minutes compiling from source.
3. **Deterministic Safety Fallback:** Any unhandled exception during LLM initialization (CUDA driver conflict, out-of-memory) must be caught by a top-level safety handler that falls back to the deterministic algorithmic explorer (`kaggle_submission.py`), ensuring that `submission.parquet` is always cleanly generated.
