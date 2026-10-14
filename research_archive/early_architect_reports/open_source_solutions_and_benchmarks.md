# Open-Source Solutions & Winning Approaches Research

**Source Conversation ID:** `1a08c608-2af4-4f7f-aafa-42ab63d141af`

## Original Prompt
```text
<USER_REQUEST>
Research open-sourced ARC-AGI-3 solutions and winning approaches. Look at these specific repositories and any others you can find:

1. https://github.com/tonghuikang/daniel-franzen-arc-agi-3/tree/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference
2. Search for 'ARC-AGI-3' on GitHub - find repos from Tufa Labs, Reki, Barada Sahu, and other top competitors
3. Search for ARC Prize 2025 solutions on Kaggle
4. Look for any Milestone 1 or Milestone 2 winner writeups
5. Search for arc-agi-3 on arxiv for any papers

For each solution found, extract:
- The overall architecture (LLM-based code generation? MCTS? Symbolic search?)
- The specific models used (Qwen, Gemma, etc.)
- How they handle the interactive environment
- Key tricks and optimizations
- Their actual scores

Provide URLs and concrete code patterns from at least 10 different solutions/repos.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-11T21:25:19+03:00.
</ADDITIONAL_METADATA>
```

---

## Architecture / Research Report

# Comprehensive Research Report: Open-Sourced ARC-AGI-3 Solutions & Winning Approaches

## Executive Summary

The ARC-AGI-3 benchmark (introduced for the ARC Prize 2026) marks a fundamental shift from static 2D grid completion (ARC-AGI-1/2) to **interactive, turn-based, multi-level puzzle/video-game environments** (up to 64×64 grids, 16 colors, 183 levels across 25 games, unknown physics/mechanics, and turn/action budgets).

Top approaches fall into four distinct architectural paradigms:
1. **Inference-Engine-Optimized Tool Calling Agents (Kaggle Milestone 1 & 2 Champions):** High-throughput local serving (SGLang/vLLM) of specialized hybrid-attention models (primarily **Qwen3.8-Flash-Next**) with aggressive prefix-caching, blockwise context trimming hysteresis, speculative decoding, and lightweight sandboxed Python REPLs.
2. **Direct Vision-as-Policy Models (e.g., Reki):** Multimodal LLMs (e.g., **Gemma-4-31B**) outputting structured JSON action plans and spatial click heuristics with reflection memory, bypassing code execution entirely.
3. **Dual-Agent Symbolic / Programmatic World Modeling (e.g., OPINE-World, Tycho, DreamTeam):** Separating the explorer/actor from a synthesizer that infers an executable world simulator in Python or finite-state Moore machines, steered by Bayesian "ontology error" or falsification events.
4. **Epistemic Constraint & Retrodictive Memory Harnesses (e.g., Retrodict, arc-skill, PRO-LONG):** Enforcing that no action may be taken without prior hypothesis falsification or explicit state prediction, coupled with programmatic append-only logs queried via code.

---

## 1. Solution Breakdown (Detailed Analysis of 12 Solutions)

---

### Solution 1: Daniel Franzen (`dfranzen`) — Milestone 2 1st Place Winner
- **URLs:** 
  - GitHub: https://github.com/tonghuikang/daniel-franzen-arc-agi-3 (fork with comparisons)
  - Upstream GitHub: https://github.com/da-fr/arc-agi-3-solution
  - Kaggle Notebook: https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution
  - Writeup: `WRITEUP.md` in repository
- **Actual Score:** **27.89%** Public Leaderboard (1st Place, ARC-AGI-3 Milestone 2, Sep 30, 2026).
- **Architecture:** Code-generation LLM agent interacting via Python REPL inside a custom patched SGLang Pennyroyal server on a single RTX PRO 6000 (96 GB).
- **Model Used:** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (quantized AutoRound W4A16 + BF16 PLE), paired with draft model `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE` (INT4 MTP draft) for 3-step speculative decoding.
- **How It Handles the Interactive Environment:**
  - Exposes an explicit **`UNDO`** action (`ACTION7`), which is vital for reversible mistake recovery without triggering game resets.
  - Distinguishes whole-board changes from gameplay changes via a 4-cell border exclusion mask (ignores budget timer bar shifts at the grid edge).
  - Supplies **fatal frames** and difference images upon game-over rather than simply displaying the restarted level.
  - Generates object-level descriptions via a `frame_diff()` tool comparing connected components.
  - Detects "transient pixels" in animations (pixels changing multiple times between frames) to prompt animation inspection.
- **Key Tricks & Optimizations:**
  - **Context Trim Hysteresis:** Context ceiling is 128 Ki (reserving 12 Ki for output). When budget is exceeded, it trims a large block down to ~59 Ki. This keeps the remaining prefix cache-hot for dozens of turns, achieving a **93.43% prefix cache hit rate**.
  - **Bounded Concurrent Admission:** Admits exactly 10 parallel game streams matching server slots to avoid KV cache thrashing. Priority scheduling prioritizes games with high progress and applies tail-fade in the last 20% of runtime.
  - **Persistent Python Functions:** Retains model-defined helper functions across all tool calls in a game session.
  - **No-Op & Stale-State Guards:** Batch execution halts if an action causes no interior board change; blocks redundant moves.
- **Concrete Code Pattern:**
```python
# Context trimming hysteresis & token accounting from ARC3-Inference/inference/agent/tool_agent.py
def _apply_context_trimming_hysteresis(self, messages, max_budget=118000, target_drain=59000):
    estimated_tokens = self._estimate_messages_tokens(messages)
    if estimated_tokens <= max_budget:
        return messages, False
    
    # Hysteresis: discard a large block so prefix stays cache-hot for many future turns
    system_msg = messages[0]
    history = messages[1:]
    dropped_count = 0
    while history and self._estimate_messages_tokens([system_msg] + history) > target_drain:
        # Drop oldest turn pairs (user + assistant)
        history.pop(0)
        dropped_count += 1
    
    # Re-insert reminder notice to prevent stale outcome confusion
    history.insert(0, {
        "role": "system",
        "content": "[Context pruned: Prior history compressed. Retaining current level progress and functions.]"
    })
    return [system_msg] + history, True
```

---

### Solution 2: Tufa Labs ("The Duck") — Milestone 1 1st Place Winner
- **URLs:**
  - GitHub: https://github.com/Tufalabs/duck-harness
  - Kaggle Writeup: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133
  - Blog: https://tufalabs.ai/research/duck-harness/
- **Actual Score:** **1st Place**, Milestone 1 (June 30, 2026).
- **Architecture:** The Duck harness built on TAAF (Tufa ARC-AGI Framework). Uses a Python REPL tool calling loop where the agent writes executable code to interact with game environments.
- **Model Used:** Qwen 3.6 27B FP8 (and later migrated to Qwen 3.8-Flash-Next).
- **How It Handles the Interactive Environment:**
  - Multimodal perception combining rendered 256x256 PNG images, ASCII grid representations, and connected-component segmentation.
  - Executes Python snippets in a sandboxed runner that calls `game.step(action)`.
- **Key Tricks & Optimizations:**
  - **"Infinite Play via Eviction":** Rolling context window manager that evicts the oldest messages once a ceiling is hit while keeping the initial system prompt and recent action history.
  - **Object-Centric REPL Representation:** Connected-component labeling converts raw numeric matrices into discrete object records.
- **Concrete Code Pattern:**
```python
# TAAF solver interaction loop pattern
class DuckAgent:
    def step(self, game_state):
        obs_text = self.format_ascii_grid(game_state.grid)
        prompt = f"Current Board:\n{obs_text}\nAvailable Actions: {game_state.available_actions}"
        response = self.llm.generate(prompt, tools=[self.python_repl_tool])
        if response.has_tool_call:
            result = self.python_repl_tool.execute(response.tool_call.code)
            return self.parse_actions(result)
        return self.parse_direct_action(response.content)
```

---

### Solution 3: lordhansolo — Milestone 2 2nd Place
- **URLs:**
  - GitHub (Copied Bundle): https://github.com/tonghuikang/daniel-franzen-arc-agi-3/tree/main/kaggle/lordhansolo
  - Kaggle Notebook: `lordhansolo/arc-agi-3-milestone-2`
  - Source Bundle: `lordhansolo/dataset-taaf-kaggle-source`
- **Actual Score:** **23.84%** Public Leaderboard (2nd Place, Milestone 2).
- **Architecture:** Agent with heavily rewritten tool harness on a customized, SHA256-pinned vLLM nightly server (0.29.1rc1 nightly e975732).
- **Model Used:** `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` (NVFP4 + FP8 mixed + BF16 PLE, with built-in NVFP4 Multi-Token Prediction head).
- **How It Handles the Interactive Environment:**
  - 14 parallel game streams running in 8 waves.
  - Rewrote tool agent, sandbox, and local analyzer; incorporates a strategy audit prompt inserted after a level consumes >25% of game time.
- **Key Tricks & Optimizations:**
  - **Catalog Stripping before History Drop:** Maintains an internal target context limit of **81,536 tokens**. When context ceiling (~126k) is exceeded, it first strips old saved Python module catalogs from earlier user messages, and only then prunes oldest turns.
  - **Serving Patches:** Pinned-host input embeddings, pruned draft vocabulary (32k), exact CUDA graphs at 44 and 52 tokens for 14 sequences with MTP3, and SM120 small BF16 GEMMs for decode rows 40 to 64.
- **Concrete Code Pattern:**
```python
# From lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/tool_agent.py
def _compact_messages_for_context(self, messages, *, phase="normal"):
    # Strip saved-module catalogs from earlier user messages first
    compacted = []
    for msg in messages[:-1]:
        if msg.get("role") == "user" and "SAVED_MODULE_CATALOG" in msg.get("content", ""):
            cleaned = self._strip_module_catalog(msg["content"])
            compacted.append({**msg, "content": cleaned})
        else:
            compacted.append(msg)
    compacted.append(messages[-1])
    
    # If still exceeding target, drop turns until within LOCAL_ANALYZER_TARGET_CONTEXT (81536)
    while self.estimate_tokens(compacted) > self._context_target_tokens and len(compacted) > 2:
        compacted.pop(1) # Drop oldest non-system turn
    return compacted
```

---

### Solution 4: Lohit Siriki (`rellik13` / `sirikilohit`) — Milestone 2 3rd Place
- **URLs:**
  - GitHub: https://github.com/LohitSiriki/arc-agi-3-milestone2-solution
  - Writeup: `WRITEUP.md` in repository
  - Kaggle Notebook: `sirikilohit/arc-agi-3-duck-18-1gc-submit`
- **Actual Score:** **22.53%** Public Leaderboard (3rd Place, Milestone 2).
- **Architecture:** SGLang Pennyroyal 2.5.0 serving Qwen3.8-Flash-Next on a single RTX PRO 6000.
- **Model Used:** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` + RadixArk FP8 PLE + NVFP4 MTP draft.
- **How It Handles the Interactive Environment:**
  - Exposes `ACTION7` as `UNDO`.
  - Replaced confusing "The game is over" message with "Level restarted."
  - Removed misleading instructions that told the agent to ignore border bars, teaching it that border bars represent the action budget deadline.
- **Key Tricks & Optimizations:**
  - **FP8 KV Cache:** Halved KV cache memory footprint, expanding the pool to ~1.0M tokens.
  - **Context Watermarks:** Raised context length to 69,632 tokens, trimming from 57,344 down to 45,056 tokens.
  - **Solved-Level Memory (M85):** Captures the verified rule and the last 30 actions of cleared levels, pinning them at the end of the system prompt so the agent does not waste 60–100 minutes rediscovering mechanics on new levels.
  - **One-Pool UCB Scheduler (M72):** Dynamically allocates GPU time slices from a shared 31k-second pool, prioritizing games that clear levels with the fewest tokens.
- **Concrete Code Pattern:**
```python
# From LohitSiriki/arc-agi-3-milestone2-solution/cells/42_patch_cell_v16_8_m85_solved_level_memory.py
def m85_update_system_prompt_with_memory(system_prompt, cleared_level, rule_text, winning_actions):
    memory_block = [
        "=== SOLVED LEVELS (facts from this game + your own conclusions) ===",
        f"Level {cleared_level} cleared in {len(winning_actions)} actions.",
        f"Rule: {rule_text}",
        f"Winning action tail: {winning_actions[-30:]}",
        "=== END SOLVED LEVELS ==="
    ]
    return system_prompt + "\n\n" + "\n".join(memory_block)
```

---

### Solution 5: Reki — Milestone 1 2nd Place
- **URLs:**
  - ARC Prize Report: https://arcprize.org/blog/arc-prize-2026-milestone-1-winners
- **Actual Score:** **2nd Place**, Milestone 1 (June 30, 2026).
- **Architecture:** "Vision-LLM-as-policy" agent. Direct image-to-action reasoning without code generation or Python REPL execution.
- **Model Used:** **Gemma-4-31B** (multimodal, served locally).
- **How It Handles the Interactive Environment:**
  - Renders recent frames as labeled images and inputs them into the VLM.
  - The model outputs a structured JSON response containing:
    1. Visual observations.
    2. A brief natural-language plan.
    3. An action queue of 1–4 actions (executed sequentially to amortize inference overhead).
- **Key Tricks & Optimizations:**
  - **Reflection Memory:** A running summary of past trial actions and their outcomes, refreshed periodically to avoid stale loops.
  - **Numpy Click Heuristics:** Prioritizes clicking on distinct visual buttons, switches, or interactable shapes over random grid coordinates.
  - **Legal-Action Masking & Self-Repair:** Validates returned JSON schema and action constraints, running an immediate self-repair routine if parsing fails.
- **Concrete Code Pattern:**
```json
// Reki Output Schema Pattern
{
  "observation": "Red agent moved adjacent to green door; yellow key collected in previous step.",
  "plan": "Step right onto green door to trigger level finish.",
  "actions": ["ACTION4", "ACTION5"],
  "reflection_update": "Green door unlocks automatically when key is held."
}
```

---

### Solution 6: OPINE-World (David Courtis, Barada Sahu, Wenhao Li, Scott Sanner)
- **URLs:**
  - GitHub: https://github.com/david-courtis/opine-world
  - ArXiv Paper: https://arxiv.org/abs/2607.01531 (arXiv:2607.01531)
  - Project Page: https://david-courtis.github.io/opine-world/
- **Actual Score:** **78.4% RHAE** (Relative Human Action Efficiency), solved 20/25 public games without training. Barada Sahu ranked 3rd on the live competition leaderboard.
- **Architecture:** Dual cooperating LLM agents in a continuous hypothesis-and-test loop:
  - **Agent 1 (Actor):** Operates directly in the game environment.
  - **Agent 2 (Modeler):** Synthesizes and maintains an explicit, object-centric programmatic world model in Python.
- **How It Handles the Interactive Environment:**
  - Replays historical transitions against the synthesized Python model. When the model accurately simulates the environment, plans are computed symbolically.
- **Key Tricks & Optimizations:**
  - **Bayesian "Ontology Error" Steering:** Calculates Dirichlet entropy over observed state transitions (`_dirichlet_entropy`) to quantify uncertainty about object interactions. Directs exploration specifically toward areas of high epistemic uncertainty.
  - **Exact-Replay Verification:** Any synthesized world model must pass 100% of recorded historical transitions before being trusted for planning.
- **Concrete Code Pattern:**
```python
# From src/opine_world/synth_loop/sigma.py
def _dirichlet_entropy(counts: dict[str, int], alphabet_size: int, alpha_0: float) -> float:
    E = max(2, int(alphabet_size))
    n = sum(counts.values())
    denom = E * alpha_0 + n
    ent = 0.0
    for cnt in counts.values():
        p = (alpha_0 + cnt) / denom
        ent -= p * math.log(p)
    n_unseen = E - len(counts)
    if n_unseen > 0 and alpha_0 > 0:
        p0 = alpha_0 / denom
        ent -= n_unseen * (p0 * math.log(p0))
    eta = ent / math.log(E)
    return max(0.0, min(1.0, eta))
```

---

### Solution 7: Tycho (NIMI Research)
- **URLs:**
  - GitHub: https://github.com/NIMI-research/Tycho
  - Verified Scorecards: https://arcprize.org/scorecards/08b98aa0-5df0-42c0-b501-856f553a21e9
- **Actual Score:** **100.00% RHAE** with Claude Opus 5 & GPT-5.6 Sol; 88.49% with Claude Opus 4.8.
- **Architecture:** Active abstraction using **Rendered Deterministic Moore Machines** (symbolic finite-state automata + neural coding agent).
- **Model Used:** Claude Opus 5, GPT-5.6 Sol, Claude Opus 4.8.
- **How It Handles the Interactive Environment:**
  - Models the environment as an unknown deterministic Moore machine with four executable components: `State`, `transition(state, action)`, `render(state) -> grid`, and `outcome(state)`.
  - When the actual rendered grid diverges from the predicted `render()`, a counterexample triggers state refinement (CEGAR).
- **Key Tricks & Optimizations:**
  - **Actor-Controlled Builder:** The actor decides when to formalize a model vs. when to reason ad-hoc.
  - **Executable Counterexamples:** Immediate falsification prevents hallucinated plans from being executed on the live environment.
- **Concrete Code Pattern:**
```python
# Tycho Executable Hypothesis Interface Pattern
class MooreWorldModel:
    def initial_state(self) -> Any: ...
    def transition(self, state: Any, action: str) -> Any: ...
    def render(self, state: Any) -> np.ndarray: ...
    def outcome(self, state: Any) -> str: # 'CONTINUE', 'LEVEL_COMPLETE', 'GAME_OVER'
        ...
        
def verify_hypothesis(model: MooreWorldModel, replay_buffer: list) -> bool:
    s = model.initial_state()
    for step in replay_buffer:
        expected_grid = model.render(s)
        if not np.array_equal(expected_grid, step.observed_grid):
            return False # Counterexample found: trigger refinement
        s = model.transition(s, step.action)
    return True
```

---

### Solution 8: Retrodict (Ryan B. Brown)
- **URLs:**
  - GitHub: https://github.com/ryanbbrown/Retrodict
  - Verified Scorecard: https://arcprize.org/scorecards/9c403765-db5b-40b1-beab-6fa3f40119b0
- **Actual Score:** **99.86% RHAE**, all 183 levels solved across all 25 public games for $654 (using 7,703 actions, 0.66B tokens on GPT-5.6 Sol).
- **Architecture:** Retrodictive reasoning harness combining forward simulation with backward hypothesis checking.
- **Model Used:** GPT-5.6 Sol.
- **How It Handles the Interactive Environment:**
  - Operates via a structured `log.txt` and a sandboxed Python execution tool with a bundled `arclog` parsing library.
  - Enforces strict prediction discipline: every action must include a predicted outcome before being executed.
- **Key Tricks & Optimizations:**
  - **Retrodiction (Hypothesis Falsification):** Before spending an action, the agent writes Python to test candidate mechanics against past recorded frames in `log.txt`. If any past frame contradicts the rule, it is discarded without burning an action.
  - **Playbook Memory (`playbook.md`):** Separated into a **working model** (proven mechanics & objectives) and **working memory** (current position & ruled-out paths). Preserved across context compactions.
  - **Image Priming:** Uses vision only once on the opening board to bootstrap hypothesis generation, then proceeds via fast symbolic/text logging.
- **Concrete Code Pattern:**
```python
# Retrodict Prediction & Retrodiction Verification Loop
def test_hypothesis_against_log(hypothesis_func, log_history):
    for frame_t, action, frame_next in log_history.transitions():
        predicted_next = hypothesis_func(frame_t, action)
        if not np.array_equal(predicted_next, frame_next):
            return False, f"Falsified at step {frame_t.step_idx}"
    return True, "Consistent with all past history"

# Action mandate: calculate prediction before execution
# arc.execute(action, expected_frame=simulated_frame)
```

---

### Solution 9: arc-skill (`pbshgthm`)
- **URLs:**
  - GitHub: https://github.com/pbshgthm/arc-skill
  - Writeup / Visualizer: https://arc-skill.vercel.app/
  - Scorecard: https://arcprize.org/scorecards/24ddb219-987e-464f-9050-6398a29cf5ac
- **Actual Score:** **100.00% RHAE**, 25/25 games, 183/183 levels solved in 7,645 actions (human median: 17,135).
- **Architecture:** Standalone CLI agent skill for Claude Code on Claude Opus 5.
- **Model Used:** Claude Opus 5 via Claude Code.
- **How It Handles the Interactive Environment:**
  - Implements a strict enforcement rule: **"Say what an action will do before you spend it."**
  - CLI command requires explicit `--predict` and `--because` arguments.
- **Key Tricks & Optimizations:**
  - **Formal Claim Grammar:** 8 supported claim primitives checked against server state:
    - `cell X,Y=V` (cell will hold color V)
    - `move X,Y DX,DY` (object shifts by displacement)
    - `region X0:X1,Y0:Y1` (bounding box changes)
    - `vanish X,Y` (pixel/object disappears)
  - Refuses execution if predictions are missing; logs failed predictions to immediately localize invalid assumptions.
- **Concrete Code Pattern:**
```bash
# arc-skill invocation pattern
arc act ACTION6 58 32 \
  --predict "cell 22,17=b; cell 10,17=4" \
  --because "right arrow: test if cursor moves one tile clockwise"
```

---

### Solution 10: PRO-LONG (Alexis Fox)
- **URLs:**
  - GitHub: https://github.com/alexisfox7/PRO-LONG
  - ArXiv Paper: https://arxiv.org/abs/2607.20064 (arXiv:2607.20064)
- **Actual Score:** **97.4% best@2** on full ARC-AGI-3 public set (using Fable 5, $1,750 cost).
- **Architecture:** Programmatic memory for long-horizon LLM agents. Instead of dumping bloated history or heuristic summaries into the prompt, the agent queries an append-only `/workspace/logs.txt` using code (`grep`, Python).
- **Model Used:** Fable 5, GPT-5.5, Claude Opus 4.6.
- **How It Handles the Interactive Environment:**
  - Appends every observation, action, and outcome to `/workspace/logs.txt`.
  - The system prompt is kept ultra-compact (~30 lines).
- **Key Tricks & Optimizations:**
  - **Sub-Linear Prompt Growth:** Prevents context window exhaustion over hundreds of turns.
  - Avoids precision degradation from reading ASCII matrices in LLM context by forcing the model to write Python to compute over matrix coordinates directly.
- **Concrete Code Pattern:**
```python
# Agent bash/python memory search pattern in PRO-LONG
# Agent executes a script to find when a specific color disappeared:
import json, re

def find_disappearance(color_code):
    with open('/workspace/logs.txt') as f:
        logs = f.read().split('--- STEP ---')
    for idx, step in enumerate(logs):
        if f"vanished: {color_code}" in step or f"target={color_code}" in step:
            print(f"Step {idx}: {step[:200]}")

find_disappearance('8') # Red
```

---

### Solution 11: TELL (Studio Dots AI)
- **URLs:**
  - GitHub: https://github.com/studio-dots-ai/TELL
  - Website: https://rednote-hilab.github.io/TELL/
- **Actual Score:** **43.9%** on ARC-AGI-3 (11/25 games won) with zero pre-training or fine-tuning ($1,406 total API cost).
- **Architecture:** Test-time Experiential Lifelong Learning in a single conversation thread.
- **Model Used:** Claude Opus 4.6.
- **How It Handles the Interactive Environment:**
  - Three tools: `screen_shot`, `bash_exec`, `write_file`.
  - Maintains and continually edits a single structured `MEMORY.md` file representing confirmed world dynamics, rules, and level transfer mappings.
- **Key Tricks & Optimizations:**
  - Context compaction triggered strictly at 80% window utilization.
  - Distills confirmed rules at level transitions so old action logs can be discarded while preserving game knowledge.
- **Concrete Code Pattern:**
```python
# TELL Compaction & Memory Distillation Hook
def on_context_threshold_reached(agent):
    # Prompt agent to summarize recent discoveries into MEMORY.md before compaction
    agent.run_turn(
        "Context limit approaching. Update MEMORY.md with all confirmed mechanics, "
        "ruled-out hypotheses, and current level goal. Discard raw turn transcripts."
    )
    agent.compact_history(keep_last_turns=3)
```

---

### Solution 12: NVIDIA DreamTeam
- **URLs:**
  - GitHub: https://github.com/NVIDIA/dream-team
  - ArXiv Paper: https://arxiv.org/abs/2605.09650 (arXiv:2605.09650)
- **Actual Score:** Solves complex multi-level environments with high coordination efficiency.
- **Architecture:** Multi-agent cooperative architecture coordinating 6 specialized agents around a shared, executable world model:
  1. **Observer:** Encodes board state and visual features.
  2. **Simulator:** Synthesizes and tests transition dynamics.
  3. **Explorer 1 & 2:** Propose exploratory probing policies.
  4. **Critic:** Checks evidence and falsifies hypotheses.
  5. **Team Leader:** Selects the final consensus action.
- **Key Tricks & Optimizations:**
  - File-based team workspace memory for inter-agent communication, avoiding monolithic context limits.

---

## 2. Summary Comparison Matrix

| Solution / Team | Overall Architecture | Model(s) Used | Context & Memory Strategy | Serving Stack / Hardware | Key Performance Metric / Score |
|---|---|---|---|---|---|
| **Daniel Franzen** (`dfranzen`) | Tool REPL Agent | Qwen3.8-Flash-Next (AutoRound W4A16) | Blockwise drain hysteresis (118k → 59k), persistent Python functions | SGLang Pennyroyal 2.5.3 (1x RTX PRO 6000) | **27.89%** LB (Milestone 2 Winner) |
| **lordhansolo** | Tool REPL Agent | Qwen3.8-Flash-Next (NVFP4-FP8 mixed) | Strip module catalogs first, then target trim (81,536) | Custom patched vLLM (1x RTX PRO 6000) | **23.84%** LB (Milestone 2 2nd Place) |
| **sirikilohit** (`rellik13`) | Tool REPL Agent | Qwen3.8-Flash-Next (W4A16 + RadixArk FP8) | Solved-level memory (M85), FP8 KV cache, watermarks (57k → 45k) | SGLang Pennyroyal 2.5.0 (1x RTX PRO 6000) | **22.53%** LB (Milestone 2 3rd Place) |
| **Tufa Labs** ("The Duck") | Python REPL Harness (TAAF) | Qwen 3.6 27B FP8 / Qwen3.8-Flash-Next | Rolling window FIFO eviction ("Infinite play") | vLLM / OpenRouter | **Milestone 1 Winner** |
| **Reki** | Direct VLM-as-Policy | Gemma-4-31B | Reflection memory + 1–4 action plan queues | Local VLM serving | **Milestone 1 2nd Place** |
| **OPINE-World** (Barada Sahu / Courtis) | Dual Actor + Modeler | Frontier LLM / Qwen | Programmatic Python model with replay verification | Distributed / Local API | **78.4% RHAE** (20/25 games won) |
| **Tycho** (NIMI Research) | Moore Machine (CEGAR) | Claude Opus 5 / GPT-5.6 Sol | Rendered Moore machine state abstraction | Cloud API / Containerized Sandboxes | **100.00% RHAE** (ceiling) |
| **Retrodict** (Ryan Brown) | Retrodictive Reasoning | GPT-5.6 Sol | `log.txt` + `playbook.md` working model | API / Sandboxed venv | **99.86% RHAE** ($654 total) |
| **arc-skill** (`pbshgthm`) | Action Prediction Gating | Claude Opus 5 (Claude Code) | Mandatory `--predict` precondition gating | Claude Code CLI | **100.00% RHAE** (7,645 actions) |
| **PRO-LONG** (Alexis Fox) | Programmatic Memory | Fable 5 / Claude Opus 4.6 | Append-only `logs.txt`, searched via code | Sandboxed Docker | **97.4% best@2** ($1,750 total) |
| **TELL** (Studio Dots AI) | Experiential In-Context Learning | Claude Opus 4.6 | `MEMORY.md` updated at 80% compaction | Anthropic API | **43.9%** (11/25 games won) |
| **DreamTeam** (NVIDIA) | 6-Agent Cooperative Team | Multiple model profiles | Shared workspace file coordination | Custom Beam Framework | High long-horizon solve rate |

---

## 3. Key Insights & Universal Winning Patterns

From analyzing all Milestone winners, Kaggle submissions, and arXiv preprints, five technical requirements distinguish winning solutions:

1. **Prefix-Cache Preservation is the #1 Driver of GPU Throughput:**
   - Small, frequent context evictions break the KV/Mamba prefix cache, causing GPU thrashing due to repeated prefill.
   - Successful solutions (Franzen, Siriki, lordhansolo) use **blockwise hysteresis**: allowing context to grow to ~120k tokens and then trimming in large blocks (e.g., down to 50–60k tokens), preserving >90% cache hits.
2. **Memory of Solved Levels (Level-Transfer Knowledge):**
   - Re-deriving basic mechanics on every level wastes up to 70% of the available time budget.
   - Preserving a distilled summary of cleared level rules and winning action tails (Siriki's M85, Retrodict's `playbook.md`, TELL's `MEMORY.md`) yields double-digit score improvements.
3. **Epistemic Discipline ("Predict Before You Act"):**
   - Direct greedy search wastes actions. Solutions achieving near-100% RHAE (Tycho, Retrodict, arc-skill) mandate that agents compute expected frame diffs or retrodictively check hypotheses against past logs before executing actions in the game.
4. **Environment Interface Fidelity:**
   - Exposing `ACTION7` explicitly as `UNDO` is critical for reversible exploration without triggering full level resets.
   - Distinguishing between gameplay changes and the shrinking action-budget timer bar at the board boundary prevents agents from mistaking timer ticks for successful game moves.
5. **Programmatic Querying over In-Prompt Dumps:**
   - Ingesting raw 64×64 grids directly into the LLM context causes spatial attention degradation.
   - Top agents use connected-component segmentation and query history files programmatically via Python or grep.

## Additional Synthesized Content

I have completed the comprehensive research into open-sourced ARC-AGI-3 solutions, Milestone 1 & 2 winners, Kaggle submissions, and arXiv papers, and have transmitted the full report with concrete code patterns, architectures, models, optimizations, scores, and URLs across 12 top solutions to the parent agent.