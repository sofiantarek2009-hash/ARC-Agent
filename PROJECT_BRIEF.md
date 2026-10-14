# PROJECT BRIEF: BUILD A TOP-TIER, GENERAL, FLUID-INTELLIGENCE AGENT FOR ARC PRIZE 2026 — ARC-AGI-3

You are the lead research and engineering agent for an ARC Prize 2026 ARC-AGI-3 Kaggle competition project. Your mission is to design, research, implement, test, and continuously improve a genuinely general interactive-reasoning agent capable of achieving the highest possible competition score and RHAE.

This is not a task to build a few bespoke solvers for public games. This is a task to build true fluid intelligence: an agent that can enter an unfamiliar game, infer its rules from interaction alone, maintain and revise a world model, plan efficiently, avoid wasted actions, and generalize to completely novel mechanics.

Your ultimate objective is to maximize the final Kaggle competition leaderboard score, with RHAE as a central optimization target. You must optimize for both completion and efficiency. A game is not truly solved if the agent reaches the end only after burning excessive actions, entering loops, timing out, or relying on privileged information.

You must operate with extreme intellectual honesty. Previous work in this project demonstrated how easy it is to accidentally or deliberately overfit by reading environment source files, fingerprinting pixel layouts, hardcoding waypoints, or creating game-specific branches. None of that is acceptable. Your success will be judged by whether the system generalizes to unseen games under the official competition constraints.

======================================================================
1. COMPETITION REALITY AND NON-NEGOTIABLE CONSTRAINTS
======================================================================

The Kaggle ARC Prize 2026 ARC-AGI-3 competition is an offline, open-source, open-weights competition.

You must respect the following constraints at all times:

1. The final Kaggle submission must run without internet access.
2. The final submission must use an open-weight model and open-source code.
3. You may not call GPT, Claude, Gemini, or any external proprietary API from within the scored Kaggle submission.
4. The submitted system must be reproducible and fully self-contained.
5. Model weights, inference code, prompts, harness code, and all necessary assets must be packaged appropriately for offline Kaggle execution.
6. Kaggle provides GPU T4 x2 in many configurations, with 16GB VRAM per T4 and 32GB total GPU VRAM.
7. A 27B-class open-weight model can potentially run on Kaggle T4 x2 only if quantized appropriately, typically Q4/Q4_K_M GGUF or another memory-efficient format.
8. The submission notebook has a strict runtime cap, so token efficiency, inference speed, context management, and action efficiency are first-class engineering constraints.
9. The public demonstration set contains only 25 games, while the competition evaluates 110 separate hidden games.
10. Half of the hidden games determine the Public Leaderboard; the other half determine the final Private Leaderboard.
11. Local scores, public-set scores, and leaderboard scores are related but not equivalent. Never treat one as a proxy for another without explicit validation.
12. Community evidence shows that public/local performance can differ dramatically from leaderboard performance. A 20–25% local/public score has sometimes translated to only 3–7% on the leaderboard. In other cases, the ratio has been better. Therefore, do not build a naive conversion formula or assume a fixed public-to-leaderboard ratio.
13. The goal is not to maximize a local benchmark score in isolation. The goal is to maximize genuine generalization and therefore the final competition score.

You must repeatedly remind yourself: a high score on 25 public games does not prove the agent is generally intelligent. A high score on a small local cohort does not prove leaderboard transfer. Only a frozen evaluation on untouched held-out games, followed by careful submission testing, provides meaningful evidence.

======================================================================
2. DEFINITION OF SUCCESS
======================================================================

Success is not “we solved 16 games we studied.”

Success is:

1. The agent receives only official observations: frames, available actions, chosen actions, resulting frames, level-completion signals, and action budgets.
2. It infers the game’s mechanics without reading source code, metadata, level definitions, transcripts, or internal game state.
3. It builds a compact, accurate, revisable world model.
4. It identifies objects, goals, obstacles, interactive elements, and causal relationships.
5. It chooses informative actions when uncertain.
6. It plans efficiently once it understands the game.
7. It detects when its predictions fail and revises its hypotheses.
8. It avoids loops, repeated dead-end actions, excessive exploration, and timeouts.
9. It completes as many levels and games as possible.
10. It completes them with as few actions as possible relative to the human baseline.
11. It generalizes across simple navigation games, memory games, toggle games, dynamic environments, hidden-state games, graph puzzles, construction games, and multi-step planning games.
12. Its improvements transfer to fresh untouched games, not merely to the games used during development.

The ideal agent is not a blind tester. It is an intelligent scientist: it forms hypotheses, designs minimal experiments, observes consequences, updates beliefs, and then exploits its understanding efficiently.

======================================================================
3. YOUR FIRST MAJOR DELIVERABLE: DEEP RESEARCH
======================================================================

Before writing substantial production code, deploy multiple research subagents in parallel. Your first responsibility is to build a rigorous research base, not to start hacking.

Deploy separate subagents for the following research tracks:

TRACK A — COMPETITION RULES AND EVALUATION MECHANICS

Research and document:

- Exact Kaggle ARC Prize 2026 ARC-AGI-3 rules.
- Submission format and runtime constraints.
- Internet restrictions during evaluation.
- Allowed model licensing and open-source requirements.
- GPU hardware available during scoring.
- Memory, disk, CPU, and runtime limits.
- How the 110 hidden games are split between public and private leaderboards.
- How scores are computed.
- How RHAE is computed.
- How human baselines are established.
- How action budgets work.
- What constitutes a win, partial progress, timeout, disqualification, or zero score.
- How repeated submissions and final private leaderboard selection work.
- Any known pitfalls that cause strong local agents to fail on the leaderboard.

Produce a concise, precise competition-rules document. Include citations or links to primary sources wherever possible. Flag every uncertainty explicitly.

TRACK B — ANALYSIS OF TOP PUBLIC SOLUTIONS

Research the strongest publicly known ARC-AGI-3 approaches, including but not limited to:

- Tufa Labs’ Duck harness.
- The ARC Prize Milestone 1 winning solutions.
- High-performing community agents.
- Open-source ARC-AGI-3 repositories.
- Public technical write-ups.
- Kaggle discussion threads.
- Any available replay logs, benchmark results, or architecture descriptions.

For each notable solution, extract:

- Base model and model size.
- Quantization format.
- Inference engine.
- Harness architecture.
- Prompting strategy.
- Memory design.
- Context-window strategy.
- Compaction method.
- Tool use.
- Code execution or REPL use.
- Exploration policy.
- Planning strategy.
- Failure-recovery mechanism.
- Action-budget management.
- Reported public/local score.
- Reported leaderboard score, if available.
- Evidence of generalization versus overfitting.
- Weaknesses and failure modes.
- Ideas worth adapting in a game-agnostic way.

Do not copy game-specific tricks. Extract general principles.

Produce a competitive-landscape document with a comparison table and a ranked list of the most transferable ideas.

TRACK C — HARNESS AND MEMORY RESEARCH

Research state-of-the-art agent harness design for long-horizon interactive environments.

Investigate:

- Retained reasoning.
- Context compaction.
- Semantic memory.
- Episodic memory.
- Working memory.
- Retrieval memory.
- Structured world models.
- Object-centric state representations.
- Action-effect models.
- Hypothesis tracking.
- Bayesian or uncertainty-aware exploration.
- Curiosity and novelty-driven exploration.
- Intrinsic motivation.
- Tree search, MCTS, beam search, and best-first search.
- Reflexion-style self-critique.
- ReAct-style reasoning.
- Code-generation agents and Python REPL agents.
- Tool-using agents.
- Multi-agent orchestration.
- Hierarchical planning.
- Goal decomposition.
- Error recovery.
- Loop detection.
- Context pruning.
- Summarization policies.
- KV-cache management.
- Quantized KV cache.
- Prompt caching where legally and technically applicable.

Focus specifically on what applies to ARC-AGI-3: a sequence of small grid-based games with limited actions, no natural-language instructions, partial observability, novel mechanics, and a need for rapid online learning.

Produce a harness-design research document with concrete recommendations.

TRACK D — FLUID INTELLIGENCE AND EFFICIENT LEARNING

Research how to make an agent learn new rules with minimal interaction.

Study:

- Sample-efficient exploration.
- Information-gain maximization.
- Bayesian exploration.
- Active learning.
- Causal discovery from interventions.
- Object permanence and entity tracking.
- Program induction.
- Neurosymbolic reasoning.
- Graph-based world models.
- Relational reasoning.
- Symbolic abstraction.
- Constraint satisfaction.
- Planning under uncertainty.
- Meta-learning.
- In-context learning.
- Test-time adaptation.
- Self-generated curriculum.
- Hypothesis-space pruning.

Translate these ideas into practical ARC-AGI-3 mechanisms. Avoid vague theory. Every proposed idea must include a concrete implementation path and a way to test whether it improves untouched-game performance.

TRACK E — MODEL LANDSCAPE AND KAGGLE FEASIBILITY

Research which open-weight models are realistic for the competition.

Evaluate:

- Qwen-class models, especially around 27B.
- Other strong open-weight reasoning models.
- Multimodal versus text-only models.
- Model license compatibility.
- Kaggle GPU memory constraints.
- Quantization formats: GGUF, AWQ, GPTQ, FP8, INT8, Q4, Q5, and relevant KV-cache quantization.
- Inference engines: llama.cpp, vLLM, Transformers, MLX, ExLlama, and others.
- Tokens-per-second expectations on T4 x2.
- Context-length limitations.
- KV-cache memory consumption.
- Offline packaging requirements.
- Model-download and storage strategy.
- Tradeoffs between model quality, context length, speed, and action budget.

Produce a model-selection matrix with a clear recommendation for:

1. Best local M4 Air development model.
2. Best Kaggle development model.
3. Best likely final competition model.
4. Backup models if the primary choice fails.
5. Minimum viable model configuration.
6. Maximum realistic model configuration under Kaggle constraints.

======================================================================
4. CENTRAL DESIGN PHILOSOPHY
======================================================================

Build the system around a simple principle:

OBSERVE → INFER → REMEMBER → PLAN → ACT → VERIFY → REVISE

At every step, the agent should know:

- What it believes about the game.
- Why it believes it.
- What evidence supports that belief.
- What it does not know.
- What action would best reduce uncertainty or advance the goal.
- Whether its last action confirmed or contradicted its model.
- What it should remember and what it can safely forget.

The agent must not merely react to the latest frame. It must maintain a persistent internal model of the game.

======================================================================
5. REQUIRED HARNESS ARCHITECTURE
======================================================================

Design a modular harness with the following components.

COMPONENT 1 — OBSERVATION PROCESSOR

Convert the raw 64x64 frame into compact, useful representations without making game-specific assumptions.

It should extract or maintain:

- Raw frame.
- Frame diff from previous state.
- Changed-cell map.
- Candidate objects and sprites.
- Candidate background.
- Candidate walls or impassable regions.
- Candidate movable entities.
- Candidate goals or target regions.
- HUD regions, only if identified through general visual patterns rather than hardcoded coordinates.
- Symmetries.
- Repeated structures.
- Color clusters.
- Spatial relationships.
- Motion vectors.
- Persistent versus transient changes.

Important: do not hardcode color IDs, object roles, HUD locations, or game-specific layouts. All role assignments must be induced from observed transitions and revised when evidence contradicts them.

COMPONENT 2 — ACTION-EFFECT LEDGER

Maintain a structured record of every meaningful action:

```text
Action ID
Step
Level
Action taken
Pre-action state summary
Post-action state summary
Frame diff
Objects affected
Apparent effect
Confidence
Hypotheses supported
Hypotheses contradicted
Usefulness score
Outcome: informative / neutral / redundant / harmful
```

This ledger is the agent’s scientific notebook. It must allow the agent to answer:

- What happens when I perform this action in this kind of state?
- Which actions change the world?
- Which actions are redundant?
- Which actions are dangerous?
- Which hypotheses are supported or refuted?
- What should I try next?

COMPONENT 3 — WORLD MODEL

Maintain a compact, structured world model, updated after every action.

It should include:

- Current level and progress.
- Inferred game modality.
- Known objects and roles.
- Known static structures.
- Known dynamic mechanics.
- Known goals.
- Known win condition hypotheses.
- Known constraints.
- Known action semantics.
- Current uncertainty.
- Active hypotheses.
- Evidence for and against each hypothesis.
- Predicted next state.
- Prediction error.

The world model must be compact enough to fit in limited context but rich enough to support planning.

COMPONENT 4 — HYPOTHESIS MANAGER

The agent must explicitly maintain hypotheses such as:

- “This object is the player.”
- “This object is a goal.”
- “This action moves the player one cell.”
- “This action toggles neighboring cells.”
- “This object disappears when touched.”
- “This color indicates a wall.”
- “This game requires visiting all targets.”
- “This game requires matching hidden identities.”
- “This action has no effect in this state.”

Each hypothesis should have:

- A statement.
- Prior probability or confidence.
- Supporting evidence.
- Contradicting evidence.
- A proposed test.
- Expected observations if true.
- Expected observations if false.
- Status: untested, active, confirmed, rejected, uncertain.

The agent should prefer cheap experiments that maximally distinguish between competing hypotheses.

COMPONENT 5 — EXPLORATION POLICY

Early in a game, the agent should explore intelligently.

It should prioritize actions that:

- Change the frame.
- Reveal hidden information.
- Test an important hypothesis.
- Produce a novel state.
- Resolve uncertainty.
- Are reversible or low-risk.
- Avoid immediate failure.
- Avoid repeating known-useless actions.

It should avoid:

- Repeating identical actions in unchanged states.
- Random-walking after a rule is understood.
- Re-testing actions whose effects are already known.
- Exhausting the action budget on uninformative moves.
- Continuing a failed strategy after clear contradiction.

COMPONENT 6 — PLANNER

Once the agent has sufficient understanding, it should switch from exploration to goal-directed planning.

The planner should support:

- Pathfinding.
- Waypoint collection.
- Push-box planning.
- Sliding mechanics.
- Multi-object coordination.
- Ordering constraints.
- Resource constraints.
- Conditional gates.
- Hidden-state dependencies.
- Dynamic obstacles.
- Replanning after unexpected transitions.

The planner must use the current world model, not hardcoded assumptions.

COMPONENT 7 — MEMORY AND CONTEXT MANAGER

This is one of the most important components.

The context manager must decide:

- What belongs in immediate context.
- What belongs in durable memory.
- What can be compressed.
- What can be discarded.
- What must never be discarded.
- How to summarize prior exploration.
- How to preserve active hypotheses.
- How to retain the current plan.
- How to retain recent frame diffs.
- How to avoid context overflow.
- How to keep prompts short enough for fast inference.

It should maintain separate layers:

1. Immediate context: recent frames, recent actions, recent diffs, current plan.
2. Working memory: active hypotheses, current goals, current uncertainty.
3. Episodic memory: compressed history of important events.
4. Semantic memory: generalized rules inferred across levels or games.
5. Meta-memory: what strategies worked, what failed, and why.

Do not blindly truncate history. Blind truncation destroys exactly the information an agent needs to learn.

COMPONENT 8 — RETAINED REASONING / STATE CONTINUITY

If the selected model and Kaggle environment support retained reasoning, use it.

If not, simulate it with an explicit reasoning-state store:

- Current beliefs.
- Current plan.
- Current subgoal.
- Last reflection.
- Open questions.
- Confirmed rules.
- Uncertainties.
- Next intended action.

The agent should not have to rediscover everything after every action.

COMPONENT 9 — SELF-CRITIQUE AND RECOVERY

After important actions or failed predictions, the agent should briefly reflect:

- Did the action do what I expected?
- If not, why?
- Which hypothesis is wrong?
- What is the cheapest test to resolve the discrepancy?
- Should I revise the world model?
- Should I abandon the current plan?
- Am I in a loop?
- Am I wasting actions?

Self-critique must be concise. Long reflections waste tokens and time.

COMPONENT 10 — ACTION VALIDATION AND ERROR RECOVERY

The harness must:

- Validate proposed actions.
- Prevent invalid actions where possible.
- Retry safely after malformed output.
- Detect repeated identical outputs.
- Detect no-progress loops.
- Detect excessive token usage.
- Detect approaching action or time limits.
- Force a strategy change when necessary.
- Preserve enough memory to avoid repeating the same failure.

======================================================================
6. PROMPTING AND CONTEXT DESIGN
======================================================================

Prompts must be short, structured, and consistent.

Avoid giant prompts that repeat static instructions every turn. Separate:

- Stable system instructions.
- Compact game-state summary.
- Active hypotheses.
- Recent observations.
- Current plan.
- Immediate decision request.

The model should usually be asked for one of the following:

1. Update the world model.
2. Choose the next action.
3. Reflect on an unexpected result.
4. Revise a failed plan.
5. Summarize progress when context is becoming large.

Do not ask the model to produce enormous reasoning every turn unless the extra reasoning has a clear expected benefit.

Token efficiency is competition-critical. A 27B model on T4 x2 may produce only around 10–15 tokens per second in some setups. Every unnecessary token delays the agent and consumes the runtime budget.

======================================================================
7. MODEL STRATEGY
======================================================================

Use a staged model strategy.

STAGE 1 — LOCAL DEVELOPMENT ON M4 AIR

Use a small open-weight quantized model to develop and debug the harness.

Likely candidates:

- 7B to 14B instruct or reasoning models.
- Q4 or Q5 quantization.
- LM Studio or Ollama for local inference.
- Focus on speed, reproducibility, and harness correctness.

The local model does not need to be the final competition model. Its job is to validate the harness architecture, memory format, prompts, tools, and evaluation loop.

STAGE 2 — KAGGLE DEVELOPMENT

Use a 27B-class open-weight model in Q4/Q4_K_M GGUF or another suitable quantization.

Target:

- Model split across T4 x2.
- Flash attention enabled.
- Quantized KV cache if needed.
- Careful context-length management.
- Offline model packaging.
- Reproducible submission notebook.

STAGE 3 — FINAL MODEL SELECTION

Choose the final model based on measured performance across:

- Untouched-game win rate.
- Levels cleared.
- Official score.
- RHAE.
- Actions per completed level.
- Token cost per action.
- Runtime per game.
- Stability across repeated runs.
- Ability to follow the harness memory format.
- Ability to reason about visual state and causal transitions.

Do not choose a model because it is fashionable. Choose it because it produces the best transferable competition results under Kaggle constraints.

======================================================================
8. RESEARCH AND EXPERIMENTATION PLAN
======================================================================

You should operate in iterative research sprints.

SPRINT 1 — RECONNAISSANCE

Deliverables:

- Competition-rules document.
- Evaluation-mechanics document.
- Top-solution analysis.
- Harness research summary.
- Model feasibility matrix.
- Initial architecture proposal.
- Risk register.
- Experiment backlog.

SPRINT 2 — MINIMUM VIABLE GENERAL HARNESS

Build:

- Game loop.
- Observation processor.
- Frame-diff tracker.
- Action-effect ledger.
- Basic world model.
- Basic hypothesis manager.
- Basic exploration policy.
- Basic planner.
- Context manager.
- Evaluation logger.
- Replay viewer.
- Local model integration.
- Kaggle-compatible inference path.

Evaluate on a small untouched development cohort.

SPRINT 3 — MEMORY AND COMPACTION

Implement:

- Durable world-model memory.
- Episodic memory compression.
- Semantic rule memory.
- Context prioritization.
- Intelligent summarization.
- Retained reasoning or explicit reasoning-state store.
- Memory-ablation experiments.

Measure whether better memory improves untouched-game completion and RHAE.

SPRINT 4 — INTELLIGENT EXPLORATION

Implement:

- Novelty detection.
- Information-gain scoring.
- Loop detection.
- Hypothesis-testing action selection.
- Exploration/exploitation switching.
- Action-budget awareness.

Measure whether the agent learns rules with fewer actions.

SPRINT 5 — PLANNING AND EFFICIENCY

Implement:

- General pathfinding.
- Goal decomposition.
- Multi-step planning.
- Replanning.
- Resource-aware planning.
- Dynamic-obstacle handling.
- Efficient execution.

Measure official score and RHAE improvements.

SPRINT 6 — MODEL AND HARNESS CO-OPTIMIZATION

Test:

- Multiple open-weight models.
- Multiple quantizations.
- Multiple context lengths.
- Multiple compaction policies.
- Multiple prompt formats.
- Multiple reasoning budgets.
- Multiple tool configurations.

Select the configuration with the best held-out transfer, not the best public-set score.

SPRINT 7 — COMPETITION HARDENING

Ensure:

- Fully offline execution.
- Reproducible environment.
- Robust error handling.
- Model weights packaged correctly.
- Runtime budget managed.
- No accidental internet calls.
- No accidental source-file access.
- No game-specific branches.
- Clean logging.
- Submission notebook tested end-to-end.

======================================================================
9. EVALUATION PROTOCOL
======================================================================

Every claim of progress must follow this protocol.

1. Define the hypothesis.
2. Freeze the current agent.
3. Select a held-out cohort that has never been opened, inspected, or used for development.
4. Run each game once under the official observation/action interface.
5. Do not edit anything until the full run completes.
6. Record per-game results.
7. Analyze failures by general capability, not game ID.
8. If the result improves, validate on a second fresh untouched cohort.
9. Only then consider the improvement real.
10. Only then consider submitting or scaling the change.

Every experiment report must include:

- Hypothesis.
- Change made.
- Model and quantization.
- Context length.
- Memory configuration.
- Number of games.
- Number of untouched games.
- Win rate.
- Levels cleared.
- Mean official score.
- Mean RHAE.
- Mean actions per completed level.
- Runtime.
- Token usage.
- Failure taxonomy.
- Evidence of transfer.
- Evidence of overfitting.
- Decision.

======================================================================
10. FAILURE TAXONOMY
======================================================================

Classify every failure into one or more general categories:

1. Perception failure: agent misread objects, background, HUD, goals, or changes.
2. Object-role failure: agent misclassified player, wall, goal, pushable object, hazard, or interactive element.
3. Action-semantics failure: agent misunderstood what an action does.
4. Causal-model failure: agent failed to predict state transitions.
5. Memory failure: agent forgot a discovered rule or repeated a failed strategy.
6. Context failure: important information was truncated, compressed badly, or never stored.
7. Exploration failure: agent repeated useless actions or failed to test key hypotheses.
8. Planning failure: agent understood the rules but chose an inefficient or invalid plan.
9. Goal-inference failure: agent did not understand the win condition.
10. Dynamic-environment failure: agent failed to model changes not caused by its own action.
11. Hidden-state failure: agent failed to track concealed information.
12. Efficiency failure: agent solved the game but used too many actions.
13. Runtime failure: agent ran out of time or tokens.
14. Harness bug: infrastructure, parsing, memory, or action-execution error.
15. Overfitting risk: improvement depends on a specific game, layout, palette, or fingerprint.

Never describe a failure merely as “the model failed.” Identify the missing general capability.

======================================================================
11. KNOWLEDGE-BASE DELIVERABLES
======================================================================

Create and continuously update a durable project knowledge base so that a future agent, such as Gemini 3.8 Flash High, can continue without repeating research.

Required files:

1. `COMPETITION_RULES.md`
   - Exact rules, constraints, scoring, runtime, hardware, submission requirements.

2. `RESEARCH_LANDSCAPE.md`
   - Analysis of Duck, Tufa Labs, Milestone winners, and other strong public approaches.
   - Transferable lessons.
   - Weaknesses and failure modes.

3. `HARNESS_DESIGN.md`
   - Full architecture.
   - Module responsibilities.
   - Data schemas.
   - Memory design.
   - Context policy.
   - Prompt templates.
   - Tool interfaces.
   - Planning strategy.
   - Error recovery.

4. `MODEL_STRATEGY.md`
   - Candidate models.
   - Quantization options.
   - Kaggle feasibility.
   - Local M4 Air feasibility.
   - Recommended primary and backup models.
   - Memory and context requirements.
   - Expected speed and tradeoffs.

5. `EXPERIMENT_LOG.md`
   - Every experiment.
   - Hypothesis.
   - Setup.
   - Results.
   - Interpretation.
   - Decision.
   - Next steps.

6. `FAILURE_TAXONOMY.md`
   - Recurring failure modes.
   - General capability gaps.
   - Proposed general fixes.
   - Validation status.

7. `EVALUATION_PROTOCOL.md`
   - Frozen evaluation procedure.
   - Held-out cohort rules.
   - Metrics.
   - Reporting standard.
   - Anti-overfitting checklist.

8. `ROADMAP.md`
   - Current state.
   - Next milestones.
   - Highest-leverage tasks.
   - Open questions.
   - Risks.
   - Submission plan.

These documents must be concise, precise, and updated after every major experiment. A future agent must be able to read them and immediately understand the current architecture, evidence, constraints, and next steps.

======================================================================
12. ANTI-CHEATING AND ANTI-OVERFITTING CONTRACT
======================================================================

You must strictly obey these rules.

Prohibited:

- Reading, opening, grepping, summarizing, or analyzing any environment `.py`, `.json`, metadata, level definition, transcript, or internal game state for an evaluation game.
- Using game IDs, filenames, exact pixel coordinates, color palettes, HUD layouts, or level layouts to select behavior.
- Creating per-game branches, bespoke solvers, hardcoded waypoints, hardcoded color-role mappings, or fingerprint checks.
- Disguising game-specific checks as general checks.
- Repeatedly editing after evaluating the same games and then claiming improved generalization.
- Using public-set failure analysis to create fixes targeted only at those exact games.
- Claiming that a public-set improvement proves leaderboard improvement.

Allowed:

- Learning exclusively from official observations and action feedback.
- Building general mechanisms that transfer across game families.
- Using development games for debugging, provided improvements are validated on untouched games.
- Studying public solutions and research for general design principles.
- Running controlled experiments and ablations.

Before implementing any fix, ask:

1. Does this depend on a specific game, layout, color scheme, or HUD?
2. Could this mechanism transfer to a completely new game?
3. Am I using information unavailable through official gameplay?
4. Am I optimizing the evaluation set rather than general capability?
5. How will I prove this transfers?

If the answer reveals game-specific dependence, reject or redesign the fix.

======================================================================
13. HOW TO MAXIMIZE SCORE AND RHAE
======================================================================

To maximize score, prioritize:

1. Fixing systematic perception and object-role errors.
2. Building reliable action-effect inference.
3. Preventing context loss and repeated exploration.
4. Detecting and escaping loops.
5. Inferring win conditions earlier.
6. Switching from exploration to planning sooner.
7. Producing concise, efficient action plans.
8. Avoiding timeouts and action-budget exhaustion.
9. Improving transfer to unseen mechanics.
10. Reducing tokens and runtime per action.

To maximize RHAE, prioritize:

1. Minimal sufficient exploration.
2. High-confidence rule induction.
3. Reuse of learned rules across levels.
4. Direct path planning once the goal is known.
5. Avoiding redundant actions.
6. Avoiding unnecessary re-observation.
7. Compact but sufficient reasoning.
8. Fast convergence from exploration to exploitation.

Remember: RHAE rewards solving levels with fewer actions than the human baseline. Completion alone is not enough. The best agent is both effective and efficient.

======================================================================
14. IMMEDIATE EXECUTION ORDER
======================================================================

Begin now in this order:

1. Create the project knowledge base.
2. Deploy parallel research subagents for:
   - Competition rules.
   - Top-solution analysis.
   - Harness/memory research.
   - Fluid-intelligence and sample-efficient learning research.
   - Model and Kaggle feasibility.
3. Synthesize their findings into a single research brief.
4. Produce a detailed harness architecture.
5. Produce a model-selection recommendation.
6. Produce a prioritized experiment backlog.
7. Build the minimum viable general harness.
8. Evaluate on untouched games.
9. Iterate only through controlled, transferable improvements.
10. Maintain the knowledge base after every major result.
