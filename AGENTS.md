<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

# ARC-AGI-3 DEVELOPMENT CONTRACT: TRUE FLUID-INTELLIGENCE AGENT

You are building an exceptionally intelligent, exceptionally efficient, genuinely general ARC-AGI-3 agent. The goal is not to solve a few known games. The goal is to build a fluid-intelligence system that can enter any unfamiliar interactive environment, infer its rules from interaction, model the world, plan efficiently, recover from mistakes, and achieve the highest possible competition score and Relative Human Action Efficiency (RHAE).

## CORE OBJECTIVE
Build one general agent that can:
1. Observe only official frames, available actions, chosen actions, resulting frames, level-completion signals, and action budgets.
2. Rapidly infer objects, affordances, causal rules, hidden state, goals, and win conditions.
3. Maintain a persistent, revisable world model across actions and levels.
4. Explore intelligently: prioritize informative actions, avoid repetition, and learn from minimal evidence.
5. Plan efficiently and execute near-human or superhuman action efficiency.
6. Detect failed predictions, revise hypotheses, and adapt online.
7. Generalize to completely novel mechanics, including both extremely simple and extremely complex games.

The standard is not “beat these games.” The standard is: learn unfamiliar games quickly, finish reliably, and do so with as few actions as possible.

## LEADERBOARD TRANSFER IS MANDATORY
Local and public scores are development signals, not final achievements.
- The public demo set has only 25 games.
- Competition evaluation uses 110 separate hidden games.
- Half of the hidden games determine the Public Leaderboard; the other half determine the final Private Leaderboard.
- A strong local/public score does not guarantee a strong leaderboard score.
- Community evidence shows large public-to-leaderboard gaps; leaderboard results have sometimes been roughly 25–45% of local/public results.
- Never optimize merely for the public games, a small local cohort, or a particular game family.
- Every meaningful improvement must be validated for transfer: first on a fresh untouched held-out cohort, then on broader evaluation.

When reporting results, always distinguish:
1. Local development score.
2. Public 25-game score.
3. Public leaderboard score.
4. Final private leaderboard score.
Never imply these are the same.

## EXPERIMENT AGGRESSIVELY, BUT SCIENTIFICALLY
Experiment freely. Test architectures, exploration policies, memory designs, planners, model sizes, and tooling setups. Use research subagents, coding subagents, ablations, repeated seeds, and deep theoretical analysis.

Every experiment must have:
- A clear hypothesis.
- A general mechanism being tested.
- A controlled comparison.
- A frozen evaluation protocol.
- A held-out validation cohort.
- Per-game results and failure analysis.
- A decision: adopt, reject, or iterate.

A change is only progress if it improves generalization on untouched games, improves action efficiency, or improves robustness across diverse mechanics.

## ANTI-OVERFITTING IS NON-NEGOTIABLE
Never build an agent for the public games.

Prohibited:
- Reading, opening, grepping, summarizing, or analyzing environment .py files, JSON, metadata, level definitions, transcripts, or internal game state for evaluation games.
- Using game IDs, filenames, exact pixel coordinates, color palettes, HUD layouts, or level layouts to select behavior.
- Creating per-game branches, bespoke solvers, hardcoded waypoints, hardcoded color-role mappings, or fingerprint checks.
- Disguising game-specific checks as general checks, such as replacing `if game_id == "bn01"` with pixel fingerprints.
- Repeatedly editing after evaluating the same games and then claiming improved generalization.
- Designing “general” fixes that are actually detectors for named games, families, pixel layouts, or known failure cases.

Allowed:
- Learning exclusively from the official observation/action interface.
- General perception, object-role induction, causal-effect testing, novelty exploration, memory, planning, search, hypothesis testing, and self-correction.
- Using development games for debugging, provided lessons are generalized and validated on untouched games.

## EVALUATION PROTOCOL
Before claiming any result:
1. Freeze the agent completely.
2. Use a held-out cohort whose source files, metadata, and internals have never been accessed.
3. Run each game once through the official interface.
4. Do not edit code, prompts, subagents, or strategy until the entire run finishes.
5. Report per-game completion, levels cleared, actions used, official score, RHAE, runtime, and failure diagnosis.
6. Validate any improvement on a new untouched cohort before treating it as real progress.
7. Run multiple seeds when computationally feasible to distinguish signal from variance.

A result only counts as general progress if it transfers beyond the games or failure modes that motivated it.

## INTELLIGENCE AND EFFICIENCY
Do not build a blind testing idiot.
The agent must balance:
- Exploration: discovering rules with minimal, informative probes.
- Exploitation: executing the inferred policy efficiently.
- Memory: retaining discovered rules, object roles, hidden state, and failed hypotheses.
- Planning: choosing actions that maximize expected progress or information.
- Self-correction: noticing when its world model is wrong and revising it.

RHAE matters. The agent should aim to match or beat human action efficiency wherever possible.

## DEVELOPMENT PRIORITIES
Improve these general capabilities:
1. Perception: separate background, tiles, sprites, HUD, goals, obstacles, and transient effects without fixed color assumptions.
2. Object-role induction: infer avatar, pushable objects, walls, goals, collectibles, hazards, and interactive elements from behavior.
3. Action-semantics induction: determine whether actions move, push, slide, toggle, cycle, reveal, paint, connect, or do nothing.
4. Causal world modeling: predict state transitions and detect prediction errors.
5. Partial-observability memory: remember hidden identities, revealed information, cooldowns, and tested hypotheses.
6. Dynamic environments: model changes not caused by the agent, such as conveyors, falling hazards, ice, trails, and timers.
7. Goal inference: infer reach-exit, collect-all, match-all, cover-all, connect-endpoints, ordering, and constraint-satisfaction objectives.
8. Efficient planning: maximize information early, then minimize actions once the rule is understood.
9. Failure recovery: recognize loops, dead ends, timeouts, and invalid assumptions; change strategy instead of repeating.

## USE OF SUBAGENTS AND RESEARCH
Deploy research and coding subagents liberally, but give each one:
- A precise general question.
- Access only to legitimate development data and prior transition logs.
- A requirement to propose mechanism-level, game-agnostic solutions.
- A requirement to predict how the change should improve untouched-game performance.
- A requirement to identify how the change could overfit.

Extract general principles from winning solutions: better exploration, state abstraction, memory, planning, uncertainty handling, and action efficiency.

## SUCCESS CRITERIA
You are succeeding when:
- The agent solves unfamiliar games without source access.
- It learns rules from few interactions.
- It generalizes across simple and complex mechanics.
- It avoids loops, timeouts, and wasted actions.
- It improves RHAE as well as completion.
- Improvements transfer to fresh untouched cohorts.
- The public/leaderboard gap is understood and minimized.
- The system increasingly behaves like true fluid intelligence rather than a set of specialized solvers.
