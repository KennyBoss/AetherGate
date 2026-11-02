# ECS Stage 3 State-Porting Implementation Plan

1. Implement `research/skill_compression/ecs_stage3_state_porting.py` by reusing
   Stage-2 task primitives, features, gate, and best-first accounting.
2. Train recurrent-context policies on depth-1/2 trajectories with start states
   `0..9`; evaluate depth-1/2/3 on start states `10..19` to guarantee semantic
   train/eval disjointness.
3. Run the built-in two-seed self-check and fix only wiring/protocol failures.
4. Run the fixed six-seed experiment and preserve PASS, FAIL, or INVALID as-is.
5. Record the result in `research/MODEL_THESIS.md`; do not start a later stage.
