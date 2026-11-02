Branch-guided pair discovery (2026-11-01)

Run from the repository root:
  python3 research/codepy/subtask_discovery/branch_guided/test_guided.py
  python3 research/codepy/subtask_discovery/branch_guided/run_guided.py
  python3 research/codepy/subtask_discovery/branch_guided/verify_guided.py

Existing protocol/source hashes are enforced on resume. For a changed implementation,
use a fresh output directory and protocol rather than replacing results. common.OUT
currently points to artifacts/codepy_branch_guided_20261101. Previous source modules
and all historical artifacts are read-only dependencies.

Diagnosis: diagnose.py replays the unchanged old mining procedure for every seed,
asserts the old logs/bodies match, and records per-order value preservation, live
conditions and number of distinct elite behaviors. Seed3 starts from a much better
value-preserving candidate; the other runs still contain conditions, but have fewer
useful behaviors and collapse to clones. Identity/swap/zero all score2 under the old
scalar order loss. Seed1's old pair library is empty, not an existing swap library.

Intervention: extract generic register-output backward slices of every source prefix,
and windows with live input-dependent predicates. Returning computed registers avoids
requiring the fragment to store its result in the original tape locations within the
same12-command cap. Outputs are explicit ABI metadata:0..2 local cells,3/4 registers.
All other state resets on every call. No new arithmetic primitive or opcode35 sort2.

25 fixed discovery pairs are scored by each strict input order independently; preserve
both values and equal-input behavior. Worst-order loss exposes improvements masked by
one easy orientation. Maintain distinct output behaviors in the16 elites; reserve niches
for both orientations, live branches and partial-order specialists. Half offspring inherit
body and layout together, protecting only live input-dependent compare/conditional
instructions when the parent already has some exact successes in both orders and at
least75% exact multiset preservation. Other offspring remain free to explore. Conditions
alone do not earn protection. Complete blocks qualify on64challenge+6429promotion pairs
in native and independent NumPy executors. These evaluations are included in mining cost.

Fresh downstream search uses the same64TRAIN cases, primitive maximum72steps, population256,
seeds0..4, six pair-call slots. Prior pair blocks may be retained in their original seed;
no final target champions or cross-seed bodies initialize a search. Total99evaluated
populations per task, split49arguments+25oldpairs+25neworder when both pair branches exist,
49+50 when one exists,99arguments when no pair exists. This rule is task-independent.
A new order-only branch avoids changing the old pair alphabet and perturbing its useful
trajectory. Minimum TRAIN objective chooses the branch before all7041 final cases.

Each of four targets saves256candidate evaluations from the accepted prior25600 budget:
4*256*64*72=4718592 primitive-case steps. Extra mining is <= old per-seed ablation cap,
which is <= those savings. All prior source/old extraction/qualification/preparation costs
remain charged to both complete pipelines. Static analysis, ABI overhead, compilation,
and final independent certificates are outside this historical VM-step metric; physical
FLOPs, wall time and memory are not claimed equal. This is a combined extraction/ABI/loss/
selection intervention, not a causal ablation establishing which individual change helped.

Verification replays all20new and20baseline saved winners on all7041inputs, requalifies
new blocks, checks training hashes, branch prefixes, budget sums, source hashes, and
unchanged historical artifact hashes. The tests exercise generic ABI/slicing/mutation
properties; no executable comparator body is supplied to discovery. Results on finite
float32 inputs are not a symbolic proof over every real-valued input.
