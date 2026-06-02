# NCCL IB Merge Auto Progress Snapshot

This file is a recovery checkpoint for the two-node IB NIC merge auto-selection work.
It records what has already been committed and what remains, so a future session can
resume without relying on chat context.

## Current State

Latest feature commit before this snapshot:

- `f906595 Dump merge auto default graph cross edges`

Implemented commits:

- `66a1abe Make IB vNIC creation idempotent`
  - IB vNIC creation is find-or-create.
  - `NCCL_IB_MERGE_NICS=0` still rejects multi-device vNICs.
  - `NCCL_IB_MERGE_NICS=2` can create/reuse multi-device vNICs.

- `0a9c379 Filter merged IB vNICs for unmerged topology`
  - Topology import skips merged IB vNICs for explicit unmerged mode.

- `3964cb2 Expose topology net merge view import`
  - Added `ncclNetMergeView`.
  - Added `ncclTopoProcessNetWithMergeView(...)`.

- `33e9e7b Add topology XML merge view filtering`
  - Added XML view copy/filter helper.
  - Added `vndevs` metadata on NET XML nodes.

- `92b4e91 Add IB merge auto mode helpers`
  - `NCCL_IB_MERGE_NICS=2` is the auto mode.
  - Added threshold/dump params:
    - `NCCL_IB_MERGE_NICS_AUTO_THRESHOLD`, default `110`
    - `NCCL_IB_MERGE_NICS_AUTO_DUMP`, default `0`

- `62e8460 Add merge auto synthetic metrics helpers`
  - Added pure helper data structures.
  - Added host-hash two-node map builder.
  - Added channel-ring cross-edge extraction.
  - Added rail-dedup metrics aggregation.
  - Added threshold selector.

- `e9484ac Dump merge auto graph ring order`
  - Added graph ring extraction from `ncclTopoGraph::intra`.
  - Added `MergeAutoDump: cand=default ch=xx ring=...` when dump is enabled.
  - Connected read-only dump after default ring graph search.

- `35fbe37 Add merge auto runtime gating logs`
  - Builds a two-node map from `comm->peerInfo[].hostHash`.
  - Adds `ncclMergeAutoCheckRuntime(...)`.
  - Logs enabled/skip decisions only when `NCCL_IB_MERGE_NICS=2`.
  - Covers skip reasons:
    - not_two_nodes
    - non_ib_net
    - num_net_devs_lt_2
    - nranks_unsupported
  - Does not run candidate graph search or change final behavior.

- `f906595 Dump merge auto default graph cross edges`
  - Reuses the computed default ring graph.
  - Builds the two-node map from `comm`.
  - Extracts cross-node edges from ring order.
  - Dumps:
    - `MergeAutoDump: cand=default ch=xx edge=src->dst dir=a->b`
  - Does not resolve netDev/HCA yet.
  - Does not build alternate candidates yet.

## Verification So Far

- `git diff --check` passed for each committed step.
- A temporary stdin synthetic harness was run for:
  - two-node host hash map
  - ring cross-edge extraction
  - merge=1 / merge=0 rail metrics
  - selector choosing merge=0 when score is 16 vs 8
- Full/object build is blocked in this local environment by missing CUDA headers:
  - `fatal error: 'cuda_runtime.h' file not found`

## Important Constraints To Preserve

- Default behavior must remain unchanged for `NCCL_IB_MERGE_NICS=1`.
- Explicit `NCCL_IB_MERGE_NICS=0` must never see merged vNICs in topology.
- `NCCL_IB_MERGE_NICS=2` is the only auto-selection entry.
- Do not mutate environment variables to build candidates.
- Do not reset or destroy IB plugin global vNIC cache.
- Do not create two rounds of real transport connections.
- First real selection version is two-node only.

## Next Work Items

1. netDev / physical rail resolve
   - Prefer graph/topology metadata.
   - Likely starting point: `ncclTopoGetNetDev(...)` in `src/graph/search.cc`.
   - Avoid calling transport listen/connect/accept.
   - Use physical rails from vNIC metadata where available; otherwise unit fallback.

2. Candidate dry-run
   - Build/import superset topology.
   - Copy/filter into:
     - `NCCL_NET_MERGE_VIEW_UNMERGED`
     - `NCCL_NET_MERGE_VIEW_MERGED_DEFAULT`
   - Run graph search for both.
   - Print candidate metrics.
   - Keep final behavior default at first.

3. Selection and commit
   - Score is `bidirBw`.
   - Threshold default is 110.
   - If merge0 is clearly higher, select unmerged.
   - Otherwise keep merged/default.
   - Candidate failure falls back to merged/default.

4. Docs/tests
   - Turn the temporary synthetic harness into a repo-local test or script.
   - Add manual validation commands and expected logs.

## Suggested Next Commit

Implement netDev/physical rail resolve for dumped default-graph edges:

- Find where graph search records or can recompute the selected NET device for a rank edge.
- Add a resolver that does not call transport listen/connect/accept.
- Fill `edge.netDev`, `edge.nPhysRails`, and rail bandwidth fields where possible.
- Extend dump lines with `net=IB/x phys={...} bw=...`.
- Do not build alternate candidates yet.
