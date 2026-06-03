# NCCL IB Merge Auto Progress Snapshot

This file is a recovery checkpoint for the two-node IB NIC merge auto-selection work.
It records what has already been committed and what remains, so a future session can
resume without relying on chat context.

## Current State

Latest feature commit before this snapshot:

- `2a1dbb7 Inline IB vNIC props matching helpers`

Implemented commits:

- `66a1abe Make IB vNIC creation idempotent`
  - IB vNIC creation is find-or-create.
  - `NCCL_IB_MERGE_NICS=0` still rejects multi-device vNICs.
  - `NCCL_IB_MERGE_NICS=2` can create/reuse multi-device vNICs.

- `fc099e1 Filter merged IB vNICs for unmerged topology`
  - Topology import skips merged IB vNICs for explicit unmerged mode.

- `e436501 Expose topology net merge view import`
  - Added `ncclNetMergeView`.
  - Added `ncclTopoProcessNetWithMergeView(...)`.

- `78a416f Add topology XML merge view filtering`
  - Added XML view copy/filter helper.
  - Added `vndevs` metadata on NET XML nodes.

- `b75aecb Add IB merge auto mode helpers`
  - `NCCL_IB_MERGE_NICS=2` is the auto mode.
  - Added threshold/dump params:
    - `NCCL_IB_MERGE_NICS_AUTO_THRESHOLD`, default `110`
    - `NCCL_IB_MERGE_NICS_AUTO_DUMP`, default `0`

- `8142b7a Add merge auto synthetic metrics helpers`
  - Added pure helper data structures.
  - Added host-hash two-node map builder.
  - Added channel-ring cross-edge extraction.
  - Added rail-dedup metrics aggregation.
  - Added threshold selector.

- `56134c3 Dump merge auto graph ring order`
  - Added graph ring extraction from `ncclTopoGraph::intra`.
  - Added `MergeAutoDump: cand=default ch=xx ring=...` when dump is enabled.
  - Connected read-only dump after default ring graph search.

- `dc6c53f Add merge auto runtime gating logs`
  - Builds a two-node map from `comm->peerInfo[].hostHash`.
  - Adds `ncclMergeAutoCheckRuntime(...)`.
  - Logs enabled/skip decisions only when `NCCL_IB_MERGE_NICS=2`.
  - Covers skip reasons:
    - not_two_nodes
    - non_ib_net
    - num_net_devs_lt_2
    - nranks_unsupported
  - Does not run candidate graph search or change final behavior.

- `b90b977 Dump merge auto default graph cross edges`
  - Reuses the computed default ring graph.
  - Builds the two-node map from `comm`.
  - Extracts cross-node edges from ring order.
  - Dumps:
    - `MergeAutoDump: cand=default ch=xx edge=src->dst dir=a->b`
  - Does not resolve netDev/HCA yet.
  - Does not build alternate candidates yet.

- `b9211bd Resolve merge auto dump edge rails`
  - Uses `ncclTopoGetNetDev(...)` as a dry-run resolver for dumped edges.
  - Reads `comm->ncclNet->getProperties(netDev).vProps`.
  - Adds physical rail ids to dump lines.
  - Uses unit bandwidth for physical rails.
  - Falls back to `net=unknown` if the dry-run resolver cannot map an edge.
  - Does not call transport listen/connect/accept.

- `c5aa6e9 Dump merge auto postset ring edges`
  - Dumps cross-node edges derived from postset ring endpoints.
  - Uses `ringSend[node] -> ringRecv[nextNode]`.
  - Runs after `ncclTopoPostset(...)`, so odd-node endpoint swaps have already been applied.
  - Only the first rank of each node emits its node direction to avoid per-rank duplicate dumps.
  - Reuses the dry-run netDev/physical rail resolver.
  - Does not change postset, channel selection, or transport setup.

- `2a1dbb7 Inline IB vNIC props matching helpers`
  - Removed the small vProps canonicalize/compare helpers.
  - Folded the same logic into `ncclIbFindOrMakeVDeviceInternal(...)`.
  - Behavior is unchanged.

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

1. Postset edge metrics / aggregation strategy
   - Current postset dump resolves each node direction on that node's first rank.
   - Real metrics need a deterministic way for all ranks to see both directions:
     - gather per-node resolved edge rail sets, or
     - expose enough graph/net metadata to resolve remote node edges consistently.
   - Keep the next step as default-candidate metrics before building alternate candidates.

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

Implement default-candidate postset metrics:

- Aggregate the resolved postset edge rails for the default candidate.
- Keep it dump/summary-only.
- Do not build alternate candidates yet.
- Do not change selection behavior.
