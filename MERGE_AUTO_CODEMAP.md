# NCCL IB Merge Auto Code Map

Scope: Module 0 read-only source mapping for two-node NIC merge auto-selection.
No code behavior is changed by this document.

## Key Findings

- `NCCL_IB_MERGE_NICS` is not used inside ring search directly. It gates IB virtual NIC creation during topology import. Graph search sees the resulting NET nodes after physical/virtual NIC population.
- The final global ring rank order is available in `rings` inside `ncclTopoPostset()`, before real transport connections are established.
- `NET/IB/x` is resolved by `ncclTopoGetNetDev()` and logged in `transport/net.cc` during send/recv setup. The same resolver can be reused for dry-run metadata without creating extra real connections if a valid graph and channel id are available.
- Existing node mapping is already stored in `comm->nNodes`, `comm->rankToNode`, and `comm->nodeRanks`; do not derive nodes from rank arithmetic.

## env param location

- `src/transport/net_ib/init.cc:29`
  Defines `NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1)`.

- `src/transport/net_ib/init.cc:174`
  `ncclIbMakeVDeviceInternal()` checks `ncclParamIbMergeNics() == 0 && props->ndevs > 1`.
  If true, it logs `NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0` and rejects multi-device vNIC creation.

- `src/graph/topo.cc:1100`
  `ncclTopoMakeVnic()` calls `netInfo->makeVDevice(...)`, which reaches IB's `ncclIbMakeVDevice()`.

- `src/graph/topo.cc:1194`
  `ncclTopoAutoMerge()` groups physical NICs into vNIC requests based on `NCCL_NET_MERGE_LEVEL`.

- `src/graph/topo.cc:1482`
  `ncclTopoProcessNet()` populates physical NICs, conditionally creates vNICs, then populates virtual NICs.
  It caches the number of vNICs via `setVirtDevCount`, so candidate construction must avoid stale/global vNIC state.

- `src/graph/topo.cc:1521`
  `ncclTopoGetSystem()` imports NET plugin devices into topology under a `netMutex`.

## graph search entry

- `src/init.cc:1091`
  Topology is built through `ncclTopoGetSystem(comm, &comm->topo)`, then paths are computed and the system is trimmed.

- `src/init.cc:1138`
  Graph search starts in `initTransportsRank()`.

- `src/init.cc:1140`
  Ring graph is initialized with `pattern = NCCL_TOPO_PATTERN_RING`, `minChannels = 1`, `maxChannels = MAXCHANNELS/2`.

- `src/init.cc:1145`
  Ring graph search calls `ncclTopoCompute(comm->topo, ringGraph)`.

- `src/graph/search.cc:1040`
  `ncclTopoCompute()` configures path limits, speed arrays, search passes, and invokes recursive graph search.

- `src/graph/search.cc:1124`
  Main recursive search call: `ncclTopoSearchRec(system, &tmpGraph, graph, &time)`.

- `src/graph/search.cc:671`
  `ncclTopoSearchRecNet()` chooses start NETs and writes `graph->inter[channel*2]`.

- `src/graph/search.cc:578`
  `ncclTopoSearchRecGpu()` writes `graph->intra[channel*ngpus + step]` and, when returning to NET, writes `graph->inter[channel*2+1]`.

## channel ring printing

- `src/graph/rings.cc:29`
  `ncclBuildRings()` builds the final rank order for each channel from `ringPrev`/`ringNext`.

- `src/graph/rings.cc:48`
  Prints `Channel %02d/%02d :`.

- `src/graph/rings.cc:49`
  Only rank 0 calls `dumpLine(rings+r*nranks, nranks, prefix)`.

- `src/init.cc:1445`
  Prints per-rank local ring neighbor summary: `Ring %02d : prev -> rank -> next`.

## coll channels printing

- `src/init.cc:1590`
  Prints `%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer`.

## via NET/IB printing

- `src/transport/net.cc:298`
  `sendSetup()` resolves send-side NET device metadata.

- `src/transport/net.cc:307`
  Calls `ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank)`.

- `src/transport/net.cc:321`
  Prints `[send] via NET/%s/%d...`, yielding strings like `via NET/IB/8/GDRDMA`.

- `src/transport/net.cc:341`
  `recvSetup()` resolves recv-side NET device metadata.

- `src/transport/net.cc:351`
  Calls `ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &proxyRank)`.

- `src/transport/net.cc:368`
  Prints `[receive] via NET/%s/%d...`.

## rank-to-node mapping

- `src/include/transport.h:48`
  `ncclPeerInfo` contains `hostHash`.

- `src/init.cc:999`
  Allocates `comm->peerInfo`, fills local peer info, then all-gathers it.

- `src/init.cc:1012`
  Early approximate `nNodes` counts ranks whose `hostHash` differs from local rank. This is not the final node map.

- `src/init.cc:1245`
  Final node detection starts after AllGather3.

- `src/init.cc:1248`
  Allocates `comm->rankToNode`.

- `src/init.cc:1251`
  Groups ranks by `allGather3Data[r].topoRanks.ringRecv[0]` to assign node ids.

- `src/init.cc:1337`
  Builds `comm->nodeRanks`, `comm->rankToLocalRank`, and `localRankToRank`.

- `src/include/comm.h:580`
  `ncclComm` stores `node`, `nNodes`, `localRank`, `localRanks`.

- `src/include/comm.h:586`
  `ncclComm` stores `rankToNode`, `rankToLocalRank`, `localRankToRank`, and `nodeRanks`.

## channel ring order structure

- `src/include/graph.h:158`
  `struct ncclTopoGraph` stores search results:
  `nChannels`, `intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES]`, and `inter[MAXCHANNELS*2]`.

- `src/include/graph.h:183`
  `struct ncclTopoRanks` stores per-rank `ringRecv`, `ringSend`, `ringPrev`, and `ringNext` for all channels.

- `src/graph/connect.cc:20`
  `ncclTopoPreset()` reads `graphs[NCCL_ALGO_RING]->intra` and fills local `topoRanks`.

- `src/graph/connect.cc:95`
  `connectRings()` connects per-node `ringRecv` and `ringSend` into global `ringPrev` and `ringNext`.

- `src/graph/connect.cc:375`
  `ncclTopoPostset()` gathers all ranks' topo ranks, applies cross-NIC ring alternation, connects rings, duplicates/copies channels, and finally builds full ring orders.

- `src/graph/connect.cc:508`
  `ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext)` materializes channel ring orders in the `rings` array.

- `src/init.cc:1432`
  `rings` is allocated as `nranks*MAXCHANNELS`.

- `src/init.cc:1503`
  `setupChannel(comm, c, rank, nranks, rings+c*nranks)` stores the built ring into the communicator.

- `src/init.cc:762`
  `setupChannel()` rotates `ringRanks` to start from the local rank and writes `comm->channels[channelId].ring.userRanks`.

- `src/include/device.h:171`
  `struct ncclRing` contains `prev`, `next`, `userRanks`, `rankToIndex`, and `index`.

Important extraction note: `comm->channels[c].ring.userRanks` is rotated to local rank. For global candidate analysis, use the `rings` array from `ncclTopoPostset()` or normalize rotated `userRanks` back to a canonical start rank.

## NET device id structure

- `src/include/graph.h:175`
  `ncclTopoGraph::intra` stores rank order within each searched local ring channel.

- `src/include/graph.h:176`
  `ncclTopoGraph::inter` stores two NET topology ids per channel.

- `src/graph/search.cc:686`
  Search writes the start NET id into `graph->inter[channel*2]`.

- `src/graph/search.cc:625`
  Search writes the return NET id into `graph->inter[channel*2+1]`.

- `src/graph/search.cc:1334`
  `ncclTopoGetNetDev()` resolves a graph/channel/rank into `netId`, plugin `netDev`, and `proxyRank`.

- `src/graph/search.cc:1339`
  It maps duplicated compute channels with `channelId % graph->nChannels`.

- `src/graph/search.cc:1341`
  It chooses inter index `0` when the passed rank is the first rank in the graph channel, otherwise index `1`.

- `src/graph/search.cc:1347`
  Converts `netId` to plugin device id via `ncclTopoIdToNetDev()`.

- `src/graph/topo.h:229`
  `ncclTopoIdToNetDev()` scans `system->nodes[NET]` and returns `node.net.dev`.

## NET bandwidth source

- `src/transport/net_ib/init.cc:93`
  IB link speed tables convert active port speed/width.

- `src/transport/net_ib/init.cc:357`
  IB device discovery sets `ncclIbDevs[n].speed`.

- `src/transport/net_ib/init.cc:465`
  Physical IB `getProperties()` returns `props->speed = ibDev->speed`.

- `src/transport/net_ib/init.cc:499`
  Merged IB `getProperties()` returns `props->speed = mergedDev->speed`.

- `src/graph/topo.cc:400`
  Topology XML `speed` is read in Mbps.

- `src/graph/topo.cc:403`
  `net->net.bw = mbps / 8000.0` stores GB/s-style bandwidth for a NET node.

- `src/graph/topo.cc:423`
  NET links use `net->net.bw`.

- `src/graph/topo.cc:362`
  `ncclTopoGetMinNetBw()` returns the minimum accessible single-device network bandwidth for a rank; it does not sum multiple NICs.

For merge-auto metrics, the best real bandwidth source for a resolved `netDev` is `comm->topo->nodes[NET][netIndex].net.bw` after converting `netDev`/`netId` to the NET node. If that mapping is unavailable, unit bandwidth fallback is consistent with the first-version design.

## Candidate construction caveat

The merge mode affects topology population, not only graph search. `NCCL_IB_MERGE_NICS=0` prevents multi-device IB vNIC creation in `ncclIbMakeVDeviceInternal()`, while `ncclTopoProcessNet()` caches virtual-device counts for the plugin. Therefore, a correct two-candidate implementation should not simply mutate the environment or rerun `ncclTopoCompute()` on the same already-populated topology. It needs an explicit merge-mode context around topology/vNIC construction or a carefully scoped candidate topology builder that avoids polluting global plugin/IB merged-device state and remains safe for concurrent communicator initialization.
