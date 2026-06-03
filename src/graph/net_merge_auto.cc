/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_merge_auto.h"
#include "comm.h"
#include "debug.h"
#include "graph.h"
#include "param.h"
#include "topo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int64_t ncclParamIbMergeNics();

NCCL_PARAM(IbMergeNicsAutoThreshold, "IB_MERGE_NICS_AUTO_THRESHOLD", 110);
NCCL_PARAM(IbMergeNicsAutoDump, "IB_MERGE_NICS_AUTO_DUMP", 0);

int ncclIbMergeNicsMode() {
  return (int)ncclParamIbMergeNics();
}

bool ncclIbMergeNicsAutoEnabled() {
  return ncclIbMergeNicsMode() == NCCL_IB_MERGE_NICS_MODE_AUTO;
}

int ncclIbMergeNicsAutoThresholdPct() {
  return (int)ncclParamIbMergeNicsAutoThreshold();
}

bool ncclIbMergeNicsAutoDumpEnabled() {
  return ncclIbMergeNicsAutoEnabled() && ncclParamIbMergeNicsAutoDump() != 0;
}

ncclResult_t ncclIbMergeNicsAutoLogEnv() {
  if (ncclIbMergeNicsAutoEnabled()) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: env mode=2 threshold=%d dump=%d",
      ncclIbMergeNicsAutoThresholdPct(), ncclIbMergeNicsAutoDumpEnabled() ? 1 : 0);
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoBuildNodeMapFromComm(struct ncclComm* comm, struct ncclMergeAutoNodeMap* map) {
  if (comm == NULL || map == NULL || comm->peerInfo == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  uint64_t* rankHostHash = (uint64_t*)malloc(sizeof(*rankHostHash) * comm->nRanks);
  if (rankHostHash == NULL) return ncclSystemError;
  for (int r = 0; r < comm->nRanks; r++) rankHostHash[r] = comm->peerInfo[r].hostHash;

  ncclResult_t ret = ncclMergeAutoBuildTwoNodeMapFromHashes(comm->nRanks, rankHostHash, map);
  free(rankHostHash);
  return ret;
}

ncclResult_t ncclMergeAutoCheckTwoNode(struct ncclComm* comm, struct ncclMergeAutoNodeMap* nodeMap, int* isTwoNode) {
  if (isTwoNode == NULL) return ncclInvalidArgument;
  *isTwoNode = 0;
  if (!ncclIbMergeNicsAutoEnabled()) return ncclSuccess;
  if (comm == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclSuccess;

  struct ncclMergeAutoNodeMap localNodeMap;
  struct ncclMergeAutoNodeMap* map = nodeMap != NULL ? nodeMap : &localNodeMap;
  NCCLCHECK(ncclMergeAutoBuildNodeMapFromComm(comm, map));
  *isTwoNode = (map->valid && map->numNodes == 2) ? 1 : 0;
  return ncclSuccess;
}

static bool ncclMergeAutoIsIbNet(struct ncclComm* comm) {
  if (comm == NULL || comm->ncclNet == NULL || comm->ncclNet->name == NULL) return false;
  return strcmp(comm->ncclNet->name, "IB") == 0;
}

ncclResult_t ncclMergeAutoCheckRuntime(
    struct ncclComm* comm,
    int minNetDeviceCount,
    struct ncclMergeAutoNodeMap* nodeMap,
    int* shouldRun) {
  if (shouldRun != NULL) *shouldRun = 0;
  if (!ncclIbMergeNicsAutoEnabled()) return ncclSuccess;
  if (comm == NULL || nodeMap == NULL) return ncclInvalidArgument;

  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=nranks_unsupported nranks=%d max=%d", comm->nRanks, NCCL_MERGE_AUTO_MAX_RANKS);
    return ncclSuccess;
  }

  ncclResult_t ret = ncclMergeAutoBuildNodeMapFromComm(comm, nodeMap);
  if (ret != ncclSuccess) return ret;

  if (!nodeMap->valid || nodeMap->numNodes != 2) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=not_two_nodes nodes=%d nranks=%d", nodeMap->numNodes, comm->nRanks);
    return ncclSuccess;
  }

  if (!ncclMergeAutoIsIbNet(comm)) {
    const char* netName = (comm->ncclNet != NULL && comm->ncclNet->name != NULL) ? comm->ncclNet->name : "none";
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=non_ib_net net=%s", netName);
    return ncclSuccess;
  }

  if (minNetDeviceCount < 2) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=num_net_devs_lt_2 count=%d", minNetDeviceCount);
    return ncclSuccess;
  }

  if (shouldRun != NULL) *shouldRun = 1;
  if (comm->rank == 0) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: enabled mode=two_node nodes=2 nranks=%d minNetDevs=%d threshold=%d",
      comm->nRanks, minNetDeviceCount, ncclIbMergeNicsAutoThresholdPct());
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoExtractGraphChannelRings(
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    struct ncclMergeAutoChannelRing* rings,
    int* rankStorage,
    int maxChannels,
    int maxRanksPerChannel,
    struct ncclMergeAutoChannelSet* out) {
  if (system == NULL || graph == NULL || rings == NULL || rankStorage == NULL || out == NULL) return ncclInvalidArgument;
  if (graph->nChannels < 0 || graph->nChannels > maxChannels) return ncclInvalidArgument;
  if (graph->pattern != NCCL_TOPO_PATTERN_RING) return ncclInvalidArgument;
  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > maxRanksPerChannel || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  out->nChannels = 0;
  out->rings = rings;
  for (int c = 0; c < graph->nChannels; c++) {
    int* ranks = rankStorage + c * maxRanksPerChannel;
    for (int i = 0; i < ngpus; i++) {
      int rank = graph->intra[c * ngpus + i];
      int gpuIndex;
      ncclResult_t ret = ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/false);
      if (ret != ncclSuccess) return ret;
      ranks[i] = rank;
    }
    rings[c].channelId = c;
    rings[c].nRanks = ngpus;
    rings[c].ranks = ranks;
    out->nChannels++;
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoDumpGraphChannelRings(const char* label, struct ncclTopoSystem* system, const struct ncclTopoGraph* graph) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "graph";
  if (system == NULL || graph == NULL || graph->nChannels < 0 || graph->nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (graph->nChannels == 0) return ncclSuccess;

  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  struct ncclMergeAutoChannelRing* rings = (struct ncclMergeAutoChannelRing*)malloc(sizeof(*rings) * graph->nChannels);
  int* rankStorage = (int*)malloc(sizeof(*rankStorage) * graph->nChannels * ngpus);
  if (rings == NULL || rankStorage == NULL) {
    free(rings);
    free(rankStorage);
    return ncclSystemError;
  }
  struct ncclMergeAutoChannelSet channels;
  ncclResult_t ret = ncclMergeAutoExtractGraphChannelRings(system, graph, rings, rankStorage, graph->nChannels, ngpus, &channels);
  if (ret != ncclSuccess) {
    free(rings);
    free(rankStorage);
    return ret;
  }

  for (int c = 0; c < channels.nChannels; c++) {
    const struct ncclMergeAutoChannelRing* ring = channels.rings + c;
    int lineSize = 64 + ring->nRanks * 16;
    char* line = (char*)malloc(lineSize);
    if (line == NULL) {
      ret = ncclSystemError;
      break;
    }
    int offset = snprintf(line, lineSize, "MergeAutoDump: cand=%s ch=%02d ring=", label, ring->channelId);
    if (offset < 0) {
      free(line);
      ret = ncclSystemError;
      break;
    }
    for (int r = 0; r < ring->nRanks && offset < lineSize; r++) {
      int written = snprintf(line + offset, lineSize - offset, "%s%d", r == 0 ? "" : " ", ring->ranks[r]);
      if (written < 0) {
        free(line);
        ret = ncclSystemError;
        break;
      }
      offset += written;
    }
    if (ret == ncclSuccess) {
      if (offset >= lineSize) snprintf(line + lineSize - 4, 4, "...");
      INFO(NCCL_GRAPH|NCCL_NET, "%s", line);
    }
    free(line);
    if (ret != ncclSuccess) break;
  }
  free(rings);
  free(rankStorage);
  return ret;
}

ncclResult_t ncclMergeAutoResolveNetDevForEdge(struct ncclComm* comm, const struct ncclTopoGraph* graph, struct ncclMergeAutoCrossEdge* edge) {
  if (comm == NULL || graph == NULL || edge == NULL) return ncclInvalidArgument;
  edge->netDev = -1;
  edge->netBw = 1.0;
  edge->nPhysRails = 0;

  int proxyRank;
  int64_t netId;
  int netDev = -1;
  ncclResult_t ret = ncclTopoGetNetDev(comm, edge->srcRank, (struct ncclTopoGraph*)graph, edge->channelId, edge->dstRank, &netId, &netDev, &proxyRank);
  if (ret != ncclSuccess || netDev < 0) return ncclSuccess;
  edge->netDev = netDev;

  if (comm->ncclNet == NULL || comm->ncclNet->getProperties == NULL) return ncclSuccess;
  ncclNetProperties_t props;
  ret = comm->ncclNet->getProperties(netDev, &props);
  if (ret != ncclSuccess) return ncclSuccess;

  int nPhysRails = props.vProps.ndevs;
  if (nPhysRails <= 0) {
    edge->nPhysRails = 1;
    edge->physRails[0] = netDev;
    edge->physRailBw[0] = 1.0;
    return ncclSuccess;
  }
  if (nPhysRails > NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE) nPhysRails = NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE;
  edge->nPhysRails = nPhysRails;
  for (int r = 0; r < nPhysRails; r++) {
    edge->physRails[r] = props.vProps.devs[r];
    edge->physRailBw[r] = 1.0;
  }
  return ncclSuccess;
}

static void ncclMergeAutoFormatPhysRails(const struct ncclMergeAutoCrossEdge* edge, char* buffer, int bufferSize) {
  if (buffer == NULL || bufferSize <= 0) return;
  if (bufferSize < 4) {
    buffer[0] = '\0';
    return;
  }
  int offset = snprintf(buffer, bufferSize, "{");
  if (offset < 0) {
    buffer[0] = '\0';
    return;
  }
  for (int r = 0; r < edge->nPhysRails && offset < bufferSize; r++) {
    int written = snprintf(buffer + offset, bufferSize - offset, "%s%d", r == 0 ? "" : ",", edge->physRails[r]);
    if (written < 0) {
      buffer[0] = '\0';
      return;
    }
    offset += written;
  }
  if (offset < bufferSize) {
    snprintf(buffer + offset, bufferSize - offset, "}");
  } else {
    snprintf(buffer + bufferSize - 4, 4, "...");
  }
}

static void ncclMergeAutoDumpResolvedEdge(const char* label, struct ncclComm* comm, const struct ncclMergeAutoCrossEdge* edge) {
  char phys[256];
  ncclMergeAutoFormatPhysRails(edge, phys, sizeof(phys));
  const char* netName = (comm != NULL && comm->ncclNet != NULL && comm->ncclNet->name != NULL) ? comm->ncclNet->name : "unknown";
  if (edge->netDev >= 0) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAutoDump: cand=%s ch=%02d edge=%d->%d dir=%d->%d net=%s/%d phys=%s bw=1",
      label, edge->channelId, edge->srcRank, edge->dstRank, edge->srcNode, edge->dstNode, netName, edge->netDev, phys);
  } else {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAutoDump: cand=%s ch=%02d edge=%d->%d dir=%d->%d net=unknown phys=%s bw=1",
      label, edge->channelId, edge->srcRank, edge->dstRank, edge->srcNode, edge->dstNode, phys);
  }
}

ncclResult_t ncclMergeAutoDumpGraphCrossEdges(
    const char* label,
    struct ncclComm* comm,
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    const struct ncclMergeAutoNodeMap* nodeMap) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "graph";
  if (system == NULL || graph == NULL || nodeMap == NULL || graph->nChannels < 0 || graph->nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (!nodeMap->valid || nodeMap->numNodes != 2) return ncclSuccess;
  if (graph->nChannels == 0) return ncclSuccess;

  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  struct ncclMergeAutoChannelRing* rings = (struct ncclMergeAutoChannelRing*)malloc(sizeof(*rings) * graph->nChannels);
  int* rankStorage = (int*)malloc(sizeof(*rankStorage) * graph->nChannels * ngpus);
  struct ncclMergeAutoCrossEdge* edges = (struct ncclMergeAutoCrossEdge*)malloc(sizeof(*edges) * graph->nChannels * ngpus);
  if (rings == NULL || rankStorage == NULL || edges == NULL) {
    free(rings);
    free(rankStorage);
    free(edges);
    return ncclSystemError;
  }

  struct ncclMergeAutoChannelSet channels;
  ncclResult_t ret = ncclMergeAutoExtractGraphChannelRings(system, graph, rings, rankStorage, graph->nChannels, ngpus, &channels);
  int nEdges = 0;
  if (ret == ncclSuccess) {
    ret = ncclMergeAutoExtractCrossEdges(&channels, nodeMap, edges, graph->nChannels * ngpus, &nEdges);
  }
  if (ret == ncclSuccess) {
    for (int e = 0; e < nEdges; e++) {
      struct ncclMergeAutoCrossEdge* edge = edges + e;
      if (comm != NULL) {
        ncclResult_t resolveRet = ncclMergeAutoResolveNetDevForEdge(comm, graph, edge);
        if (resolveRet != ncclSuccess) ret = resolveRet;
      }
      if (ret != ncclSuccess) break;
      ncclMergeAutoDumpResolvedEdge(label, comm, edge);
    }
  }

  free(rings);
  free(rankStorage);
  free(edges);
  return ret;
}

ncclResult_t ncclMergeAutoDumpGraphCrossEdgesFromComm(const char* label, struct ncclComm* comm, const struct ncclTopoGraph* graph) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (comm == NULL || graph == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclSuccess;

  struct ncclMergeAutoNodeMap nodeMap;
  ncclResult_t ret = ncclMergeAutoBuildNodeMapFromComm(comm, &nodeMap);
  if (ret != ncclSuccess) return ret;
  return ncclMergeAutoDumpGraphCrossEdges(label, comm, comm->topo, graph, &nodeMap);
}

ncclResult_t ncclMergeAutoDumpPostsetRingEdges(
    const char* label,
    struct ncclComm* comm,
    const struct ncclTopoGraph* graph,
    struct ncclTopoRanks** allTopoRanks,
    const int* firstRanks,
    int nChannels) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "postset";
  if (comm == NULL || graph == NULL || allTopoRanks == NULL || firstRanks == NULL) return ncclInvalidArgument;
  if (nChannels < 0 || nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (comm->nNodes != 2 || comm->rankToNode == NULL || comm->node < 0 || comm->node >= comm->nNodes) return ncclSuccess;

  int node = comm->node;
  if (firstRanks[node] != comm->rank) return ncclSuccess;
  int nextNode = (node + 1) % comm->nNodes;
  int srcFirstRank = firstRanks[node];
  int dstFirstRank = firstRanks[nextNode];
  if (srcFirstRank < 0 || srcFirstRank >= comm->nRanks || dstFirstRank < 0 || dstFirstRank >= comm->nRanks) return ncclInvalidArgument;
  if (allTopoRanks[srcFirstRank] == NULL || allTopoRanks[dstFirstRank] == NULL) return ncclInvalidArgument;

  for (int c = 0; c < nChannels; c++) {
    int src = allTopoRanks[srcFirstRank]->ringSend[c];
    int dst = allTopoRanks[dstFirstRank]->ringRecv[c];
    if (src < 0 || src >= comm->nRanks || dst < 0 || dst >= comm->nRanks) return ncclInvalidArgument;
    int srcNode = comm->rankToNode[src];
    int dstNode = comm->rankToNode[dst];
    if (srcNode < 0 || srcNode >= comm->nNodes || dstNode < 0 || dstNode >= comm->nNodes) return ncclInvalidArgument;
    if (srcNode == dstNode) continue;

    struct ncclMergeAutoCrossEdge edge;
    memset(&edge, 0, sizeof(edge));
    edge.channelId = c;
    edge.srcRank = src;
    edge.dstRank = dst;
    edge.srcNode = srcNode;
    edge.dstNode = dstNode;
    edge.direction = (srcNode == 0 && dstNode == 1) ? 0 : 1;
    edge.netDev = -1;
    edge.netBw = 1.0;
    NCCLCHECK(ncclMergeAutoResolveNetDevForEdge(comm, graph, &edge));
    ncclMergeAutoDumpResolvedEdge(label, comm, &edge);
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoBuildTwoNodeMapFromHashes(int nranks, const uint64_t* rankHostHash, struct ncclMergeAutoNodeMap* map) {
  if (rankHostHash == NULL || map == NULL || nranks <= 0 || nranks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  memset(map, 0, sizeof(*map));
  map->nranks = nranks;
  map->valid = 0;
  for (int r = 0; r < nranks; r++) map->rankToNode[r] = -1;

  for (int r = 0; r < nranks; r++) {
    int node = -1;
    for (int n = 0; n < map->numNodes; n++) {
      if (map->nodeHash[n] == rankHostHash[r]) {
        node = n;
        break;
      }
    }
    if (node == -1) {
      if (map->numNodes == 2) {
        map->numNodes = 3;
        return ncclSuccess;
      }
      node = map->numNodes++;
      map->nodeHash[node] = rankHostHash[r];
    }
    map->rankToNode[r] = node;
    map->nodeRankCount[node]++;
  }

  map->valid = (map->numNodes == 2 && map->nodeRankCount[0] > 0 && map->nodeRankCount[1] > 0);
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoExtractCrossEdges(
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoNodeMap* nodeMap,
    struct ncclMergeAutoCrossEdge* edges,
    int maxEdges,
    int* nEdges) {
  if (channels == NULL || nodeMap == NULL || edges == NULL || nEdges == NULL || maxEdges < 0) return ncclInvalidArgument;
  if (channels->nChannels < 0 || (channels->nChannels > 0 && channels->rings == NULL)) return ncclInvalidArgument;
  if (!nodeMap->valid || nodeMap->numNodes != 2) return ncclInvalidArgument;

  *nEdges = 0;
  for (int c = 0; c < channels->nChannels; c++) {
    const struct ncclMergeAutoChannelRing* ring = channels->rings + c;
    if (ring->nRanks < 2 || ring->ranks == NULL) return ncclInvalidArgument;

    for (int i = 0; i < ring->nRanks; i++) {
      int src = ring->ranks[i];
      int dst = ring->ranks[(i + 1) % ring->nRanks];
      if (src < 0 || src >= nodeMap->nranks || dst < 0 || dst >= nodeMap->nranks) return ncclInvalidArgument;
      int srcNode = nodeMap->rankToNode[src];
      int dstNode = nodeMap->rankToNode[dst];
      if (srcNode < 0 || dstNode < 0) return ncclInvalidArgument;
      if (srcNode == dstNode) continue;
      if (srcNode > 1 || dstNode > 1) return ncclInvalidArgument;
      if (*nEdges == maxEdges) return ncclInvalidArgument;

      struct ncclMergeAutoCrossEdge* edge = edges + (*nEdges)++;
      memset(edge, 0, sizeof(*edge));
      edge->channelId = ring->channelId;
      edge->srcRank = src;
      edge->dstRank = dst;
      edge->srcNode = srcNode;
      edge->dstNode = dstNode;
      edge->direction = (srcNode == 0 && dstNode == 1) ? 0 : 1;
      edge->netDev = -1;
      edge->netBw = 1.0;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclMergeAutoAddUnique(int id, double bw, int* ids, double* bws, int* count) {
  if (id < 0) return ncclSuccess;
  if (bw <= 0.0) bw = 1.0;
  for (int i = 0; i < *count; i++) {
    if (ids[i] == id) {
      if (bws[i] < bw) bws[i] = bw;
      return ncclSuccess;
    }
  }
  if (*count == NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS) return ncclInvalidArgument;
  ids[*count] = id;
  bws[*count] = bw;
  (*count)++;
  return ncclSuccess;
}

static double ncclMergeAutoSumBw(const double* bws, int count) {
  double sum = 0.0;
  for (int i = 0; i < count; i++) sum += bws[i];
  return sum;
}

static ncclResult_t ncclMergeAutoMergeUniqueSets(const int* ids, const double* bws, int count, int* totalIds, double* totalBws, int* totalCount) {
  for (int i = 0; i < count; i++) {
    ncclResult_t ret = ncclMergeAutoAddUnique(ids[i], bws[i], totalIds, totalBws, totalCount);
    if (ret != ncclSuccess) return ret;
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoAggregateMetrics(
    int merge,
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoCrossEdge* edges,
    int nEdges,
    struct ncclMergeAutoMetrics* metrics) {
  if (channels == NULL || edges == NULL || metrics == NULL || nEdges < 0) return ncclInvalidArgument;
  if (channels->nChannels < 0 || (channels->nChannels > 0 && channels->rings == NULL)) return ncclInvalidArgument;
  memset(metrics, 0, sizeof(*metrics));
  metrics->merge = merge;
  metrics->valid = 1;
  metrics->nChannels = channels->nChannels;
  metrics->nCrossEdges = nEdges;

  int net01[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int net10[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int netTotal[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double netBw01[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double netBw10[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double netBwTotal[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int rail01[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int rail10[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int railTotal[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double railBw01[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double railBw10[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double railBwTotal[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int nNet01 = 0, nNet10 = 0, nNetTotal = 0;
  int nRail01 = 0, nRail10 = 0, nRailTotal = 0;

  for (int e = 0; e < nEdges; e++) {
    const struct ncclMergeAutoCrossEdge* edge = edges + e;
    if (edge->direction != 0 && edge->direction != 1) return ncclInvalidArgument;
    int syntheticId = edge->channelId * 2 + edge->direction;
    int netDev = edge->netDev >= 0 ? edge->netDev : syntheticId;
    if (edge->direction == 0) {
      ncclResult_t ret = ncclMergeAutoAddUnique(netDev, edge->netBw, net01, netBw01, &nNet01);
      if (ret != ncclSuccess) return ret;
    } else {
      ncclResult_t ret = ncclMergeAutoAddUnique(netDev, edge->netBw, net10, netBw10, &nNet10);
      if (ret != ncclSuccess) return ret;
    }

    if (edge->nPhysRails < 0 || edge->nPhysRails > NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE) return ncclInvalidArgument;
    if (edge->nPhysRails == 0) {
      ncclResult_t ret = edge->direction == 0 ?
        ncclMergeAutoAddUnique(netDev, edge->netBw, rail01, railBw01, &nRail01) :
        ncclMergeAutoAddUnique(netDev, edge->netBw, rail10, railBw10, &nRail10);
      if (ret != ncclSuccess) return ret;
    } else {
      for (int r = 0; r < edge->nPhysRails; r++) {
        ncclResult_t ret = edge->direction == 0 ?
          ncclMergeAutoAddUnique(edge->physRails[r], edge->physRailBw[r], rail01, railBw01, &nRail01) :
          ncclMergeAutoAddUnique(edge->physRails[r], edge->physRailBw[r], rail10, railBw10, &nRail10);
        if (ret != ncclSuccess) return ret;
      }
    }
  }

  ncclResult_t ret = ncclMergeAutoMergeUniqueSets(net01, netBw01, nNet01, netTotal, netBwTotal, &nNetTotal);
  if (ret != ncclSuccess) return ret;
  ret = ncclMergeAutoMergeUniqueSets(net10, netBw10, nNet10, netTotal, netBwTotal, &nNetTotal);
  if (ret != ncclSuccess) return ret;
  ret = ncclMergeAutoMergeUniqueSets(rail01, railBw01, nRail01, railTotal, railBwTotal, &nRailTotal);
  if (ret != ncclSuccess) return ret;
  ret = ncclMergeAutoMergeUniqueSets(rail10, railBw10, nRail10, railTotal, railBwTotal, &nRailTotal);
  if (ret != ncclSuccess) return ret;

  metrics->uniqueNetDevs01 = nNet01;
  metrics->uniqueNetDevs10 = nNet10;
  metrics->uniqueNetDevsTotal = nNetTotal;
  metrics->uniqueRails01 = nRail01;
  metrics->uniqueRails10 = nRail10;
  metrics->uniqueRailsTotal = nRailTotal;
  metrics->dirBw01 = ncclMergeAutoSumBw(railBw01, nRail01);
  metrics->dirBw10 = ncclMergeAutoSumBw(railBw10, nRail10);
  metrics->bidirBw = 2.0 * (metrics->dirBw01 < metrics->dirBw10 ? metrics->dirBw01 : metrics->dirBw10);
  double maxBw = metrics->dirBw01 > metrics->dirBw10 ? metrics->dirBw01 : metrics->dirBw10;
  metrics->balance = maxBw > 0.0 ? (metrics->bidirBw / 2.0) / maxBw : 0.0;
  metrics->score = metrics->bidirBw;
  return ncclSuccess;
}

int ncclMergeAutoSelect(const struct ncclMergeAutoMetrics* merge0, const struct ncclMergeAutoMetrics* merge1, int thresholdPct) {
  if (thresholdPct <= 0) thresholdPct = 110;
  if ((merge0 == NULL || !merge0->valid) && (merge1 == NULL || !merge1->valid)) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  if (merge0 == NULL || !merge0->valid) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  if (merge1 == NULL || !merge1->valid) return NCCL_IB_MERGE_NICS_MODE_UNMERGED;

  if (merge0->score * 100.0 > merge1->score * (double)thresholdPct) return NCCL_IB_MERGE_NICS_MODE_UNMERGED;
  if (merge1->score * 100.0 > merge0->score * (double)thresholdPct) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  return NCCL_IB_MERGE_NICS_MODE_MERGED;
}
