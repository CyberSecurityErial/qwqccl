/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_merge_auto.h"
#include "debug.h"
#include "param.h"
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
  return ncclParamIbMergeNicsAutoDump() != 0;
}

ncclResult_t ncclIbMergeNicsAutoLogEnv() {
  if (ncclIbMergeNicsAutoEnabled()) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: env mode=2 threshold=%d dump=%d",
      ncclIbMergeNicsAutoThresholdPct(), ncclIbMergeNicsAutoDumpEnabled() ? 1 : 0);
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
