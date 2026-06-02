/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_MERGE_AUTO_H_
#define NCCL_NET_MERGE_AUTO_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_MERGE_AUTO_MAX_RANKS 4096
#define NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE 16
#define NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS 256

enum ncclIbMergeNicsMode {
  NCCL_IB_MERGE_NICS_MODE_UNMERGED = 0,
  NCCL_IB_MERGE_NICS_MODE_MERGED = 1,
  NCCL_IB_MERGE_NICS_MODE_AUTO = 2
};

struct ncclMergeAutoNodeMap {
  int nranks;
  int numNodes;
  int rankToNode[NCCL_MERGE_AUTO_MAX_RANKS];
  int nodeRankCount[2];
  uint64_t nodeHash[2];
  int valid;
};

struct ncclMergeAutoChannelRing {
  int channelId;
  int nRanks;
  const int* ranks;
};

struct ncclMergeAutoChannelSet {
  int nChannels;
  const struct ncclMergeAutoChannelRing* rings;
};

struct ncclMergeAutoCrossEdge {
  int channelId;
  int srcRank;
  int dstRank;
  int srcNode;
  int dstNode;
  int direction; // 0: node0->node1, 1: node1->node0
  int netDev;
  double netBw;
  int nPhysRails;
  int physRails[NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE];
  double physRailBw[NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE];
};

struct ncclMergeAutoMetrics {
  int merge;
  int valid;
  int nChannels;
  int nCrossEdges;
  int uniqueNetDevsTotal;
  int uniqueNetDevs01;
  int uniqueNetDevs10;
  int uniqueRailsTotal;
  int uniqueRails01;
  int uniqueRails10;
  double dirBw01;
  double dirBw10;
  double bidirBw;
  double balance;
  double score;
};

int ncclIbMergeNicsMode();
bool ncclIbMergeNicsAutoEnabled();
int ncclIbMergeNicsAutoThresholdPct();
bool ncclIbMergeNicsAutoDumpEnabled();
ncclResult_t ncclIbMergeNicsAutoLogEnv();
ncclResult_t ncclMergeAutoBuildTwoNodeMapFromHashes(int nranks, const uint64_t* rankHostHash, struct ncclMergeAutoNodeMap* map);
ncclResult_t ncclMergeAutoExtractCrossEdges(
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoNodeMap* nodeMap,
    struct ncclMergeAutoCrossEdge* edges,
    int maxEdges,
    int* nEdges);
ncclResult_t ncclMergeAutoAggregateMetrics(
    int merge,
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoCrossEdge* edges,
    int nEdges,
    struct ncclMergeAutoMetrics* metrics);
int ncclMergeAutoSelect(const struct ncclMergeAutoMetrics* merge0, const struct ncclMergeAutoMetrics* merge1, int thresholdPct);

#endif
