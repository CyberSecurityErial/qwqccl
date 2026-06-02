/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_merge_auto.h"
#include "debug.h"
#include "param.h"

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
