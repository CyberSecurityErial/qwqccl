/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_MERGE_AUTO_H_
#define NCCL_NET_MERGE_AUTO_H_

#include "nccl.h"

enum ncclIbMergeNicsMode {
  NCCL_IB_MERGE_NICS_MODE_UNMERGED = 0,
  NCCL_IB_MERGE_NICS_MODE_MERGED = 1,
  NCCL_IB_MERGE_NICS_MODE_AUTO = 2
};

int ncclIbMergeNicsMode();
bool ncclIbMergeNicsAutoEnabled();
int ncclIbMergeNicsAutoThresholdPct();
bool ncclIbMergeNicsAutoDumpEnabled();
ncclResult_t ncclIbMergeNicsAutoLogEnv();

#endif
