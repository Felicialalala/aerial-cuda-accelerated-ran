/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//#define CUPHY_DEBUG 1

#include "ldpc2_split.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_half_96KB_BG2_Z288()
cuphyStatus_t decode_ldpc2_split_half_96KB_BG2_Z288(const LDPC_config&        cfg,
                                                    const LDPC_kernel_params& params,
                                                    const dim3&               grdDim,
                                                    const dim3&               blkDim,
                                                    cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    switch(cfg.mb)
    {
    default: break;
    }
    return s;
}

} // namespace ldpc2
