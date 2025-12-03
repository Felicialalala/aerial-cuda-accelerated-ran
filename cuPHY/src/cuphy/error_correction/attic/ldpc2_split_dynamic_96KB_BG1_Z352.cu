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

#include "ldpc2_split_dynamic.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_half_96KB_BG1_Z352()
cuphyStatus_t decode_ldpc2_split_dynamic_half_96KB_BG1_Z352(const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    constexpr int  BG        = 1;
    constexpr int  Z         = 352;
    constexpr int  Kb        = 22;
    
    typedef __half                                                                    T;
    typedef cC2V_index<__half, BG, sign_store_policy_src, sign_store_policy_split_src> cC2V_t;
    //typedef cC2V_index<__half, BG, sign_store_policy_dst, sign_store_policy_split_src> cC2V_t;
   
    switch(cfg.mb)
    {
    case 24:
    case 25:
        s = launch_split_dynamic<T, BG, Kb, Z, 23, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 26:
    case 27:
    case 28:
    case 29:
        s = launch_split_dynamic<T, BG, Kb, Z, 22, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 30:
    case 31:
    case 32:
    case 33:
        s = launch_split_dynamic<T, BG, Kb, Z, 21, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 34:
    case 35:
    case 36:
    case 37:
        s = launch_split_dynamic<T, BG, Kb, Z, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 38:
    case 39:
    case 40:
    case 41:
        s = launch_split_dynamic<T, BG, Kb, Z, 19, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 42:
    case 43:
    case 44:
    case 45:
        s = launch_split_dynamic<T, BG, Kb, Z, 18, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 46:
        s = launch_split_dynamic<T, BG, Kb, Z, 17, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    default:
        break;
    }
#endif // if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    return s;
}

} // namespace ldpc2