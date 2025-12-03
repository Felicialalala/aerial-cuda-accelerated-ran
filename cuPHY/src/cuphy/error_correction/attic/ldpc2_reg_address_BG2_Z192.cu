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

#include "ldpc2_reg.cuh"
#include "ldpc2_sign.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_float_BG2_Z192()
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z192(ldpc::decoder&            dec,
                                                      const LDPC_config&        cfg,
                                                      const LDPC_kernel_params& params,
                                                      const dim3&               grdDim,
                                                      const dim3&               blkDim,
                                                      cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 2
    constexpr int  BG = 2;
    constexpr int  Z  = 192;
    constexpr int  Kb = 10;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef float                                                     T;
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Sign manager type
    typedef sign_store_policy_dst<float, sign_order_be<float>> sign_mgr_dst_be_t;
    typedef sign_store_policy_dst<float, sign_order_le<float>> sign_mgr_dst_le_t;
    typedef sign_store_policy_src<float, sign_order_be<float>> sign_mgr_src_be_t;
    typedef sign_store_policy_src<float, sign_order_le<float>> sign_mgr_src_le_t;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Check to Variable (C2V) message type
    // USE THIS for min1 delta
    typedef cC2V_address<T, BG, min1_policy_delta, sign_mgr_dst_be_t> cC2V_t;

    // USE THIS for min1 default
    //typedef cC2V_address<T, BG, min1_policy_default, sign_mgr_dst_be_t> cC2V_t;

    switch(cfg.mb)
    {
    case  4:  s = launch_register_kernel<T, BG, Kb, Z,  4, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  5:  s = launch_register_kernel<T, BG, Kb, Z,  5, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  6:  s = launch_register_kernel<T, BG, Kb, Z,  6, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  7:  s = launch_register_kernel<T, BG, Kb, Z,  7, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  8:  s = launch_register_kernel<T, BG, Kb, Z,  8, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  9:  s = launch_register_kernel<T, BG, Kb, Z,  9, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 10:  s = launch_register_kernel<T, BG, Kb, Z, 10, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 11:  s = launch_register_kernel<T, BG, Kb, Z, 11, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 12:  s = launch_register_kernel<T, BG, Kb, Z, 12, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    default:                                                                                                          break;
    }
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 2
    return s;
}

} // namespace ldpc2
