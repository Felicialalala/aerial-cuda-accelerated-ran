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

#include "ldpc2_split.hpp"
#include "cuphy_internal.h"

using namespace ldpc2;

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index()
cuphyStatus_t decode_ldpc2_split_index(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm)
{
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    if(llrType == CUPHY_R_16F)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Convert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the maximum amount of shared memory that the
        // current device can support
        int32_t device_shmem_max = dec.max_shmem_per_block_optin();
        if(device_shmem_max <= 0)
        {
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        switch(device_shmem_max)
        {
        case (96*1024): s = decode_ldpc2_split_half_96KB(config, params, grdDim, blkDim, strm); break; // Volta:  96 KiB max (opt-in)
        case (64*1024):                                                                         break; // Turing: 64 KiB max (opt-in)
        default:                                                                                break;
        }
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ldpc2_split_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_index_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg)
{
    // For now, cC2V class in ldp2c.cuh loads and stores from global memory
    // using an offset that assumes ALL cC2Vs are in global memory, even
    // though some are in shared memory.
    // TODO: Modify the global load/store and reduce the amount of
    // allocation here.
    if(CUPHY_R_32F == cfg.type)
    {
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
    }
    else if(CUPHY_R_16F == cfg.type)
    {
        // Assumes all of workspace is used for cC2V messages (i.e. no APP values)
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int2));
    }
    else
    {
        return std::pair<bool, size_t>(false, 0);
    }

    return std::pair<bool, size_t>(true, 0);
}

} // namespace ldpc
