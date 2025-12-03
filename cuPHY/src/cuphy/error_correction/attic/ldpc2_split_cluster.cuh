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

#if !defined(LDPC2_SPLIT_CLUSTER_CUH_INCLUDED_)
#define LDPC2_SPLIT_CLUSTER_CUH_INCLUDED_

#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_schedule_cluster.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"
#include "ldpc2_sign_split.cuh"
#include <float.h>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// launch_split_cluster()
// Launch the LDPC kernel that uses a "split" C2V message cache (part
// global and part shared memory), with ALL APP values stored in shared
// memory, and a cluster row schedule.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_C2V_SMEM,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_split_cluster(ldpc::decoder&            dec,
                                   const LDPC_kernel_params& params,
                                   const dim3&               grdDim,
                                   const dim3&               blkDim,
                                   cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (shared memory here)
    typedef c2v_cache_split<BG, Z, NUM_C2V_SMEM, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (variable number of check nodes)
    typedef ldpc_schedule_cluster<BG,          // base graph
                                  app_loc_t,   // APP location/address calc
                                  c2v_cache_t, // C2V cache
                                  NUM_PARITY> sched_t;
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    //typedef llr_loader_variable<T, Z, ldpc2::max_variable_nodes<BG>::value> llr_loader_t;
    typedef llr_loader_fixed<T, Z, Kb + NUM_PARITY> llr_loader_t;
    
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    uint32_t shmem_size = 0;
    //uint32_t       c2v_size = ldpc2::get_c2v_shared_mem_size(NUM_C2V_SMEM, Z, sizeof(T));
    uint32_t       c2v_size = c2v_cache_t::get_c2v_size_bytes(NUM_PARITY);
    const uint32_t app_size = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                    Z,                    // lifting size
                                                    sizeof(T));           // element size
    shmem_size = c2v_size + app_size;
    int32_t device_shmem_max = dec.max_shmem_per_block_optin();
    if(device_shmem_max <= 0)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if(shmem_size > device_shmem_max)
    {
        printf("c2v_size = %u, app_size = %u, shmem_size = %u, device_shmem_max = %u\n", c2v_size, app_size, shmem_size, device_shmem_max);
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //printf("c2v_size = %u, app_size = %u, shmem_size = %u\n", c2v_size, app_size, shmem_size);
    cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_size);
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, shmem_size);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 sched_t,          // schedule type
                 llr_loader_t,     // LLR loader type
                 BLOCKS_PER_SM>    // launch bounds
                 <<<grdDim, blkDim, shmem_size, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2

#endif // !defined(LDPC2_SPLIT_CLUSTER_CUH_INCLUDED_)
