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

#include "cumac.h"
#include "cumac_muUeGrp_api.h"
#include "l2_muUeGrp_test.h"
#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "nv_lockfree.hpp"
#include <yaml-cpp/yaml.h>

// #define muUeGrpKernel_v1
#define muUeGrpKernel_v2

constexpr const char* YAML_CUMAC_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/cumac_nvipc.yaml";

#define CHECK_CUDA_ERR(stmt)                                                                                                                                     \
    do                                                                                                                                                           \
    {                                                                                                                                                            \
        cudaError_t result1 = (stmt);                                                                                                                            \
        if (cudaSuccess != result1)                                                                                                                              \
        {                                                                                                                                                        \
            NVLOGW(TAG, "[%s:%d] cuda failed with result1 %s", __FILE__, __LINE__, cudaGetErrorString(result1));                                            \
            cudaError_t result2 = cudaGetLastError();                                                                                                            \
            if (cudaSuccess != result2)                                                                                                                          \
            {                                                                                                                                                    \
                NVLOGW(TAG, "[%s:%d] cuda failed with result2 %s result1 %s", __FILE__, __LINE__, cudaGetErrorString(result2), cudaGetErrorString(result1)); \
                cudaError_t result3 = cudaGetLastError(); /*check for stickiness*/                                                                               \
                if (cudaSuccess != result3)                                                                                                                      \
                {                                                                                                                                                \
                    NVLOGE(TAG, AERIAL_CUDA_API_EVENT, "[%s:%d] cuda failed with result3 %s result2 %s result1 %s",                                          \
                               __FILE__,                                                                                                                         \
                               __LINE__,                                                                                                                         \
                               cudaGetErrorString(result3),                                                                                                      \
                               cudaGetErrorString(result2),                                                                                                      \
                               cudaGetErrorString(result1));                                                                                                     \
                }                                                                                                                                                \
            }                                                                                                                                                    \
        }                                                                                                                                                        \
    } while (0)

struct test_task_t {
    uint32_t        sfn;
    uint32_t        slot;
    uint8_t*        gpu_buf; // storing data for all cells in contiguous GPU memory
    uint32_t        gpu_buf_len_per_cell; // length of the allocated memory for each cell
    uint16_t        num_cell; // number of cells for this task
    nv_ipc_msg_t    recv_msg[MAX_NUM_CELL];
    cudaStream_t    strm;
};

struct cumac_CUDA_kernel_param_t {
    cumac_CUDA_kernel_param_t(const sys_param_t& sys_param) {
        num_subband = sys_param.num_subband;
        num_bs_ant_port = sys_param.num_bs_ant_port;
        num_prg_samp_per_subband = sys_param.num_prg_samp_per_subband;

        k1_num_blocks_per_row_chanOrtMat = sys_param.num_blocks_per_row_chanOrtMat;
#ifdef muUeGrpKernel_v1
        k1_num_threads_per_block = 1024;
        k1_num_blocks_per_cell = sys_param.num_srs_ue_per_slot*MAX_NUM_UE_ANT_PORT*k1_num_blocks_per_row_chanOrtMat;
#endif
#ifdef muUeGrpKernel_v2
        k1_num_threads_per_block = sys_param.num_srs_ue_per_cell*MAX_NUM_UE_ANT_PORT/k1_num_blocks_per_row_chanOrtMat;
        k1_num_threads_per_block = k1_num_threads_per_block > 1024 ? 1024 : k1_num_threads_per_block;
        k1_num_blocks_per_prg = sys_param.num_srs_ue_per_slot*MAX_NUM_UE_ANT_PORT*k1_num_blocks_per_row_chanOrtMat;
        k1_num_blocks_per_cell = sys_param.num_subband*sys_param.num_prg_samp_per_subband*k1_num_blocks_per_prg;
#endif
        k1_num_blocks_grid = k1_num_blocks_per_cell*sys_param.num_cell;
        
        k2_num_blocks_grid = sys_param.num_cell;
        k2_num_threads_per_block = 1024;
    }

    int num_subband;
    int num_prg_samp_per_subband;
    int num_bs_ant_port;

    // parameters for kernel 1 (channel orthogonality computation)
    int k1_num_blocks_per_row_chanOrtMat;
    int k1_num_blocks_per_prg;
    int k1_num_blocks_per_cell;
    int k1_num_blocks_grid;
    int k1_num_threads_per_block;

    // parameters for kernel 2 (UE grouping)
    int k2_num_blocks_grid;
    int k2_num_threads_per_block;
};

void alloc_mem_test_task(test_task_t *task, int num_cell)
{
    uint32_t allocMemLength = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_buf, allocMemLength*num_cell));
    task->gpu_buf_len_per_cell = allocMemLength;
}