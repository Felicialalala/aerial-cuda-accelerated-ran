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

#include "cumac_muUeGrp_test.h"

// #define LATENCY_TEST
constexpr int NUM_ITR_KERNEL_RUN = 100;

int cuMAC_WORKER_THREAD_CORE;
int cuMAC_RECV_THREAD_CORE;
int NUM_TIME_SLOTS;
int NUM_CELL;

constexpr int LEN_LOCKFREE_RING_POOL = 10;

// global variables (accessible to all threads)
nv_ipc_t* ipc = NULL;
sem_t task_sem;
nv::lock_free_ring_pool<test_task_t>* test_task_ring;

// CUDA stream for cuMAC
cudaStream_t cuStrmCumac;

void* cumac_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumac_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(cuMAC_RECV_THREAD_CORE);

    nv_ipc_msg_t recv_msg;

    int num_slot = 0;
    
    while(num_slot < NUM_TIME_SLOTS) {
        NVLOGI(TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for notification of incoming cuMAC scheduling message
        ipc->rx_tti_sem_wait(ipc);
        num_slot++;

        struct timespec msg_recv_start, msg_recv_end;
        clock_gettime(CLOCK_REALTIME, &msg_recv_start);

        test_task_t* task;

        if ((task = test_task_ring->alloc()) == nullptr) {
            NVLOGW(TAG, "RECV: task process can't catch up with enqueue, drop slot");
            continue;
        }
        task->num_cell = 0;
        task->strm = cuStrmCumac;

        // enqueue the incoming NVIPC message
        while(ipc->rx_recv_msg(ipc, &recv_msg) >= 0) {
            cumac_muUeGrp_req_msg_t* req = (cumac_muUeGrp_req_msg_t*)recv_msg.msg_buf;

            // NVLOGC(TAG, "cuMAC RECV: SFN = %u.%u, cell ID = %d, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                    //req->sfn, req->slot, recv_msg.cell_id, recv_msg.msg_id, get_cumac_msg_name(recv_msg.msg_id), recv_msg.msg_len, recv_msg.data_len);
            
            task->sfn = req->sfn;
            task->slot = req->slot;
            task->num_cell++;
            task->recv_msg[recv_msg.cell_id] = recv_msg;
        }

        test_task_ring->enqueue(task);
        sem_post(&task_sem);

        clock_gettime(CLOCK_REALTIME, &msg_recv_end);
        int64_t msg_recv_duration = nvlog_timespec_interval(&msg_recv_start, &msg_recv_end);

        NVLOGC(TAG, "cuMAC RECV: NVIPC message receive duration: %f microseconds", msg_recv_duration/1000.0);

        printf("cuMAC RECV: time slot %d, received messages of %d cells\n", num_slot-1, task->num_cell);
    }

    printf("cuMAC receiver process: test completed successfully\n");
    return NULL;
}

#define dir 0 // controls direction of comparator sorts

template<typename T1, typename T2>
inline __device__ void bitonicSort(T1* valueArr, T2* uidArr, uint16_t n)
{
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              T1 t;
              T2 t_uid;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == d) {
                  t = valueArr[pos];
                  valueArr[pos] = valueArr[pos + stride];
                  valueArr[pos + stride] = t;
                  t_uid = uidArr[pos];
                  uidArr[pos] = uidArr[pos + stride];
                  uidArr[pos + stride] = t_uid;
              }
           }
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           T1 t;
           T2 t_uid;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == dir) {
               t = valueArr[pos];
               valueArr[pos] = valueArr[pos + stride];
               valueArr[pos + stride] = t;
             
               t_uid = uidArr[pos];
               uidArr[pos] = uidArr[pos + stride];
               uidArr[pos + stride] = t_uid;
           }
        }
    }

    __syncthreads(); 
}

__global__ void ueGrpKernel_64T64R_chanCorr_v2(uint8_t*    main_in_buf, 
                                               uint8_t*    task_in_buf, 
                                               float*      gpu_srsChanOrt,
                                               uint32_t    task_in_buf_len_per_cell,
                                               int         num_cell,
                                               int         num_bs_ant_port,
                                               int         num_subband,
                                               int         num_prg_samp_per_subband,
                                               int         num_blocks_per_prg,
                                               int         num_blocks_per_cell,
                                               int         num_blocks_per_row_chanOrtMat)
{
    uint16_t cellId = blockIdx.x / num_blocks_per_cell;
    uint16_t blockInd_in_cell = blockIdx.x - cellId*num_blocks_per_cell;
    uint16_t prgIdx = blockInd_in_cell/num_blocks_per_prg;
    uint16_t blockIdx_in_prg = blockInd_in_cell - prgIdx*num_blocks_per_prg;
    uint16_t row_idx_chanOrtMat = blockIdx_in_prg / num_blocks_per_row_chanOrtMat;
    uint16_t blockIdx_in_row_chanOrtMat = blockIdx_in_prg - row_idx_chanOrtMat*num_blocks_per_row_chanOrtMat;
    uint16_t srs_info_idx = row_idx_chanOrtMat / MAX_NUM_UE_ANT_PORT;
    uint16_t srs_info_ant_port = row_idx_chanOrtMat - srs_info_idx*MAX_NUM_UE_ANT_PORT;

    cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (task_in_buf + cellId*task_in_buf_len_per_cell);

    cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (req_info_ptr->srsInfo + req_info_ptr->numSrsInfo);

    if (srs_info_idx >= req_info_ptr->numSrsInfo) {
        return;
    }

    uint16_t num_ue_ant_port = req_info_ptr->srsInfo[srs_info_idx].nUeAnt;

    if (srs_info_ant_port >= num_ue_ant_port) {
        return;
    }

    uint16_t ue_info_per_block = req_info_ptr->numUeInfo/num_blocks_per_row_chanOrtMat;
    uint16_t first_ue_info_idx_in_block = ue_info_per_block*blockIdx_in_row_chanOrtMat;
    uint16_t last_ue_info_idx_in_block;
    if (blockIdx_in_row_chanOrtMat == num_blocks_per_row_chanOrtMat - 1) { // last block
        last_ue_info_idx_in_block = req_info_ptr->numUeInfo - 1;
    } else {
        last_ue_info_idx_in_block = first_ue_info_idx_in_block + ue_info_per_block - 1;
    }

    uint16_t portIdx = threadIdx.x;
    uint16_t tx_port_idx = portIdx%MAX_NUM_UE_ANT_PORT;
    uint16_t num_rnd = ceil(static_cast<float>((last_ue_info_idx_in_block - first_ue_info_idx_in_block + 1)*MAX_NUM_UE_ANT_PORT)/blockDim.x);
    
    __shared__ cuComplex    srs_info_chanEst[MAX_NUM_BS_ANT_PORT];

    uint16_t row_idx = req_info_ptr->srsInfo[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;

    // update SRS SNRs
    if (srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
        float* srsSnr_main_buf_ptr = (float*) (main_in_buf + sizeof(cuComplex)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*num_cell + sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*cellId);

        srsSnr_main_buf_ptr[req_info_ptr->srsInfo[srs_info_idx].id] = req_info_ptr->srsInfo[srs_info_idx].srsWbSnr;
    }

    cuComplex* chanEst_main_buf_ptr = (cuComplex*) (main_in_buf + sizeof(cuComplex)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

    float* gpu_srsChanOrt_ptr = gpu_srsChanOrt + (cellId*num_subband*num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
    
    for (int idx = threadIdx.x; idx < num_bs_ant_port; idx += blockDim.x) {
        srs_info_chanEst[idx] = req_info_ptr->srsInfo[srs_info_idx].srsChanEst[prgIdx*num_ue_ant_port*num_bs_ant_port + srs_info_ant_port*num_bs_ant_port + idx];
    }
    __syncthreads();

    for (int rIdx = 0; rIdx < num_rnd; rIdx++) {
        cuComplex innerProduct = make_cuComplex(0.0f, 0.0f);
        float corrVal = -1.0f;

        uint16_t real_uIdx_in_block = portIdx/MAX_NUM_UE_ANT_PORT + rIdx*(blockDim.x/MAX_NUM_UE_ANT_PORT);
        
        if ((real_uIdx_in_block + first_ue_info_idx_in_block) <= last_ue_info_idx_in_block) {
            uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].nUeAnt;

            if (tx_port_idx < num_ue_ant_port_real_uIdx) {
                uint8_t flags_ue_info = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].flags;

                if ((flags_ue_info & 0x04) > 0) { // SRS chanEst available
                    uint16_t real_ue_id = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].id;

                    if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                        uint16_t srsInfoIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].srsInfoIdx;

                        if (row_idx_chanOrtMat == 0) {
                            for (int idx = 0; idx < num_bs_ant_port; idx++) {
                                cuComplex tmp1 = req_info_ptr->srsInfo[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + idx];
                                
                                innerProduct.x += srs_info_chanEst[idx].x*tmp1.x + srs_info_chanEst[idx].y*tmp1.y;
                                innerProduct.y += srs_info_chanEst[idx].x*tmp1.y - srs_info_chanEst[idx].y*tmp1.x;
    
                                chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + idx] = tmp1;
                            }
                        } else {
                            for (int idx = 0; idx < num_bs_ant_port; idx++) {
                                cuComplex tmp1 = req_info_ptr->srsInfo[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + idx];
                                
                                innerProduct.x += srs_info_chanEst[idx].x*tmp1.x + srs_info_chanEst[idx].y*tmp1.y;
                                innerProduct.y += srs_info_chanEst[idx].x*tmp1.y - srs_info_chanEst[idx].y*tmp1.x;
                            }
                        }
                    } else {    
                        for (int idx = 0; idx < num_bs_ant_port; idx++) {
                            cuComplex tmp1 = chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + idx];

                            innerProduct.x += srs_info_chanEst[idx].x*tmp1.x + srs_info_chanEst[idx].y*tmp1.y;
                            innerProduct.y += srs_info_chanEst[idx].x*tmp1.y - srs_info_chanEst[idx].y*tmp1.x;
                        }
                    }
    
                    uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;

                    corrVal = sqrt(innerProduct.x*innerProduct.x + innerProduct.y*innerProduct.y);
                
                    if (row_idx >= col_idx) {
                        gpu_srsChanOrt_ptr[row_idx * (row_idx + 1) / 2 + col_idx] = corrVal;
                    } else {
                        gpu_srsChanOrt_ptr[col_idx * (col_idx + 1) / 2 + row_idx] = corrVal;
                    }    
                } 
            }
        }
    }
}

__global__ void ueGrpKernel_64T64R_chanCorr_v1(uint8_t*    main_in_buf, 
                                               uint8_t*    task_in_buf, 
                                               float*      gpu_srsChanOrt,
                                               uint32_t    task_in_buf_len_per_cell,
                                               int         num_cell,
                                               int         num_bs_ant_port,
                                               int         num_subband,
                                               int         num_prg_samp_per_subband,
                                               int         num_blocks_per_cell,
                                               int         num_blocks_per_row_chanOrtMat)
{
    uint16_t cellId = blockIdx.x / num_blocks_per_cell;
    uint16_t row_idx_chanOrtMat = (blockIdx.x - cellId*num_blocks_per_cell) / num_blocks_per_row_chanOrtMat;
    uint16_t blockIdx_in_row_chanOrtMat = blockIdx.x - cellId*num_blocks_per_cell - row_idx_chanOrtMat*num_blocks_per_row_chanOrtMat;
    uint16_t srs_info_idx = row_idx_chanOrtMat / MAX_NUM_UE_ANT_PORT;
    uint16_t srs_info_ant_port = row_idx_chanOrtMat % MAX_NUM_UE_ANT_PORT;

    cumac_muUeGrp_req_info_t* req_info_ptr = (cumac_muUeGrp_req_info_t*) (task_in_buf + cellId*task_in_buf_len_per_cell);

    cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (req_info_ptr->srsInfo + req_info_ptr->numSrsInfo);

    if (srs_info_idx >= req_info_ptr->numSrsInfo) {
        return;
    }

    uint16_t num_ue_ant_port = req_info_ptr->srsInfo[srs_info_idx].nUeAnt;

    if (srs_info_ant_port >= num_ue_ant_port) {
        return;
    }

    uint16_t ue_info_per_block = req_info_ptr->numUeInfo/num_blocks_per_row_chanOrtMat;
    uint16_t first_ue_info_idx_in_block = ue_info_per_block*blockIdx_in_row_chanOrtMat;
    uint16_t last_ue_info_idx_in_block;
    if (blockIdx_in_row_chanOrtMat == num_blocks_per_row_chanOrtMat - 1) { // last block
        last_ue_info_idx_in_block = req_info_ptr->numUeInfo - 1;
    } else {
        last_ue_info_idx_in_block = first_ue_info_idx_in_block + ue_info_per_block - 1;
    }

    uint16_t portIdx = threadIdx.x/(num_bs_ant_port/2);
    uint16_t tx_port_idx = portIdx%MAX_NUM_UE_ANT_PORT;
    uint16_t rx_port_idx = threadIdx.x%(num_bs_ant_port/2);
    uint16_t num_rnd = ceil(static_cast<float>((last_ue_info_idx_in_block - first_ue_info_idx_in_block + 1)*MAX_NUM_UE_ANT_PORT)/(blockDim.x/(num_bs_ant_port/2)));

    __shared__ cuComplex    srs_info_chanEst[MAX_NUM_BS_ANT_PORT];
    __shared__ cuComplex    ue_info_chanEst[2048];
    __shared__ uint16_t     corrIdx[MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT];
    __shared__ float        corrValues[MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT];

    // update SRS SNRs
    if (srs_info_ant_port == 0 && blockIdx_in_row_chanOrtMat == 0 && threadIdx.x == 0) {
        float* srsSnr_main_buf_ptr = (float*) (main_in_buf + sizeof(cuComplex)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*num_cell + sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*cellId);

        srsSnr_main_buf_ptr[req_info_ptr->srsInfo[srs_info_idx].id] = req_info_ptr->srsInfo[srs_info_idx].srsWbSnr;
    }

    cuComplex* chanEst_main_buf_ptr = (cuComplex*) (main_in_buf + sizeof(cuComplex)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*cellId);

    for (int prgIdx = 0; prgIdx < num_subband*num_prg_samp_per_subband; prgIdx++) {
        for (int idx = threadIdx.x; idx < MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT; idx += blockDim.x) {
            corrIdx[idx] = 0x0000FFFF;
        }
    
        float* gpu_srsChanOrt_ptr = gpu_srsChanOrt + (cellId*num_subband*num_prg_samp_per_subband + prgIdx)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;
            
        if (threadIdx.x < num_bs_ant_port) {
            srs_info_chanEst[threadIdx.x] = req_info_ptr->srsInfo[srs_info_idx].srsChanEst[prgIdx*num_ue_ant_port*num_bs_ant_port + srs_info_ant_port*num_bs_ant_port + threadIdx.x];
        }
        
        for (int rIdx = 0; rIdx < num_rnd; rIdx++) {
            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = make_cuComplex(0.0f, 0.0f);
            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = make_cuComplex(0.0f, 0.0f);
    
            uint16_t real_uIdx_in_block = portIdx/MAX_NUM_UE_ANT_PORT + rIdx*(blockDim.x/(MAX_NUM_UE_ANT_PORT*num_bs_ant_port/2));
        
            uint16_t real_ue_id = 0xFFFF;
    
            if ((real_uIdx_in_block + first_ue_info_idx_in_block) <= last_ue_info_idx_in_block) {
                uint16_t num_ue_ant_port_real_uIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].nUeAnt;
    
                if (tx_port_idx < num_ue_ant_port_real_uIdx) {
                    uint8_t flags_ue_info = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].flags;
    
                    if ((flags_ue_info & 0x04) > 0) { // SRS chanEst available
                        real_ue_id = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].id;
                        
                        if ((flags_ue_info & 0x08) > 0) { // has updated SRS info in the current slot
                            uint16_t srsInfoIdx = ueInfo_ptr[real_uIdx_in_block + first_ue_info_idx_in_block].srsInfoIdx;
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = req_info_ptr->srsInfo[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + rx_port_idx];
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = req_info_ptr->srsInfo[srsInfoIdx].srsChanEst[prgIdx*num_ue_ant_port_real_uIdx*num_bs_ant_port + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)];
                            if (row_idx_chanOrtMat == 0) {
                                chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx] = ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx];
                                chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)];
                            }
                        } else {    
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx];
                            ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = chanEst_main_buf_ptr[real_ue_id*num_subband*num_prg_samp_per_subband*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + prgIdx*num_bs_ant_port*MAX_NUM_UE_ANT_PORT + tx_port_idx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)];
                        }

                        cuComplex innerProduct;
                        innerProduct.x = srs_info_chanEst[rx_port_idx].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x + srs_info_chanEst[rx_port_idx].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y;
                        innerProduct.y = srs_info_chanEst[rx_port_idx].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y - srs_info_chanEst[rx_port_idx].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x;

                        ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx] = innerProduct;

                        innerProduct.x = srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].x + srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].y;
                        innerProduct.y = srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].x*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].y - srs_info_chanEst[rx_port_idx + (num_bs_ant_port/2)].y*ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)].x;

                        ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + (num_bs_ant_port/2)] = innerProduct;
                    }
                }
            }
            __syncthreads();
    
            // parallel reduction
            uint16_t h = num_bs_ant_port;
            uint16_t s = ceilf(h*0.5f);

            while(s > 1) {
                if(rx_port_idx < (h - s)) {
                    ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].x += ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + s].x;
                    ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx].y += ue_info_chanEst[portIdx*num_bs_ant_port + rx_port_idx + s].y;
                }
                h = s; 
                s = ceilf(h*0.5f);
    
                __syncthreads();
            }
    
            if (rx_port_idx == 0) {
                ue_info_chanEst[portIdx*num_bs_ant_port].x += ue_info_chanEst[portIdx*num_bs_ant_port + 1].x;
                ue_info_chanEst[portIdx*num_bs_ant_port].y += ue_info_chanEst[portIdx*num_bs_ant_port + 1].y;
    
                if (real_ue_id != 0xFFFF) {
                    uint16_t col_idx = real_ue_id*MAX_NUM_UE_ANT_PORT + tx_port_idx;
                    uint16_t row_idx = req_info_ptr->srsInfo[srs_info_idx].id*MAX_NUM_UE_ANT_PORT + srs_info_ant_port;
    
                    if (row_idx >= col_idx) {
                        corrIdx[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = row_idx * (row_idx + 1) / 2 + col_idx;
                    } else {
                        corrIdx[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = col_idx * (col_idx + 1) / 2 + row_idx;
                    }
    
                    corrValues[real_uIdx_in_block*MAX_NUM_UE_ANT_PORT + tx_port_idx] = sqrt((ue_info_chanEst[portIdx*num_bs_ant_port].x*ue_info_chanEst[portIdx*num_bs_ant_port].x + ue_info_chanEst[portIdx*num_bs_ant_port].y*ue_info_chanEst[portIdx*num_bs_ant_port].y));
                }
            }
        }
        __syncthreads();
    
        for (int idx = threadIdx.x; idx < MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT; idx += blockDim.x) {
            if (corrIdx[idx] != 0x0000FFFF) {
                gpu_srsChanOrt_ptr[corrIdx[idx]] = corrValues[idx];
            }
        }
        __syncthreads();
    }
}

__global__ void ueGrpKernel_64T64R_ueGrp(uint8_t*    main_in_buf, 
                                         uint8_t*    task_in_buf, 
                                         uint8_t*    task_out_buf,
                                         float*      gpu_srsChanOrt,
                                         uint16_t    num_bs_ant_port,
                                         uint16_t    num_subband,
                                         uint16_t    num_prg_samp_per_subband,
                                         uint32_t    task_in_buf_len_per_cell,
                                         uint32_t    task_out_buf_len_per_cell)
{
    uint32_t                    cellId                  = blockIdx.x;

    cumac_muUeGrp_req_info_t*   req_info_ptr            = (cumac_muUeGrp_req_info_t*) (task_in_buf + cellId*task_in_buf_len_per_cell);

    float*                      srsSnr_main_buf_ptr     = (float*) (main_in_buf + 
                                                          sizeof(cuComplex)*num_bs_ant_port*MAX_NUM_UE_ANT_PORT*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*gridDim.x + 
                                                          sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*cellId);

    float*                      gpu_srsChanOrt_cell_ptr = gpu_srsChanOrt + 
                                                          cellId*num_subband*num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*
                                                          (MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

    cumac_muUeGrp_resp_info_t*  resp_info_ptr           = (cumac_muUeGrp_resp_info_t*) (task_out_buf + cellId*task_out_buf_len_per_cell);

    __shared__ float    weights[512];
    __shared__ uint16_t ueIds[512];
    __shared__ uint16_t ueRnti[MAX_NUM_SRS_UE_PER_CELL];
    __shared__ uint16_t muMimoInd[MAX_NUM_SRS_UE_PER_CELL];
    __shared__ uint16_t nUeAnt[MAX_NUM_SRS_UE_PER_CELL];

    if (threadIdx.x < 512) {
        weights[threadIdx.x] = -1.0;
        ueIds[threadIdx.x] = 0xFFFF;
    }
    
    cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (req_info_ptr->srsInfo + req_info_ptr->numSrsInfo);

    if (threadIdx.x < req_info_ptr->numUeInfo) {
        uint8_t flags   = ueInfo_ptr[threadIdx.x].flags;
        uint16_t ue_id  = ueInfo_ptr[threadIdx.x].id;
        nUeAnt[ue_id]   = ueInfo_ptr[threadIdx.x].nUeAnt;
        ueRnti[ue_id]   = ueInfo_ptr[threadIdx.x].rnti;
        if ((flags & 0x01) > 0 && ueInfo_ptr[threadIdx.x].bufferSize > 0) { // valid UE info
            weights[threadIdx.x] = static_cast<float>(ueInfo_ptr[threadIdx.x].currRate)/ueInfo_ptr[threadIdx.x].avgRate;
            ueIds[threadIdx.x] = ue_id;

            if ((flags & 0x02) > 0) { // new TX indication
                if ((flags & 0x04) > 0) { // SRS chanEst available
                    if (srsSnr_main_buf_ptr[ue_id] >= req_info_ptr->srsSnrThr) {
                        muMimoInd[ue_id] = 1;
                    } else {
                        muMimoInd[ue_id] = 0;
                    }
                } else {
                    muMimoInd[ue_id] = 0;
                }
            } else { // re-TX
                muMimoInd[ue_id] = 0;
            }
        } else {
            muMimoInd[ue_id] = 0;
        }
    }
    __syncthreads();

    // Sorting
    uint32_t minPow2 = 2;
    while (minPow2 < req_info_ptr->numUeInfo) {
        minPow2 *= 2;
    }
    bitonicSort<float, uint16_t>(weights, ueIds, minPow2);

    // load correlation values from global memory to shared memory
    __shared__ uint8_t orth_ind[MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_UE_FOR_GRP_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2];
    int totNumElem = req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT*(req_info_ptr->numUeForGrpPerCell*MAX_NUM_UE_ANT_PORT+1)/2;

    for (uint32_t subbandIdx = 0; subbandIdx < num_subband; subbandIdx++) {
        for (uint32_t idx = threadIdx.x; idx < totNumElem; idx += blockDim.x) {
            orth_ind[idx] = 0xFF;

            uint32_t row_idx = 0;
            while (row_idx*(row_idx+1)/2 <= idx) {row_idx++;}
            row_idx--;
            uint32_t col_idx = idx-row_idx*(row_idx+1)/2;

            uint32_t ue_idx_row = row_idx/MAX_NUM_UE_ANT_PORT;
            uint32_t ue_idx_col = col_idx/MAX_NUM_UE_ANT_PORT;
            uint32_t ant_port_row = row_idx - ue_idx_row*MAX_NUM_UE_ANT_PORT;
            uint32_t ant_port_col = col_idx - ue_idx_col*MAX_NUM_UE_ANT_PORT;
            uint32_t ue_id_row = ueIds[ue_idx_row];
            uint32_t ue_id_col = ueIds[ue_idx_col];

            if (ue_id_row != 0xFFFF && ue_id_col != 0xFFFF && ant_port_row < nUeAnt[ue_id_row] && ant_port_col < nUeAnt[ue_id_col]) {
                uint32_t row_idx_main = ue_id_row*MAX_NUM_UE_ANT_PORT + ant_port_row;
                uint32_t col_idx_main = ue_id_col*MAX_NUM_UE_ANT_PORT + ant_port_col;

                float corrValues;

                for (uint32_t prgIdx = 0; prgIdx < num_prg_samp_per_subband; prgIdx++) {
                    float* gpu_srsChanOrt_ptr = gpu_srsChanOrt_cell_ptr + 
                                                (subbandIdx*num_prg_samp_per_subband + prgIdx)*
                                                MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2;

                    if (row_idx_main >= col_idx_main) {
                        corrValues = gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + col_idx_main];
                    } else {
                        corrValues = gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + row_idx_main];
                    }
                    corrValues /= sqrt(gpu_srsChanOrt_ptr[row_idx_main*(row_idx_main+1)/2 + row_idx_main]*gpu_srsChanOrt_ptr[col_idx_main*(col_idx_main+1)/2 + col_idx_main]);

                    if (corrValues == 1.0f) {
                        orth_ind[idx] = 1;
                    } else if(corrValues > req_info_ptr->chanCorrThr) {
                        orth_ind[idx] = 0;
                        break;
                    } else {
                        orth_ind[idx] = 1;
                    }
                }
            }
        }
        __syncthreads();

        // below is only reference code for determining UE pairing solution based on the computed channel orthogonality indication matrix orth_ind
        if (threadIdx.x == 0) {
            if (subbandIdx == 0) {
                resp_info_ptr->numSchdUeg = num_subband;
            }

            // PRB allocation
            resp_info_ptr->schdUegInfo[subbandIdx].allocPrgStart = req_info_ptr->nPrbGrp/num_subband*subbandIdx;
            resp_info_ptr->schdUegInfo[subbandIdx].allocPrgEnd = subbandIdx == (num_subband-1) ? req_info_ptr->nPrbGrp : (req_info_ptr->nPrbGrp/num_subband)*(subbandIdx + 1);

            uint16_t num_ue_schd = 0;
            for (int uIdx = 0; uIdx < req_info_ptr->numUeForGrpPerCell; uIdx++) {
                uint16_t ue_id = ueIds[uIdx];
                if (ue_id != 0xFFFF) {
                    if (num_ue_schd == 0) { // first UE in the UEG
                        resp_info_ptr->schdUegInfo[subbandIdx].numUeInGrp = 1;
                        resp_info_ptr->schdUegInfo[subbandIdx].flags = 0x01;
                    } else {
                        if (muMimoInd[ue_id] == 1) { // MU-MIMO feasible UE
                            resp_info_ptr->schdUegInfo[subbandIdx].numUeInGrp++;
                        } else {
                            continue;
                        }
                    }
                        
                    resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].rnti = ueRnti[ue_id];
                    resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].id = ue_id;
                    if (orth_ind[uIdx*MAX_NUM_UE_ANT_PORT*(uIdx*MAX_NUM_UE_ANT_PORT+1)/2 + uIdx*MAX_NUM_UE_ANT_PORT] == 1) {
                        resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].layerSel = 0x01;
                    } else {
                        resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].layerSel = 0x00;
                    }
                    resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].ueOrderInGrp = num_ue_schd;
                    resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].nSCID = 0;
                    resp_info_ptr->schdUegInfo[subbandIdx].ueInfo[num_ue_schd].flags = 0x01;

                    ueIds[uIdx] = 0xFFFF;

                    if (num_ue_schd == 0 && muMimoInd[ue_id] != 1) { // SU-MIMO feasible UE
                        break;
                    }

                    num_ue_schd++;

                    if (num_ue_schd >= req_info_ptr->nMaxUeSchdPerCellTTI) {
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
}

int main(int argc, char** argv)
{
    // cuMAC worker (sender) thread
    NVLOGC(TAG, "cuMAC: parameter config YAML file: %s", YAML_PARAM_CONFIG_PATH);
    NVLOGC(TAG, "cuMAC: cuMAC primary NVIPC config YAML file: %s", YAML_CUMAC_NVIPC_CONFIG_PATH);

    // simulation parameters
    sys_param_t sys_param(YAML_PARAM_CONFIG_PATH);
    cuMAC_WORKER_THREAD_CORE = sys_param.cumac_worker_thread_core;
    cuMAC_RECV_THREAD_CORE = sys_param.cumac_recv_thread_core;
    NUM_TIME_SLOTS = sys_param.num_time_slots;
    NUM_CELL = sys_param.num_cell;

    // Load nvipc configuration from YAML file
    nv_ipc_config_t config;
    load_nv_ipc_yaml_config(&config, YAML_CUMAC_NVIPC_CONFIG_PATH, NV_IPC_MODULE_PRIMARY);

    // Initialize NVIPC interface and connect to the cuMAC-CP
    if ((ipc = create_nv_ipc_interface(&config)) == NULL)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: create cuMAC primary IPC interface failed", __func__);
        return -1;
    }

    // Initialize semaphore for GPU task finish notification
    sem_init(&task_sem, 0, 0);

    // Initialize CUDA stream for cuMAC
    CHECK_CUDA_ERR(cudaStreamCreate(&cuStrmCumac));

    std::unique_ptr<cumac_CUDA_kernel_param_t> cuda_kernel_param = std::make_unique<cumac_CUDA_kernel_param_t>(sys_param);

    // Initialize GPU main buffer to store UE_FREQ_ACCESS data
    uint8_t* gpu_main_in_buf; // GPU global memory for storting SRS channel estimates and SRS SNRs per SRS-enabled UEs in each cell
    uint8_t* gpu_out_buf;
    float* gpu_srsChanOrt;
    uint32_t gpu_out_buf_len_per_cell = sizeof(cumac_muUeGrp_resp_info_t);
    CHECK_CUDA_ERR(cudaMalloc(&gpu_main_in_buf, sizeof(cuComplex)*sys_param.num_bs_ant_port*MAX_NUM_UE_ANT_PORT*sys_param.num_subband*sys_param.num_prg_samp_per_subband*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell + sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMalloc(&gpu_out_buf, gpu_out_buf_len_per_cell*sys_param.num_cell));
    CHECK_CUDA_ERR(cudaMalloc(&gpu_srsChanOrt, sizeof(float)*MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT*(MAX_NUM_SRS_UE_PER_CELL*MAX_NUM_UE_ANT_PORT+1)/2*sys_param.num_subband*sys_param.num_prg_samp_per_subband*sys_param.num_cell));
    // GPU global memory allocation for storting computed channel correlation values (lower triangular matrix)
    
    // Initialize lock-free ring pool for tasks
    test_task_ring = new nv::lock_free_ring_pool<test_task_t>("test_task", LEN_LOCKFREE_RING_POOL, sizeof(test_task_t));
    uint32_t ring_len = test_task_ring->get_ring_len();
    for (int i = 0; i < ring_len; i++)
    {
        test_task_t *task = test_task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE(TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring lengh: i=%d length=%d", i, ring_len);
            return -1;
        }

        alloc_mem_test_task(task, sys_param.num_cell);
    }

    // Create cuMAC receiver thread
    pthread_t thread_id;
    int ret = pthread_create(&thread_id, NULL, cumac_blocking_recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret=%d", __func__, ret);
        return -1;
    }

    // Set cuMAC worker thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumac_worker");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(cuMAC_WORKER_THREAD_CORE);

    int task_count = 0;
    while(task_count < NUM_TIME_SLOTS) {
        // Wait for GPU processing finish notification by semaphore task_sem
        sem_wait(&task_sem);

        // Create CUDA events
        cudaEvent_t startCopyH2D, stopCopyH2D;
        cudaEvent_t startKernel, stopKernel;
        cudaEvent_t startCopyD2H, stopCopyD2H;
        cudaEventCreate(&startCopyH2D);
        cudaEventCreate(&stopCopyH2D);
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventCreate(&startCopyD2H);
        cudaEventCreate(&stopCopyD2H);

        struct timespec cumac_prepare_start, cumac_prepare_end;
        clock_gettime(CLOCK_REALTIME, &cumac_prepare_start);

        // Dequeue task from lock-free task ring
        test_task_t* task = test_task_ring->dequeue();
        if (task == nullptr)
        {
            // No task to process
            NVLOGI(TAG, "%s: no task to process", __func__);
            continue;
        }

        // prepare response message and allocate NVIPC buffer
        // Alloc NVIPC buffer: msg_buf will be allocated by default. Set data_pool to get data_buf
            // define CPU data buffer pointers for out
        std::vector<uint8_t*> out_data_buf(task->num_cell);
        nv_ipc_msg_t send_msg[task->num_cell];
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            send_msg[cIdx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;

            // Allocate NVIPC buffer which contains MSG part and DATA part
            if(ipc->tx_allocate(ipc, &send_msg[cIdx], 0) != 0) {
                NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
                return -1;
            }

            // MSG part
            cumac_muUeGrp_resp_msg_t* resp = (cumac_muUeGrp_resp_msg_t*) send_msg[cIdx].msg_buf;
            out_data_buf[cIdx] = (uint8_t*) send_msg[cIdx].data_buf;

            resp->sfn = task->sfn;
            resp->slot = task->slot;
            resp->offsetData = 0;

            // Update the msg_len and data_len of the NVIPC message header
            send_msg[cIdx].msg_id = CUMAC_SCH_TTI_RESPONSE;
            send_msg[cIdx].cell_id = task->recv_msg[cIdx].cell_id;
            send_msg[cIdx].msg_len = sizeof(cumac_muUeGrp_resp_msg_t);
            send_msg[cIdx].data_len = sizeof(cumac_muUeGrp_resp_info_t);
        }

        clock_gettime(CLOCK_REALTIME, &cumac_prepare_end);
        int64_t cumac_prepare_duration = nvlog_timespec_interval(&cumac_prepare_start, &cumac_prepare_end);
        NVLOGC(TAG, "cuMAC Slot preparation duration: %f microseconds", cumac_prepare_duration/1000.0);

        // Copy data from NVIPC buffer to GPU buffer
        cudaEventRecord(startCopyH2D);            
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            CHECK_CUDA_ERR(cudaMemcpyAsync(task->gpu_buf + cIdx*task->gpu_buf_len_per_cell, task->recv_msg[cIdx].data_buf, task->recv_msg[cIdx].data_len, cudaMemcpyHostToDevice, task->strm));
        }
        cudaEventRecord(stopCopyH2D);
        cudaEventSynchronize(stopCopyH2D);

        // Launch PFM sorting kernel
        cudaEventRecord(startKernel);
        for (int rIdx = 0; rIdx < NUM_ITR_KERNEL_RUN; rIdx++) {
#ifdef muUeGrpKernel_v1
            ueGrpKernel_64T64R_chanCorr_v1<<<cuda_kernel_param->k1_num_blocks_grid, cuda_kernel_param->k1_num_threads_per_block, 0, task->strm>>>(gpu_main_in_buf, task->gpu_buf, gpu_srsChanOrt,
                task->gpu_buf_len_per_cell, task->num_cell, cuda_kernel_param->num_bs_ant_port, cuda_kernel_param->num_subband, cuda_kernel_param->num_prg_samp_per_subband, cuda_kernel_param->k1_num_blocks_per_cell, cuda_kernel_param->k1_num_blocks_per_row_chanOrtMat);
#endif
#ifdef muUeGrpKernel_v2
            ueGrpKernel_64T64R_chanCorr_v2<<<cuda_kernel_param->k1_num_blocks_grid, cuda_kernel_param->k1_num_threads_per_block, 0, task->strm>>>(gpu_main_in_buf, task->gpu_buf, gpu_srsChanOrt,
                task->gpu_buf_len_per_cell, task->num_cell, cuda_kernel_param->num_bs_ant_port, cuda_kernel_param->num_subband, cuda_kernel_param->num_prg_samp_per_subband, cuda_kernel_param->k1_num_blocks_per_prg, cuda_kernel_param->k1_num_blocks_per_cell, cuda_kernel_param->k1_num_blocks_per_row_chanOrtMat);
#endif
            ueGrpKernel_64T64R_ueGrp<<<cuda_kernel_param->k2_num_blocks_grid, cuda_kernel_param->k2_num_threads_per_block, 0, task->strm>>>(gpu_main_in_buf, task->gpu_buf, gpu_out_buf, gpu_srsChanOrt, cuda_kernel_param->num_bs_ant_port, cuda_kernel_param->num_subband, cuda_kernel_param->num_prg_samp_per_subband, task->gpu_buf_len_per_cell, gpu_out_buf_len_per_cell);
        }
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);

        cudaEventRecord(startCopyD2H);
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            CHECK_CUDA_ERR(cudaMemcpyAsync(out_data_buf[cIdx], gpu_out_buf + cIdx*gpu_out_buf_len_per_cell, send_msg[cIdx].data_len, cudaMemcpyDeviceToHost, task->strm));
        }
        cudaEventRecord(stopCopyD2H);
        cudaEventSynchronize(stopCopyD2H);

        CHECK_CUDA_ERR(cudaStreamSynchronize(task->strm));

        // calculate timings
        float timeH2D, timeKernel, timeD2H;

        cudaEventElapsedTime(&timeH2D, startCopyH2D, stopCopyH2D);
        printf("Host to Device copy time: %f microseconds\n", timeH2D*1000.0);

        cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);
        cumac_muUeGrp_req_info_t* req_ptr = (cumac_muUeGrp_req_info_t*) task->recv_msg[0].data_buf;
        printf("Kernel execution time: %f microseconds, numSrsInfo = %d\n", timeKernel*1000.0/NUM_ITR_KERNEL_RUN, req_ptr->numSrsInfo);

        cudaEventElapsedTime(&timeD2H, startCopyD2H, stopCopyD2H);
        printf("Device to Host copy time: %f microseconds\n", timeD2H*1000.0);
        printf("Total Device-Host copy time: %f microseconds\n", timeH2D*1000.0 + timeD2H*1000.0);

        struct timespec msg_send_start, msg_send_end;
        clock_gettime(CLOCK_REALTIME, &msg_send_start);

        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            // send cuMAC schedule result to L2
            // Send the message
            //NVLOGC(TAG, "cuMAC SEND: SFN = %u.%u, cell ID = %u, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                //task->sfn, task->slot, task->recv_msg[cIdx].cell_id, send_msg[cIdx].msg_id, get_cumac_msg_name(send_msg[cIdx].msg_id), send_msg[cIdx].msg_len, send_msg[cIdx].data_len);
            
            if(ipc->tx_send_msg(ipc, &send_msg[cIdx]) < 0) {
                NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
                return -1;
            }
        }

        if(ipc->tx_tti_sem_post(ipc) < 0) {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
            return -1;
        }

        clock_gettime(CLOCK_REALTIME, &msg_send_end);
        int64_t msg_send_duration = nvlog_timespec_interval(&msg_send_start, &msg_send_end);
        NVLOGC(TAG, "cuMAC SEND: NVIPC message send duration: %f microseconds", msg_send_duration/1000.0);

        // Release the NVIPC message buffer
        for (int cIdx = 0; cIdx < task->num_cell; cIdx++) {
            ipc->rx_release(ipc, &task->recv_msg[cIdx]);
        }

        // Task is finished, free the task buffer
        test_task_ring->free(task);
        task_count++;
    }

    CHECK_CUDA_ERR(cudaFree(gpu_main_in_buf));
    CHECK_CUDA_ERR(cudaFree(gpu_out_buf));
    CHECK_CUDA_ERR(cudaFree(gpu_srsChanOrt));
    for (int i = 0; i < ring_len; i++) {
        CHECK_CUDA_ERR(cudaFree(test_task_ring->get_buf_addr(i)->gpu_buf));
    }

    printf("cuMAC sender/worker process: test completed successfully\n");
    return 0;
}