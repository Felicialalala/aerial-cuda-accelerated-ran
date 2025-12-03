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

#ifndef _NV_IPC_MEMPOOL_H_
#define _NV_IPC_MEMPOOL_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * IPC memory pool interface. Can be CPU memory pool or GPU memory pool, configured by cuda_device_id.
 *
 * Contains CPU SHM block divided into header and body parts.
 * For GPU memory pool, the CPU shared memory body is used for store CUDA IPC info, and the body for buffer
 * allocation is replaced with GPU shared memory block.
 */
typedef struct nv_ipc_mempool_t nv_ipc_mempool_t;
struct nv_ipc_mempool_t
{
    int32_t (*alloc)(nv_ipc_mempool_t* mempool);   //!< Allocate buffer from pool
    int (*free)(nv_ipc_mempool_t* mempool, int32_t index);  //!< Free buffer by index
    int (*get_index)(nv_ipc_mempool_t* mempool, void* buf); //!< Get index from buffer pointer
    void* (*get_addr)(nv_ipc_mempool_t* mempool, int32_t index);  //!< Get buffer address by index
    int (*get_buf_size)(nv_ipc_mempool_t* mempool);  //!< Get buffer size
    int (*get_pool_len)(nv_ipc_mempool_t* mempool);  //!< Get pool length
    int (*close)(nv_ipc_mempool_t* mempool);         //!< Close memory pool
    void* (*get_free_queue)(nv_ipc_mempool_t* mempool);  //!< Get free queue (debug only)
    int (*get_free_count)(nv_ipc_mempool_t* mempool);    //!< Get free buffer count (debug only)
    int (*memcpy_to_host)(nv_ipc_mempool_t* mempool, void* host, const void* device, size_t size);    //!< Copy device to host
    int (*memcpy_to_device)(nv_ipc_mempool_t* mempool, void* device, const void* host, size_t size);  //!< Copy host to device
#ifdef NVIPC_GDRCPY_ENABLE
    int (*poolReInit)(nv_ipc_mempool_t* mempool);  //!< Reinitialize GPU_DATA_POOL
#endif
};

#define NV_IPC_MEMPOOL_NO_CUDA_DEV (-1)        //!< No CUDA device
#define NV_IPC_MEMPOOL_USE_EXT_DOCA_BUFS (-2)  //!< Use external DOCA buffers

/**
 * Open memory pool
 *
 * @param[in] primary Primary process flag
 * @param[in] name Pool name
 * @param[in] buf_size Buffer size in bytes
 * @param[in] pool_len Number of buffers
 * @param[in] cuda_device_id CUDA device ID (use NV_IPC_MEMPOOL_NO_CUDA_DEV for CPU pool)
 * @return Pointer to memory pool on success, NULL on failure
 */
nv_ipc_mempool_t* nv_ipc_mempool_open(int primary, const char* name, int buf_size, int pool_len, int cuda_device_id);

/**
 * Reset memory pool
 *
 * @param[in] mempool Memory pool to reset
 * @return 0 on success, -1 on failure
 */
int nv_ipc_mempool_reset(nv_ipc_mempool_t* mempool);

/**
 * Set external buffers for memory pool
 *
 * @param[in] mempool Memory pool
 * @param[in] gpu_addr GPU buffer address
 * @param[in] cpu_addr CPU buffer address
 * @return 0 on success, -1 on failure
 */
int nv_ipc_mempool_set_ext_bufs(nv_ipc_mempool_t* mempool, void* gpu_addr, void* cpu_addr);

/**
 * Lock memory pages for DMA
 *
 * @param[in] phost Host memory pointer
 * @param[in] size Memory size in bytes
 * @return 0 on success, -1 on failure
 */
int nv_ipc_page_lock(void* phost, size_t size);

/**
 * Unlock memory pages
 *
 * @param[in] phost Host memory pointer
 * @return 0 on success, -1 on failure
 */
int nv_ipc_page_unlock(void* phost);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_MEMPOOL_H_ */
