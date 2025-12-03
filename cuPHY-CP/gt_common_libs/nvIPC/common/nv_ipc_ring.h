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

#ifndef _NV_IPC_RING_H_
#define _NV_IPC_RING_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** Ring buffer type */
typedef enum {
    RING_TYPE_SHM_SECONDARY = 0, //!< Shared memory secondary process
    RING_TYPE_SHM_PRIMARY = 1,   //!< Shared memory primary process
    RING_TYPE_APP_INTERNAL = 2,  //!< Application internal memory (no sharing)
    RING_TYPE_INVALID = 3,       //!< Invalid type
} ring_type_t;

/**
 * IPC ring buffer interface
 *
 * Provides lock-free ring buffer for message passing
 */
typedef struct nv_ipc_ring_t nv_ipc_ring_t;
struct nv_ipc_ring_t
{
    int32_t (*alloc)(nv_ipc_ring_t* ring);           //!< Allocate buffer from ring
    int (*free)(nv_ipc_ring_t* ring, int32_t index); //!< Free buffer by index
    int (*get_index)(nv_ipc_ring_t* ring, void* buf); //!< Get index from buffer pointer
    void* (*get_addr)(nv_ipc_ring_t* ring, int32_t index); //!< Get buffer address by index
    int (*enqueue_by_index)(nv_ipc_ring_t* ring, int32_t index);  //!< Enqueue buffer by index
    int32_t (*dequeue_by_index)(nv_ipc_ring_t* ring);  //!< Dequeue buffer, returns index
    int (*get_buf_size)(nv_ipc_ring_t* ring);         //!< Get buffer size
    int (*enqueue)(nv_ipc_ring_t* ring, void* obj);   //!< Enqueue: alloc, copy, enqueue
    int (*dequeue)(nv_ipc_ring_t* ring, void* obj);   //!< Dequeue: copy, dequeue, free
    int (*close)(nv_ipc_ring_t* ring);                //!< Close ring
    int (*get_count)(nv_ipc_ring_t* ring);            //!< Get enqueued count (Not thread safe, may be changed by other threads concurrently)
    int (*get_free_count)(nv_ipc_ring_t* ring);       //!< Get free buffer count (Not thread safe, may be changed by other threads concurrently)
    int (*get_ring_len)(nv_ipc_ring_t* ring);         //!< Get ring length
};

/**
 * Open ring buffer
 *
 * @param[in] type Ring type
 * @param[in] name Ring name
 * @param[in] rin_len Ring length (number of buffers)
 * @param[in] buf_size Buffer size in bytes
 * @return Pointer to ring on success, NULL on failure
 */
nv_ipc_ring_t* nv_ipc_ring_open(ring_type_t type, const char* name, int32_t rin_len, int32_t buf_size);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_RING_H_ */
