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

#ifndef _INTEGER_RING_H_
#define _INTEGER_RING_H_

#include <stdint.h>
#include <stdatomic.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define DEBUG_TEST

#define NV_NAME_MAX_LEN 32
#define NV_NAME_SUFFIX_MAX_LEN 16

/** Lock-free array queue header */
typedef struct
{
    atomic_ulong head;      //!< Points to earliest enqueued element
    atomic_ulong tail;      //!< Points to latest enqueued element
    atomic_ulong enq_count; //!< Total enqueue count (for debug)
    atomic_ulong deq_count; //!< Total dequeue count (for debug)
    atomic_ulong queue[];   //!< Queue array data
} array_queue_header_t;

/**
 * Lock-free array queue interface
 *
 * Provides thread-safe queue operations using atomic operations
 */
typedef struct array_queue_t array_queue_t;
struct array_queue_t
{
    int32_t (*get_length)(array_queue_t* queue);       //!< Get queue length
    int (*enqueue)(array_queue_t* queue, int32_t value);    //!< Enqueue value
    int32_t (*dequeue)(array_queue_t* queue);          //!< Dequeue value
    int (*close)(array_queue_t* queue);                //!< Close queue
    char* (*get_name)(array_queue_t* queue);           //!< Get queue name
    int32_t (*get_max_length)(array_queue_t* queue);   //!< Get maximum length
    int32_t (*get_count)(array_queue_t* queue);        //!< Get current count
    int32_t (*get_next)(array_queue_t* queue, int32_t base);  //!< Get next value
};

/**
 * Align number to 8-byte boundary
 *
 * @param[in] num Number to align
 * @return Aligned number
 */
static inline int32_t align_8(int32_t num)
{
    return (num + 7) & 0xFFFFFFF8;
}

/** Calculate memory size needed for queue header. For calculating the size of the header memory pool. */
#define ARRAY_QUEUE_HEADER_SIZE(queue_len) (align_8(sizeof(array_queue_header_t) + sizeof(atomic_ulong) * (queue_len)))

/**
 * Open array queue
 *
 * @param[in] primary Primary process flag
 * @param[in] name Queue name
 * @param[in] header Pre-allocated header memory (size calculated from ARRAY_QUEUE_HEADER_SIZE)
 * @param[in] length Queue length
 * @return Pointer to queue on success, NULL on failure
 */
array_queue_t* array_queue_open(int primary, const char* name, void* header, int32_t length);

/**
 * Reset queue to empty state
 *
 * @param[in] queue Queue to reset
 * @return 0 on success, -1 on failure
 */
int array_queue_reset(array_queue_t* queue);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _INTEGER_RING_H_ */
