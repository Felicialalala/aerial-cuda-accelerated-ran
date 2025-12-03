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

#ifndef AERIAL_ring_BUFFER_HPP__
#define AERIAL_ring_BUFFER_HPP__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"

namespace aerial_fh
{
class Fronthaul;

/**
 * Lock-free ring buffer for inter-thread communication
 *
 * Wrapper around DPDK rte_ring providing thread-safe FIFO queue operations.
 * Supports both single-producer/single-consumer (SP/SC) and
 * multi-producer/multi-consumer (MP/MC) configurations.
 */
class RingBuffer {
public:
    /**
     * Constructor
     * @param[in] fhi Fronthaul instance handle
     * @param[in] info Ring buffer configuration
     */
    RingBuffer(Fronthaul* fhi, RingBufferInfo const* info);

    /**
     * Destructor - frees ring buffer resources
     */
    ~RingBuffer();

    /**
     * Get the associated fronthaul instance
     * @return Pointer to fronthaul instance
     */
    Fronthaul* get_fronthaul() const;

    /**
     * Enqueue a single object (blocking if full in SP/SC mode)
     * @param[in] obj Object pointer to enqueue
     * @return 0 on success, negative on error
     */
    int        enqueue(void* obj);

    /**
     * Enqueue multiple objects in bulk (blocks until all enqueued)
     * @param[in] objs Array of object pointers
     * @param[in] count Number of objects to enqueue
     * @return Number of objects actually enqueued
     */
    size_t     enqueue_bulk(void* const* objs, size_t count);

    /**
     * Enqueue multiple objects in burst (non-blocking, best effort)
     * @param[in] objs Array of object pointers
     * @param[in] count Maximum number of objects to enqueue
     * @return Number of objects actually enqueued (may be less than count)
     */
    size_t     enqueue_burst(void* const* objs, size_t count);

    /**
     * Dequeue a single object (blocking if empty in SP/SC mode)
     * @param[out] obj Pointer to store dequeued object
     * @return 0 on success, negative on error
     */
    int        dequeue(void** obj);

    /**
     * Dequeue multiple objects in bulk (blocks until all dequeued)
     * @param[out] objs Array to store dequeued object pointers
     * @param[in] count Number of objects to dequeue
     * @return Number of objects actually dequeued
     */
    size_t     dequeue_bulk(void** objs, size_t count);

    /**
     * Dequeue multiple objects in burst (non-blocking, best effort)
     * @param[out] objs Array to store dequeued object pointers
     * @param[in] count Maximum number of objects to dequeue
     * @return Number of objects actually dequeued (may be less than count)
     */
    size_t     dequeue_burst(void** objs, size_t count);

    /**
     * Get number of free slots in ring buffer
     * @return Number of available slots
     */
    size_t     free_count() const;

    /**
     * Check if ring buffer is full
     * @return true if full, false otherwise
     */
    bool       full() const;

    /**
     * Check if ring buffer is empty
     * @return true if empty, false otherwise
     */
    bool       empty() const;

protected:
    Fronthaul*     fhi_;      //!< Associated fronthaul instance
    rte_ring*      ring_{};   //!< DPDK ring buffer handle
    RingBufferInfo info_;     //!< Ring buffer configuration
    unsigned int   id_;       //!< Ring buffer ID
};

} // namespace aerial_fh

#endif //ifndef AERIAL_ring_BUFFER_HPP__