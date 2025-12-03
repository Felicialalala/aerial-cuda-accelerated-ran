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

#ifndef ULBUFFER_H
#define ULBUFFER_H

#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include "gpudevice.hpp"
#include "time.hpp"
#include "constant.hpp"
#include "cuphydriver_api.hpp"

/**
 * @brief UL input buffer for received uplink data.
 * 
 * Manages host and device memory buffers for uplink channel data (PUSCH/PUCCH/SRS/PRACH).
 * Provides thread-safe reservation and cleanup mechanisms.
 */
class ULInputBuffer {
public:
    /**
     * @brief Constructs a UL input buffer.
     * 
     * @param _pdh cuPHYdriver handle
     * @param _gDev GPU device structure pointer
     * @param _cell_id Cell ID associated with this buffer
     * @param _size Buffer size in bytes
     */
    ULInputBuffer(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id, size_t _size);
    
    /**
     * @brief Destructor.
     */
    ~ULInputBuffer();

    /**
     * @brief Gets the buffer ID.
     * 
     * @return Buffer unique identifier
     */
    uint64_t            getId() const;
    
    /**
     * @brief Gets the associated cell ID.
     * 
     * @return Cell ID
     */
    cell_id_t           getCellId() const;
    
    /**
     * @brief Reserves the buffer for exclusive use.
     * 
     * @return 0 on success, non-zero if already reserved
     */
    int                 reserve();
    
    /**
     * @brief Releases the buffer reservation.
     */
    void                release();
    
    /**
     * @brief Cleans up the buffer using a CUDA stream.
     * 
     * @param stream CUDA stream for asynchronous cleanup operations
     */
    void                cleanup(cudaStream_t stream);
    
    /**
     * @brief Gets the buffer size.
     * 
     * @return Buffer size in bytes
     */
    size_t              getSize() const;
    
    /**
     * @brief Gets the device buffer pointer.
     * 
     * @return Pointer to device memory buffer
     */
    uint8_t *           getBufD() const;
    
    /**
     * @brief Gets the host buffer pointer.
     * 
     * @return Pointer to host memory buffer
     */
    uint8_t *           getBufH() const;

    MemFoot             mf; ///< Memory footprint tracker

protected:
    uint64_t                    id;        ///< Buffer unique identifier
    phydriver_handle            pdh;       ///< cuPHYdriver handle
    std::unique_ptr<host_buf>   addr_h;    ///< Host buffer (pinned memory)
    std::unique_ptr<dev_buf>    addr_d;    ///< Device buffer (GPU memory)
    size_t                      addr_sz;   ///< Buffer size in bytes
    std::atomic<bool>           active;    ///< Reservation status (true = reserved)
    Mutex                       mlock;     ///< Mutex for thread-safe operations
    cell_id_t                   cell_id;   ///< Associated cell ID
    GpuDevice*                  gDev;      ///< GPU device structure pointer
};

#endif