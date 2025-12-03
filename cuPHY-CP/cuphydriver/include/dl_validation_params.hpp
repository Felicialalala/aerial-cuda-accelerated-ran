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

#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include "cuphydriver_api.hpp"
#include "aerial-fh-driver/api.hpp"

#ifndef DL_VALIDATION_PARAMS_H
#define DL_VALIDATION_PARAMS_H

/**
 * @brief Holds ORAN timing validation parameters for downlink processing across multiple cells
 * 
 * This class stores and manages ORAN timing parameters (T1a, Tcp) used for DL validation testing.
 * It maintains per-cell timing values for C-plane and U-plane advance times.
 * - Stores per-cell ORAN timing parameters (T1a, Tcp)
 * - Enables dedicated validation worker threads to monitor fronthaul packet arrival times
 * - Classifies packets as EARLY, ONTIME, or LATE based on ORAN timing windows
 * - Collects statistics for performance analysis and timing verification
 */
class DLValidationParams {
public:
    /**
     * @brief Construct DL validation parameters
     * 
     * @param _pdh         Physical layer driver handle
     * @param _start_cell  Starting cell index for validation
     * @param _num_cells   Number of cells to validate
     */
    DLValidationParams(phydriver_handle _pdh, int _start_cell, int _num_cells);
    
    /**
     * @brief Destructor
     */
    ~DLValidationParams();
    
    /**
     * @brief Get physical layer driver handle
     * 
     * @return phydriver_handle  Driver handle for PHY layer operations
     */
    phydriver_handle    getPhyDriverHandler(void) const;
    
    /**
     * @brief Set physical layer driver handle
     * 
     * @param pdh_  New driver handle to set
     */
    void setPhyDriverHandler(phydriver_handle pdh_);
    
    /**
     * @brief Get starting cell index for validation
     * 
     * @return int  First cell index in validation range
     */
    int    getStartCell(void) const{ return start_cell; };
    
    /**
     * @brief Get number of cells being validated
     * 
     * @return int  Total number of cells in validation scope
     */
    int    getNumCells(void) const { return num_cells; };
    
    int cell_T1a_max_up_ns[API_MAX_NUM_CELLS];     ///< Per-cell T1a max: Max advance time for DL U-plane to arrive before transmission (ns)
    int cell_T1a_max_cp_ul_ns[API_MAX_NUM_CELLS];  ///< Per-cell T1a max: Max advance time for UL C-plane to arrive (ns)
    int cell_Tcp_adv_dl_ns[API_MAX_NUM_CELLS];     ///< Per-cell Tcp advance: Time to send DL C-plane before DL U-plane (ns)
protected:
    phydriver_handle                         pdh;         ///< Physical layer driver handle for validation operations
    int start_cell;                                       ///< Starting cell index in validation range (0-based)
    int num_cells;                                        ///< Total number of cells to validate

};

#endif