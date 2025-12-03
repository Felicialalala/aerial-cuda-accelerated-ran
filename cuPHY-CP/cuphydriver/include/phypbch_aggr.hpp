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

#ifndef PHY_PBCH_AGGR_CHANNEL_H
#define PHY_PBCH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"

/**
 * @class PhyPbchAggr
 * @brief Handles PBCH (Physical Broadcast Channel) / SSB (Synchronization Signal Block) transmission processing.
 *
 * This class implements the downlink PBCH/SSB transmission pipeline using the cuPHY library.
 * PBCH carries the Master Information Block (MIB) and is transmitted as part of the SSB,
 * which includes PSS, SSS, and PBCH signals for cell synchronization and system information broadcast.
 * Supports aggregated multi-cell processing with both CUDA graph and stream-based execution modes.
 */
class PhyPbchAggr : public PhyChannel {
public:
    /**
     * @brief Creates the cuPHY SSB TX object.
     *
     * Initializes the cuPHY SSB transmission handle, builds the cell list, allocates
     * output buffers, and configures static parameters. Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int                             createPhyObj();
    
    /**
     * @brief Configures the SSB transmission for the current slot.
     *
     * Sets up dynamic parameters including MIB data, output buffer pointers, per-cell
     * SSB configuration, per-block parameters, and precoding matrices. Calls cuphySetupSsbTx().
     *
     * @param aggr_dlbuf List of downlink output buffers (one per cell) for SSB waveform output
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @return 0 on success, -1 on failure (setup error)
     */
    int                             setup(const std::vector<DLOutputBuffer *>& aggr_dlbuf, const std::vector<Cell *>& aggr_cell_list);
    
    /**
     * @brief Executes the SSB transmission processing on GPU.
     *
     * Invokes cuphyRunSsbTx() to perform PBCH encoding, modulation, and SSB waveform generation.
     * Only runs if setup completed successfully. Uses either CUDA graphs or streams based on configuration.
     *
     * @return 0 on success, -1 on failure (run error)
     */
    int                             run();
    
    /**
     * @brief Post-processing callback after SSB transmission completes.
     *
     * Not used for PBCH in the current architecture (returns 0 immediately without being called).
     *
     * @return Always returns 0
     */
    int                             callback();
    
    /**
     * @brief Retrieves the PBCH group dynamic parameters for the current slot.
     *
     * Extracts the PBCH parameters (MIB data, SSB configuration) from the aggregated
     * slot command structure for use in setup().
     *
     * @return Pointer to the PBCH group parameters for the current slot
     */
    slot_command_api::pbch_group_params * getDynParams();

public:
    static constexpr int MAX_MIB_BITS             = 24;                 ///< Maximum number of bits in Master Information Block (MIB payload)
    static constexpr int MAX_MIB_BYTES            = MAX_MIB_BITS >> 3;  ///< Maximum number of bytes in MIB (3 bytes for 24 bits)

public:
    /**
     * @brief Constructs a PhyPbchAggr object.
     *
     * Initializes the PBCH channel with the given cuphydriver handle, GPU device, CUDA stream,
     * and MPS context. Sets up memory footprint tracking and channel metadata.
     *
     * @param _pdh Cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channel CUDA stream for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     */
    explicit PhyPbchAggr(
            phydriver_handle _pdh,
            GpuDevice*       _gDev,
            cudaStream_t     _s_channel,
            MpsCtx * _mpsCtx);
    
    /**
     * @brief Destructor for PhyPbchAggr.
     *
     * Frees allocated output buffers and destroys the cuPHY SSB TX handle.
     */
    ~PhyPbchAggr();

    PhyPbchAggr(const PhyPbchAggr&) = delete;              ///< Deleted copy constructor
    PhyPbchAggr& operator=(const PhyPbchAggr&) = delete;   ///< Deleted copy assignment operator
    // TODO - Move ctor

protected:

    cuphySsbTxHndl_t                   handle;         ///< cuPHY SSB transmission handle (opaque handle to cuPHY SSB TX object)
    std::vector<uint16_t>              cell_id_list;   ///< List of physical cell IDs being processed (built during createPhyObj)
    cuphySsbDynPrms_t                  ssbDynPrms;     ///< cuPHY dynamic parameters for SSB TX (per-slot configuration)
    cuphySsbStatPrms_t                 ssbStatParams;  ///< cuPHY static parameters for SSB TX (setup-time configuration)
    cuphySsbDataIn_t                   DataIn;         ///< cuPHY input data structure (contains MIB payload pointers)
    cuphySsbDataOut_t                  DataOut;        ///< cuPHY output data structure (contains SSB waveform output buffer pointers)
};
#endif
