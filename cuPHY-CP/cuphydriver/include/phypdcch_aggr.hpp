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

#ifndef PHY_PDCCH_AGGR_CHANNEL_H
#define PHY_PDCCH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"

/**
 * @class PhyPdcchAggr
 * @brief Handles PDCCH (Physical Downlink Control Channel) transmission processing for both DL and UL DCIs.
 *
 * This class implements the downlink PDCCH transmission pipeline using the cuPHY library.
 * PDCCH carries Downlink Control Information (DCI) messages that schedule uplink and downlink
 * transmissions for UEs. A single PhyPdcchAggr instance handles either PDCCH_DL (for DL scheduling)
 * or PDCCH_UL (for UL grants), configured at construction time. Supports aggregated multi-cell
 * processing with both CUDA graph and stream-based execution modes.
 */
class PhyPdcchAggr : public PhyChannel {
public:
    /**
     * @brief Constructs a PhyPdcchAggr object.
     *
     * Initializes the PDCCH channel with the given cuphydriver handle, GPU device, CUDA stream,
     * MPS context, and channel type (PDCCH_DL or PDCCH_UL). Sets up memory footprint tracking
     * and channel metadata.
     *
     * @param _pdh cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channel CUDA stream for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     * @param ch Channel type (PDCCH_DL for downlink DCIs, PDCCH_UL for uplink grants)
     */
    PhyPdcchAggr(
        phydriver_handle _pdh,
        GpuDevice*       _gDev,
        cudaStream_t     _s_channel,
        MpsCtx*          _mpsCtx,
        slot_command_api::channel_type ch);

    /**
     * @brief Destructor for PhyPdcchAggr.
     *
     * Frees allocated DCI input/output buffers and destroys the cuPHY PDCCH TX handle.
     */
    ~PhyPdcchAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Configures the PDCCH transmission for the current slot.
     *
     * Sets up dynamic parameters including DCI payloads, CORESET configuration, output buffer
     * pointers, and precoding matrices. Calls cuphySetupPdcchTx().
     *
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @param aggr_dlbuf List of downlink output buffers (one per cell) for PDCCH waveform output
     * @return 0 on success, -1 on failure (setup error)
     */
    int setup(const std::vector<Cell *> &aggr_cell_list, const std::vector<DLOutputBuffer *> &aggr_dlbuf);
    
    /**
     * @brief Executes the PDCCH transmission processing on GPU.
     *
     * Invokes cuphyRunPdcchTx() to perform DCI encoding, polar coding, modulation, and PDCCH
     * waveform generation. Only runs if setup completed successfully. Uses either CUDA graphs
     * or streams based on configuration.
     *
     * @return 0 on success, -1 on failure (run error)
     */
    int run();
    
    /**
     * @brief Post-processing callback after PDCCH transmission completes.
     *
     * Currently has placeholder code for future DL callback integration. Returns 0 without
     * performing any operations in the current implementation.
     *
     * @return Always returns 0
     */
    int callback();
    
    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    /**
     * @brief Creates the cuPHY PDCCH TX object.
     *
     * Initializes the cuPHY PDCCH transmission handle, builds the cell list with static
     * parameters (antennas, PRBs, numerology), allocates DCI input/output buffers, and
     * configures processing mode. Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int createPhyObj();
    
    /**
     * @brief Retrieves the PDCCH group dynamic parameters for the current slot.
     *
     * Extracts the PDCCH parameters (DCI payloads, CORESET configuration) from the aggregated
     * slot command structure for use in setup(). Works for both PDCCH_DL and PDCCH_UL.
     *
     * @return Pointer to the PDCCH group parameters for the current slot, nullptr on error
     */
    slot_command_api::pdcch_group_params* getDynParams();

public:
    static constexpr int MAX_XIN_BITS             = 46;    ///< Maximum number of information bits for DCI (before polar coding)
    static constexpr int MAX_POLAR_CODED_SEQUENCE = 512;   ///< Maximum length of polar-coded sequence for PDCCH
    static constexpr int MAX_SCRAM_SEQ_LENGTH     = 1200;  ///< Maximum scrambling sequence length for PDCCH
    static constexpr int MAX_CRC_LENGTH           = 560;   ///< Maximum CRC length for PDCCH
    static constexpr int MAX_SUPPORTED_ND         = 14;    ///< Maximum supported number of OFDM symbols for CORESET duration

    using host32_buf = IOBuf<uint32_t, hpinned_alloc>;                        ///< Type alias for host pinned 32-bit buffer
    using tensor_pinned_R_8U = cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>;  ///< Type alias for pinned 8-bit unsigned tensor

protected:
    cuphyPdcchTxHndl_t handle;                         ///< cuPHY PDCCH transmission handle (opaque handle to cuPHY PDCCH TX object)
    cuphyPdcchDynPrms_t dyn_params;                    ///< cuPHY dynamic parameters for PDCCH TX (per-slot configuration)
    cuphyPdcchStatPrms_t static_params;                ///< cuPHY static parameters for PDCCH TX (setup-time configuration)
    std::vector<cell_id_t> cell_id_list;               ///< List of physical cell IDs being processed (built during createPhyObj)
    std::vector<cuphyCellStatPrm_t> static_params_cell;  ///< Per-cell static parameters (antennas, PRBs, numerology)
    cuphyPdcchDataIn_t DataIn;                         ///< cuPHY input data structure (contains DCI payload pointers)
    cuphyPdcchDataOut_t DataOut;                       ///< cuPHY output data structure (contains PDCCH waveform output buffer pointers)
};
#endif
