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

#ifndef PHY_ULBFW_AGGR_H
#define PHY_ULBFW_AGGR_H

#include "phychannel.hpp"
#include "physrs_aggr.hpp"

/**
 * @brief Uplink Beamforming Weights (BFW) aggregated processing channel
 *
 * Handles computation of uplink beamforming weights using channel estimation
 * data from SRS (Sounding Reference Signal) measurements. Supports digital beamforming
 * for massive MIMO with GPU-accelerated weight calculation for uplink transmission.
 */
class PhyUlBfwAggr : public PhyChannel {
public:
    /**
     * @brief Create cuPHY BFW TX object for uplink
     *
     * Initializes cuPHY library handle for uplink beamforming weight computation and
     * configures static parameters (antenna count, PRB groups, UE groups, compression).
     *
     * @return 0 on success, error code otherwise
     */
    int                             createPhyObj();
    
    /**
     * @brief Setup UL BFW processing for current slot
     *
     * Configures dynamic parameters including channel estimation buffer inputs from SRS,
     * output buffers for beamforming weights, and processing mode (graph/non-graph).
     *
     * @param[in] aggr_cell_list - List of cells being processed
     * @return 0 on success, -1 on error
     */
    int             setup( std::vector<Cell *>& aggr_cell_list);
    
    /**
     * @brief Execute UL BFW computation on GPU
     *
     * Launches cuPHY BFW TX kernel to compute uplink beamforming weights from channel
     * estimation data. Supports both graph and non-graph execution modes.
     *
     * @return 0 on success, -1 on error
     */
    int             run();
    
    /**
     * @brief Callback after UL BFW processing completion
     *
     * @return 0 on success
     */
    int             callback();
    
    /**
     * @brief Validate UL BFW output and generate debug dump if enabled
     *
     * When ULBFW_H5DUMP is defined, writes debug buffers to HDF5 file for analysis.
     *
     * @return 0 on success, -1 on error
     */
    int             validate();

    /**
     * @brief Get pointer to dynamic BFW parameters from slot command
     *
     * Retrieves BFW parameters from aggregated slot command including channel
     * estimation buffer indices and input/output descriptors.
     *
     * @return Pointer to BFW parameters, nullptr if not available
     */
    slot_command_api::bfw_params* getDynParams(); // TODO : Replace with DLBFW params

public:
    /**
     * @brief Construct uplink BFW channel processing object
     *
     * @param[in] _pdh        - Cuphydriver handle
     * @param[in] _gDev       - GPU device struct handle
     * @param[in] _s_channel  - CUDA stream for channel processing
     * @param[in] _mpsCtx     - MPS/green context for GPU resource partitioning
     */
    explicit PhyUlBfwAggr(
            phydriver_handle _pdh,
            GpuDevice*       _gDev,
            cudaStream_t     _s_channel,
            MpsCtx * _mpsCtx
        );
    
    /**
     * @brief Destructor - destroys cuPHY BFW TX handle and frees resources
     */
    ~PhyUlBfwAggr();


protected:
    std::unique_ptr<cuphyBfwTxHndl_t> handle;                                      ///< cuPHY library handle for BFW TX operations
    std::unique_ptr<hdf5hpp::hdf5_file> debugFileH;                                ///< HDF5 file handle for debug output (when ULBFW_H5DUMP enabled)
    std::vector<uint16_t> cell_id_list;                                            ///< List of PHY cell IDs being processed
    cuphy::tensor_device ulBfwChEstBuffInfo[slot_command_api::MAX_SRS_CHEST_BUFFERS]; ///< GPU tensor descriptors for channel estimation buffers (currently unused)
    cuphyBfwDataIn_t data_in;                                                      ///< Input parameters for BFW computation (channel estimation from SRS)
    cuphyBfwDataOut_t data_out;                                                    ///< Output tensor parameters for BFW results (unused, output via dyn_params)
    cuphyBfwDynPrms_t dyn_params;                                                  ///< Dynamic parameters passed to cuPHY (per-slot configuration)
    cuphyBfwStatPrms_t stat_params;                                                ///< Static parameters passed to cuPHY (setup-time configuration)
    cuphyBfwDbgPrms_t              ulBfwDbgPrms;                                   ///< Debug parameters for runtime output (HDF5 dump control)
    cuphyBfwStatDbgPrms_t          ulBfwStatDbgPrms;                               ///< Static debug parameters (setup-time HDF5 dump control)
    cuphyBfwDynDbgPrms_t            ulBfwDynDbgPrms;                               ///< Dynamic debug parameters (per-slot HDF5 API logging control)
    bool ref_check;                                                                ///< Enable reference/validation checking (set if validation mode enabled)
};

#endif

