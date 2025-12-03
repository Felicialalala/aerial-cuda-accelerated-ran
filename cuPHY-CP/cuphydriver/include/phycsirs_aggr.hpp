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

#ifndef PHY_CSIRS_CHANNEL_AGGR_H
#define PHY_CSIRS_CHANNEL_AGGR_H

#include "phychannel.hpp"

/**
 * @brief CSI-RS (Channel State Information Reference Signal) downlink channel processing
 *
 * Implements aggregated CSI-RS transmission processing using the cuPHY library.
 * CSI-RS are reference signals used by UEs to measure channel conditions and
 * report Channel State Information back to the base station. Supports multiple
 * cells per slot with configurable CSI-RS patterns and parameters.
 */
class PhyCsiRsAggr : public PhyChannel {
public:
    /**
     * @brief Create cuPHY CSI-RS TX object
     *
     * Initializes the cuPHY library handle and allocates resources for CSI-RS
     * transmission processing across configured cells.
     *
     * @return 0 on success, error code otherwise
     */
    int                             createPhyObj();
    
    /**
     * @brief Setup CSI-RS processing for current slot
     *
     * Configures dynamic parameters including output buffers, cell information,
     * and CSI-RS RRC configuration parameters for the current slot.
     *
     * @param[in] aggr_dlbuf      - Output buffers for aggregated downlink data
     * @param[in] aggr_cell_list  - List of cells to process in this slot
     *
     * @return 0 on success, error code otherwise
     */
    int             setup(const std::vector<DLOutputBuffer *>& aggr_dlbuf, const std::vector<Cell *>& aggr_cell_list);
    
    /**
     * @brief Execute CSI-RS transmission processing
     *
     * Launches cuPHY CSI-RS TX kernel to generate CSI-RS symbols in time domain.
     *
     * @return 0 on success, error code otherwise
     */
    int             run();
    
    /**
     * @brief Execute downlink completion callback
     *
     * Invokes registered callback to notify upper layers of CSI-RS processing completion.
     *
     * @return 0 on success, error code otherwise
     */
    int             callback();
    
    /**
     * @brief Validate CSI-RS output against reference
     *
     * Compares generated CSI-RS output with reference data when validation mode is enabled.
     *
     * @return 0 if validation passes, error code otherwise
     */
    int             validate();

    /**
     * @brief Get dynamic CSI-RS parameters for current slot
     *
     * Returns pointer to CSI-RS parameters from slot command API.
     *
     * @return Pointer to CSI-RS dynamic parameters, nullptr if not available
     */
    slot_command_api::csirs_params* getDynParams();
    
    /**
     * @brief Get minimum starting PRB index across all CSI-RS resources
     *
     * Iterates through all CSI-RS RRC parameters to find the smallest startRb value.
     *
     * @return Minimum starting PRB index (UINT16_MAX if no resources configured)
     */
    uint16_t        getMinPrb();
    
    /**
     * @brief Get smallest ending PRB across all CSI-RS resources
     *
     * Finds the minimum of (startRb + nRb) across all CSI-RS allocations.
     *
     * @return Smallest ending PRB value (UINT16_MAX if no resources configured)
     */
    uint16_t        getMaxPrb();
public:
    using host32_buf          = IOBuf<uint32_t, hpinned_alloc>;                ///< Host pinned 32-bit unsigned integer buffer type
    using tensor_pinned_R_32U = cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>; ///< Pinned real 32-bit unsigned integer tensor type
    using tensor_pinned_C_32F = cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>; ///< Pinned complex 32-bit float tensor type

public:
    /**
     * @brief Construct CSI-RS channel processing object
     *
     * @param[in] _pdh        - Cuphydriver context handle
     * @param[in] _gDev       - GPU device struct handle
     * @param[in] _s_channel  - CUDA stream for channel processing
     * @param[in] _mpsCtx     - MPS/green context for GPU resource partitioning
     */
    explicit PhyCsiRsAggr(
            phydriver_handle _pdh,
            GpuDevice*       _gDev,
            cudaStream_t     _s_channel,
            MpsCtx * _mpsCtx
        );
    
    /**
     * @brief Destructor
     *
     * Destroys cuPHY CSI-RS TX object and frees allocated resources.
     */
    ~PhyCsiRsAggr();

    PhyCsiRsAggr(const PhyCsiRsAggr&) = delete;
    PhyCsiRsAggr& operator=(const PhyCsiRsAggr&) = delete;
    // TODO - Move ctor

protected:
    std::unique_ptr<cuphyCsirsTxHndl_t> handle;                ///< cuPHY library handle for CSI-RS TX operations
    std::vector<uint16_t> cell_id_list;                        ///< List of PHY cell IDs being processed
    tensor_pinned_C_32F ref_output;                            ///< Reference output for validation (complex float)
    cuphyCsirsDataOut_t data_out;                              ///< Output tensor parameters for CSI-RS transmission
    cuphyCsirsDynPrms_t dyn_params;                            ///< Dynamic parameters passed to cuPHY (per-slot configuration)
    cuphyCsirsStatPrms_t stat_params;                          ///< Static parameters passed to cuPHY (setup-time configuration)
    std::vector<cuphyCellStatPrm_t> static_params_cell;        ///< Per-cell static configuration parameters
    bool ref_check;                                            ///< Enable validation against reference output
    size_t dyn_params_num;                                     ///< Number of dynamic CSI-RS RRC parameter sets for current slot
};

#endif
