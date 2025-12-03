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

#ifndef PHY_PUCCH_AGGR_CHANNEL_H
#define PHY_PUCCH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"

/**
 * @class PhyPucchAggr
 * @brief Handles PUCCH (Physical Uplink Control Channel) reception and UCI decoding processing.
 *
 * This class implements the uplink PUCCH reception pipeline using the cuPHY library.
 * PUCCH carries Uplink Control Information (UCI) including HARQ-ACK feedback, CSI reports,
 * and scheduling requests. Supports aggregated multi-cell processing with CUDA graph and
 * stream-based execution modes.
 */
class PhyPucchAggr : public PhyChannel {
public:
    /**
     * @brief Constructs a PhyPucchAggr object.
     *
     * Initializes the PUCCH channel with the given cuphydriver handle, GPU device, CUDA streams,
     * and MPS context. Sets up memory footprint tracking, allocates format-specific output buffers
     * (UCI outputs, payload buffers, quality metrics), and initializes UCI flags.
     *
     * @param _pdh cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channels Array of CUDA streams (uses first stream) for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     */
    PhyPucchAggr(phydriver_handle _pdh, GpuDevice* _gDev, cudaStream_t* _s_channels, MpsCtx * _mpsCtx);
    
    /**
     * @brief Destructor for PhyPucchAggr.
     *
     * Frees allocated buffers and destroys the cuPHY PUCCH RX handle.
     */
    ~PhyPucchAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Configures the PUCCH reception for the current slot.
     *
     * Sets up dynamic parameters including PUCCH UCI parameters, input buffer
     * pointers from UL order kernel (st1 buffers), and output buffer configuration. Calls
     * cuphySetupPucchRx().
     *
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @param aggr_ulbuf_st1 List of UL input buffers (st1/o1) containing ordered PUCCH/PUSCH samples
     * @param stream CUDA stream for setup operations
     * @return 0 on success, -1 on failure (setup error)
     */
    int          setup(const std::vector<Cell *>& aggr_cell_list, const std::vector<ULInputBuffer *>& aggr_ulbuf_st1, cudaStream_t stream);
    
    /**
     * @brief Executes the PUCCH UCI decoding processing on GPU.
     *
     * Invokes cuphyRunPucchRx() to perform UCI demodulation, decoding, and quality metric
     * computation (RSSI, SINR, timing advance). Copies results to host buffers. Uses either
     * CUDA graphs or streams based on configuration.
     *
     * @return 0 on success, -1 on failure (run error)
     */
    int          run();
    
    /**
     * @brief Validates PUCCH output against reference data.
     *
     * Currently a no-op (returns 0 immediately). Provided for API consistency.
     *
     * @return Always returns 0
     */
    int          validate();
    
    /**
     * @brief Post-processing callback after PUCCH decoding completes.
     *
     * Invokes registered UL callback (uci_cb_fn2) to report decoded UCI (HARQ-ACK, CSI, SR)
     * and quality metrics to upper layers (MAC). Updates per-cell PUCCH metrics when enabled.
     *
     * @return 0 on success, error code otherwise
     */
    int          callback();
    
    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    /**
     * @brief Retrieves the PUCCH dynamic parameters for the current slot.
     *
     * Extracts the PUCCH parameters (UCI configs, formats) from the aggregated slot
     * command structure for use in setup().
     *
     * @return Pointer to the PUCCH parameters for the current slot
     */
    slot_command_api::pucch_params* getDynParams();
    
    /**
     * @brief Validates PUCCH output (unimplemented).
     *
     * Function declared but not implemented for PUCCH. Returns false if called.
     *
     * @return false (unimplemented)
     */
    bool                            validateOutput();
    
    /**
     * @brief Gets minimum PRB index (unimplemented).
     *
     * Function declared but not implemented for PUCCH. Returns 0 if called.
     *
     * @return 0 (unimplemented)
     */
    uint16_t                        getMinPrb();
    
    /**
     * @brief Gets maximum PRB index (unimplemented).
     *
     * Function declared but not implemented for PUCCH. Returns 0 if called.
     *
     * @return 0 (unimplemented)
     */
    uint16_t                        getMaxPrb();
    
    /**
     * @brief Creates the cuPHY PUCCH RX object.
     *
     * Initializes the cuPHY PUCCH reception handle, builds the cell list with static
     * parameters (antennas, PRBs, numerology). Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int                             createPhyObj();
    
    /**
     * @brief Prints PUCCH parameters for debugging.
     *
     * Debug utility function that prints PUCCH configuration parameters.
     *
     * @param pucch_params PUCCH parameters to print
     */
    void print_pucch_params(const slot_command_api::pucch_params * pucch_params);
    
    /**
     * @brief Prints static and dynamic PUCCH parameters for debugging.
     *
     * Debug utility function that prints cuPHY PUCCH static and dynamic parameters.
     *
     * @param stat Pointer to static parameters
     * @param l2 Pointer to dynamic cell group parameters
     */
    void printParameters(const cuphyPucchStatPrms_t* stat, const cuphyPucchCellGrpDynPrm_t* l2);
    
    /**
     * @brief Updates physical cell ID in static parameters.
     *
     * Used during cell reconfiguration to update the PHY cell ID without recreating
     * the entire cuPHY object.
     *
     * @param phyCellId_old Old physical cell ID to replace
     * @param phyCellId_new New physical cell ID
     */
    void updatePhyCellId(uint16_t,uint16_t);
    
    /**
     * @brief Clears UCI detection status flags to default values.
     *
     * Resets HARQ, CSI Part1, CSI Part2 detection status flags to 2 (not detected),
     * and CRC flags to 1 (CRC fail). Called before each slot processing.
     *
     * @return 0 on success
     */
    int  clearUciFlags();

protected:
    // FIX ME - PUCCH
    // PucchParams                 pucch_params_static;
    
    ///< cuPHY PUCCH reception handle (created via cuphyCreatePucchRx())
    cuphyPucchRxHndl_t handle;
    
    ///< cuPHY PUCCH batch parameter handle (future batching support)
    cuphyPucchBatchPrmHndl_t batch_handle;
    
    ///< Processing mode bitmask: stream based processing(0x0) or graph based processing(0x1)
    uint64_t                    procModeBmsk;

    ///< Vector of per-cell static PUCCH parameters (antennas, PRBs, numerology)
    std::vector<cuphyPucchCellStatPrm_t> pucchCellStatPrmsVec;
    
    ///< cuPHY input data structure (tensor descriptors and pointers to received samples)
    cuphyPucchDataIn_t          DataIn;
    
    ///< cuPHY output data structure (pointers to UCI payloads, quality metrics)
    cuphyPucchDataOut_t         DataOut;
    
    ///< Dynamic parameters for cuphyRunPucchRx() (stream, input/output pointers, per-slot configs)
    cuphyPucchDynPrms_t         dyn_params;
    
    ///< Static parameters for cuphyCreatePucchRx() (cell list, antenna configuration)
    cuphyPucchStatPrms_t        static_params;
    
    ///< Vector of per-cell static parameters (physical cell ID, antennas)
    std::vector<cuphyCellStatPrm_t> static_params_cell;
    
    ///< Debug parameters (HDF5 logging control, currently disabled)
    cuphyPucchDbgPrms_t         dbg_params;
    
    ///< HDF5 file handle for debug output (currently unused, controlled by #if 0 block)
    std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
    
    ///< Status output from cuphyRunPucchRx() (success/failure, cell/UCI index on error)
    cuphyPucchStatusOut_t       statusOut;

    ///< Host pinned buffer for PUCCH Format 0 UCI outputs (HARQ-ACK, SR)
    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc> pf0OutBuffer;
    
    ///< Host pinned buffer for PUCCH Format 1 UCI outputs (HARQ-ACK, SR)
    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc> pf1OutBuffer;
    
    ///< Tensor descriptors for PUCCH received sample data per cell
    cuphy::tensor_desc pucch_data_rx_desc[UL_MAX_CELLS_PER_SLOT];
    
    ///< Tensor parameter for received data (used for setup)
    cuphyTensorPrm_t                   tDataRx;

    ///< Number of PUCCH users
    int pucch_users;
    
    ///< Host pinned buffer for PUCCH Format 2 output offsets (UCI payload start positions)
    cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc> bPucchF2OutOffsets;
    
    ///< Host pinned buffer for PUCCH Format 3 output offsets (UCI payload start positions)
    cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc> bPucchF3OutOffsets;
    
    ///< Host pinned buffer for decoded UCI payloads (CSI, HARQ-ACK bits)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bUciPayloads;
    
    ///< Host pinned buffer for CRC status flags (0=pass, 1=fail)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bCrcFlags;
    
    ///< Host pinned buffer for Received Signal Strength Indicator (RSSI) per UCI
    cuphy::buffer<float, cuphy::pinned_alloc> bRssi;
    
    ///< Host pinned buffer for Signal-to-Interference-plus-Noise Ratio (SINR) per UCI
    cuphy::buffer<float, cuphy::pinned_alloc> bSnr;
    
    ///< Host pinned buffer for interference power per UCI
    cuphy::buffer<float, cuphy::pinned_alloc> bInterf;
    
    ///< Host pinned buffer for Timing Advance (TA) estimates per UCI
    cuphy::buffer<float, cuphy::pinned_alloc> bTaEst;
    
    ///< Host pinned buffer for Reference Signal Received Power (RSRP) per UCI
    cuphy::buffer<float, cuphy::pinned_alloc> bRsrp;
    
    ///< Host pinned buffer for number of CSI Part2 bits per UCI
    cuphy::buffer<uint16_t, cuphy::pinned_alloc> bNumCsi2Bits;
    
    ///< Host pinned buffer for HARQ-ACK detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bHarqDetectionStatus;
    
    ///< Host pinned buffer for CSI Part1 detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bCsiP1DetectionStatus;
    
    ///< Host pinned buffer for CSI Part2 detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> bCsiP2DetectionStatus;

    // Standalone Start
    ///< Vector of PUCCH Format 0 UCI parameters (per-UCI configuration)
    std::vector<cuphyPucchUciPrm_t>    f0Params;
    
    ///< Vector of PUCCH Format 1 UCI parameters (per-UCI configuration)
    std::vector<cuphyPucchUciPrm_t>    f1Params;
    
    ///< Per-cell dynamic parameters (physical cell ID)
    cuphyPucchCellDynPrm_t             pucch_params_cell_dyn;
    
    ///< Cell group dynamic parameters (UCI configs for all formats)
    cuphyPucchCellGrpDynPrm_t          pucch_params_cell_grp;
    // Standalone End

    ///< UCI output parameters structure for callback reporting
    slot_command_api::uci_output_params    pucch_out;
    
    ///< List of cell IDs being processed in the current slot
    std::vector<cell_id_t> cell_id_list;
    
    ///< Host buffer for CRC error count
    std::unique_ptr<host_buf>   pucch_crc_errors_h;

    ///< Start timestamp for PUCCH reception processing (nanoseconds)
    t_ns                        start_t_rx;
    
    ///< End timestamp for PUCCH reception processing (nanoseconds)
    t_ns                        end_t_rx;

    ///< Maximum number of PUCCH Format 0 UCIs supported per slot (across all cells)
    static constexpr uint MAX_PUCCH_F0_UCIS = 256U;
    
    ///< Maximum number of PUCCH Format 1 UCIs supported per slot (across all cells)
    static constexpr uint MAX_PUCCH_F1_UCIS = 256U;
};
#endif
