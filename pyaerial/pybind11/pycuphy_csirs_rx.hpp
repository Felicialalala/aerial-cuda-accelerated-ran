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

#ifndef PYCUPHY_CSIRS_RX_HPP
#define PYCUPHY_CSIRS_RX_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuda_array_interface.hpp"


namespace pycuphy {

/**
 * @brief Structure holding channel estimation output data for CSI-RS reception
 *
 * This structure contains all the necessary buffers and parameters for storing
 * channel estimation results from CSI-RS reception for a single UE.
 */
struct CsiRsRxChEst {
    std::vector<uint16_t>              startPrbChEst;     ///< Starting PRB indices for channel estimation
    std::vector<uint16_t>              sizePrbChEst;      ///< Number of PRBs for channel estimation
    std::vector<cuphy::tensor_device>  chEstTensor;       ///< CUDA tensors for channel estimation
    std::vector<cuphyTensorPrm_t>      chEstTensorPrm;    ///< Tensor parameters for channel estimation
};


/**
 * @brief Core CSI-RS reception class for CUDA processing
 *
 * This class handles the low-level CSI-RS reception operations on CUDA devices.
 * It manages the CUDA resources and provides the interface for running CSI-RS reception.
 */
class CsiRsRx final {

public:
    /**
     * @brief Construct a new CSI-RS receiver
     *
     * @param pStatPrms Pointer to static parameters for CSI-RS reception
     * @throw cuphy::cuphy_fn_exception if creation fails
     */
    explicit CsiRsRx(cuphyCsirsStatPrms_t const* pStatPrms);

    /**
     * @brief Destroy the CSI-RS receiver and clean up CUDA resources
     */
    ~CsiRsRx();

    CsiRsRx(const CsiRsRx&) = delete;
    CsiRsRx& operator=(const CsiRsRx&) = delete;

    /**
     * @brief Run CSI-RS reception processing
     *
     * @param pDynPrms Pointer to dynamic parameters for CSI-RS reception
     * @return cuphyStatus_t Status of the operation
     */
    [[nodiscard]] cuphyStatus_t run(cuphyCsirsRxDynPrms_t* pDynPrms);

private:
    cuphyCsirsRxHndl_t m_csirsRxHndl{};  ///< Handle to the CSI-RS receiver instance
};


/**
 * @brief Python interface for CSI-RS reception
 *
 * This class provides a Python-friendly interface for CSI-RS reception,
 * handling the conversion between Python and C++ data types and managing
 * the underlying CUDA resources.
 */
class __attribute__((visibility("default"))) PyCsiRsRx final {

public:
    /**
     * @brief Construct a new Python CSI-RS receiver
     *
     * @param numPrbDlBwp Vector of PRB counts for each cell's DL bandwidth part
     */
    explicit PyCsiRsRx(const std::vector<uint16_t>& numPrbDlBwp);

    /**
     * @brief Run CSI-RS reception processing
     *
     * @param pyCsiRsConfigs List of CSI-RS configurations for each cell
     * @param rxData Vector of received data arrays, one per UE
     * @param ueCellAssociation Vector mapping UEs to their associated cells
     * @param cudaStream CUDA stream handle for asynchronous operations
     * @return const std::vector<std::vector<cuda_array_complex_float>>& Channel estimates for each UE
     * @throw std::runtime_error if processing fails
     */
    const std::vector<std::vector<cuda_array_complex_float>>& run(
        const pybind11::list& pyCsiRsConfigs,
        const std::vector<cuda_array_complex_float>& rxData,
        const std::vector<int>& ueCellAssociation,
        uint64_t cudaStream);

private:
    /**
     * @brief Populate dynamic parameters for CSI-RS reception
     *
     * @param pyCsiRsConfigs List of CSI-RS configurations for each cell
     * @param rxData Vector of received data arrays for each UE
     * @param ueCellAssociation Vector mapping UEs to their associated cells
     * @param cudaStream CUDA stream handle for asynchronous operations
     */
    void populateDynPrms(
        const pybind11::list& pyCsiRsConfigs,
        const std::vector<cuda_array_complex_float>& rxData,
        const std::vector<int>& ueCellAssociation,
        uint64_t cudaStream);

    /**
     * @brief Populate cell-level dynamic parameters
     *
     * @param pyCsiRsConfigs List of CSI-RS configurations for each cell
     * @return uint16_t Total number of RRC parameters across all cells
     */
    uint16_t populateCellDynPrms(const pybind11::list& pyCsiRsConfigs);

    /**
     * @brief Populate UE-level dynamic parameters
     *
     * @param rxData Vector of received data arrays for each UE
     * @param ueCellAssociation Vector mapping UEs to their associated cells
     */
    void populateUeDynPrms(
        const std::vector<cuda_array_complex_float>& rxData,
        const std::vector<int>& ueCellAssociation);

    /**
     * @brief Populate RRC-level dynamic parameters
     *
     * @param pyCsiRsConfigs List of CSI-RS configurations for each cell
     * @param nTotalRrcPrms Total number of RRC parameters across all cells
     */
    void populateRrcDynPrms(const pybind11::list& pyCsiRsConfigs, uint16_t nTotalRrcPrms);

    /**
     * @brief Populate received data tensors
     *
     * @param rxData Vector of received data arrays for each UE
     */
    void populateRxData(const std::vector<cuda_array_complex_float>& rxData);

    /**
     * @brief Populate buffer info for channel estimation
     */
    void populateChEstBuffInfo();

    // Static parameters
    cuphyCsirsStatPrms_t    m_statPrms{};      ///< Static parameters for CSI-RS reception
    cuphyCsirsRxDynPrms_t   m_dynPrms{};       ///< Dynamic parameters for CSI-RS reception

    // Tracking and cell parameters
    cuphyTracker_t                          m_tracker{};           ///< Memory tracking information
    std::vector<cuphyCellStatPrm_t>         m_cellStatPrms;        ///< Static parameters for each cell

    // Dynamic parameters for different levels
    std::vector<cuphyCsirsRrcDynPrm_t>      m_csiRsRrcDynPrms;     ///< RRC-level dynamic parameters
    std::vector<cuphyCsirsCellDynPrm_t>     m_csiRsCellDynPrms;    ///< Cell-level dynamic parameters
    std::vector<cuphyCsirsUeDynPrm_t>       m_csiRsUeDynPrms;      ///< UE-level dynamic parameters
    cuphyCsirsRxDataOut_t                   m_dataOut{};           ///< Output data structure
    cuphyCsirsRxDataIn_t                    m_dataIn{};            ///< Input data structure

    // Channel estimation outputs
    std::vector<CsiRsRxChEst>                m_chEstOutput;         ///< Channel estimation outputs
    std::vector<cuphyCsirsRxChEstBuffInfo_t> m_chEstBuffInfo;       ///< Buffer info for channel estimation

    // Input data
    std::vector<cuphy::tensor_device>       m_dataRx;              ///< Received data tensors
    std::vector<cuphyTensorPrm_t>           m_dataRxTensorPrm;     ///< Received data tensor parameters

    std::unique_ptr<CsiRsRx> m_csiRsRx;     ///< Core CSI-RS receiver instance

    // Python outputs
    std::vector<std::vector<cuda_array_complex_float>> m_chEst;    ///< Channel estimates for each UE
};


}  // namespace pycuphy


#endif // PYCUPHY_CSIRS_RX_HPP