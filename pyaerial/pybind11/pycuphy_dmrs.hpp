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

#ifndef PYCUPHY_DMRS_HPP
#define PYCUPHY_DMRS_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuphy.hpp"
#include "cuda_array_interface.hpp"


namespace pycuphy {

/**
 * @brief Class for handling PDSCH DMRS (Demodulation Reference Signal) transmission on CUDA devices.
 *
 * This class manages the transmission of DMRS signals for PDSCH (Physical Downlink Shared Channel)
 * using CUDA acceleration. It handles the necessary memory allocations and CUDA stream management
 * for efficient DMRS signal processing.
 */
class PdschDmrsTx final {

public:
    /**
     * @brief Construct a new PdschDmrsTx object
     * @param cuStream CUDA stream for asynchronous operations
     */
    explicit PdschDmrsTx(cudaStream_t cuStream);

    /**
     * @brief Execute DMRS transmission processing
     * @param dmrsParams CPU buffer containing DMRS parameters for processing
     * @param dmrsParamsDev Device buffer containing DMRS parameters for processing
     * @param numTbs Number of transport blocks to process
     */
    void run(const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
             const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
             uint32_t numTbs);

private:
    cudaStream_t m_cuStream;  ///< CUDA stream for asynchronous operations

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;  ///< Pinned CPU buffer for dynamic descriptors
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;  ///< Device buffer for dynamic descriptors
};


/**
 * @brief Python interface class for PDSCH DMRS transmission.
 *
 * This class provides an interface towards Python for DMRS transmission operations,
 * handling the conversion between Python objects and CUDA device memory. It manages
 * memory allocations, data type conversions, and parameter processing for DMRS transmission.
 */
class PyPdschDmrsTx final {

public:
    /**
     * @brief Construct a new PyPdschDmrsTx object
     * @param cuStream CUDA stream handle for asynchronous operations
     * @param maxNumCells Maximum number of cells per slot (default: MAX_CELLS_PER_SLOT)
     * @param maxNumTbs Maximum number of transport blocks per cell group (default: PDSCH_MAX_UES_PER_CELL_GROUP)
     */
    explicit PyPdschDmrsTx(uint64_t cuStream,
                           uint32_t maxNumCells = MAX_CELLS_PER_SLOT,
                           uint32_t maxNumTbs = PDSCH_MAX_UES_PER_CELL_GROUP);

    /**
     * @brief Process DMRS transmission for multiple input buffers
     * @param txBuffers Vector of input/output transmission buffers containing complex float data for each cell
     * @param slot Slot number
     * @param dmrsParams Vector of Python objects containing DMRS parameters for each transport block
     *
     * This method processes DMRS transmission for multiple cells in parallel. Each element in txBuffers
     * corresponds to a cell's transmission buffer, and each element in dmrsParams contains the DMRS
     * configuration for a transport block.
     */
    void run(std::vector<cuda_array_complex_float>& txBuffers,
             uint32_t slot,
             const std::vector<pybind11::object>& dmrsParams);

private:
    PdschDmrsTx m_dmrsTx;  ///< Underlying CUDA DMRS transmission handler
    cudaStream_t m_cuStream{};  ///< CUDA stream for asynchronous operations

    cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc> m_dmrsParams;  ///< Pinned CPU buffer for DMRS parameters
    cuphy::buffer<PdschDmrsParams, cuphy::device_alloc> m_dmrsParamsDev;  ///< Device buffer for DMRS parameters

    std::vector<cuphy::unique_device_ptr<__half2>> m_txBufHalf;  ///< Device buffers for half-precision complex data
    std::vector<cuphy::tensor_device> m_txTensors;  ///< CUDA tensor descriptors for half-precision tensors

    /**
     * @brief Extract DMRS parameters from Python objects
     * @param dmrsParams Vector of Python objects containing DMRS configuration
     * @param slot Slot number
     */
    void readDmrsParams(const std::vector<pybind11::object>& dmrsParams, uint32_t slot) const;
};

} // namespace pycuphy

#endif // PYCUPHY_DMRS_HPP