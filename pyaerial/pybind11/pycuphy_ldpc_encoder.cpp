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

#include <iostream>
#include <memory>
#include <stdexcept>
#include "util.hpp"
#include "utils.cuh"
#include "cuphy_internal.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_util.hpp"

using namespace std::complex_literals;


namespace pycuphy {


LdpcEncoder::LdpcEncoder(void* outputDevicePtr, const cudaStream_t cuStream):
m_outputDevicePtr(outputDevicePtr),
m_puncture(1),
m_effN(0),
m_cuStream(cuStream) {
    if (!m_outputDevicePtr) {
        throw std::runtime_error("LdpcEncoder::encode: Memory for outputs not allocated!");
    }

    constexpr int maxUes = PDSCH_MAX_UES_PER_CELL_GROUP;
    size_t descSize = 0, allocSize = 0, workspaceSize = 0;
    cuphyStatus_t status = cuphyLDPCEncodeGetDescrInfo(&descSize, &allocSize, maxUes, &workspaceSize);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyLDPCEncodeGetDescrInfo error {}", status);
        throw std::runtime_error("LdpcEncoder::LdpcEncoder: cuphyLDPCEncodeGetDescrInfo error!");
    }

    m_dLdpcDesc = cuphy::buffer<uint8_t, cuphy::device_alloc>(descSize);
    m_hLdpcDesc = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(descSize);
    m_dWorkspace = cuphy::buffer<uint8_t, cuphy::device_alloc>(workspaceSize);
    m_hWorkspace = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(workspaceSize);
}


void LdpcEncoder::setPuncturing(uint8_t puncture) {
    m_puncture = puncture;
}


const cuphy::tensor_device& LdpcEncoder::encode(const cuphy::tensor_device& inputData,
                                                const std::vector<PdschPerTbParams>& tbParams) {

    static constexpr uint32_t ELEMENT_SIZE = sizeof(uint32_t) * CHAR_BIT;

    if(tbParams.empty()) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Got empty TB params!");
        throw std::runtime_error("Got empty TB params!");
    }

    // Parameter selection based on base graph.
    uint16_t maxParityNodes;
    int nCwNodes;
    if(tbParams[0].bg == 1) {
        maxParityNodes = CUPHY_LDPC_MAX_BG1_PARITY_NODES;
        nCwNodes = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES;
    }
    else {
        maxParityNodes = CUPHY_LDPC_MAX_BG2_PARITY_NODES;
        nCwNodes = CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
    }
    m_effN = tbParams[0].Zc * (nCwNodes + (m_puncture ? 0 : 2));

    // Output tensor in device memory.
    if(inputData.rank() != 2) {
        throw std::runtime_error("LdpcEncoder::encode: Invalid input data!");
    }
    const uint32_t K = inputData.dimensions()[0] * ELEMENT_SIZE;
    const uint32_t C = inputData.dimensions()[1];
    const uint32_t roundedN = round_up_to_next(m_effN, ELEMENT_SIZE);
    m_dOutputTensor = cuphy::tensor_device(m_outputDevicePtr, CUPHY_BIT, roundedN, C);

    // Allocate launch config struct.
    std::unique_ptr<cuphyLDPCEncodeLaunchConfig> ldpcHandle = std::make_unique<cuphyLDPCEncodeLaunchConfig>();

    // Setup the LDPC Encoder
    constexpr uint8_t descAsyncCopy = 1; // Copy descriptor to the GPU during setup
    cuphyStatus_t status = cuphySetupLDPCEncode(ldpcHandle.get(),
                                                inputData.desc().handle(),
                                                inputData.addr(),
                                                m_dOutputTensor.desc().handle(),
                                                m_dOutputTensor.addr(),
                                                tbParams[0].bg,
                                                tbParams[0].Zc,
                                                m_puncture,
                                                maxParityNodes,
                                                tbParams[0].rv,
                                                0,
                                                1,
                                                nullptr,
                                                nullptr,
                                                m_hWorkspace.addr(),
                                                m_dWorkspace.addr(),
                                                m_hLdpcDesc.addr(),
                                                m_dLdpcDesc.addr(),
                                                descAsyncCopy,
                                                m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupLDPCEncode error {}", status);
        throw std::runtime_error("LdpcEncoder::encode: Invalid argument(s) for cuphySetupLDPCEncode!");
    }

    if(CUresult r = launch_kernel(ldpcHandle->m_kernelNodeParams, m_cuStream); r != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "LDPC encoder kernel launch failed!");
        throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
    }

    ldpcHandle.reset();

    return m_dOutputTensor;
}


PyLdpcEncoder::PyLdpcEncoder(const uint64_t outputDevicePtr, const uint64_t cuStream):
m_cuStream(reinterpret_cast<cudaStream_t>(cuStream)),
m_ldpcEncoder(reinterpret_cast<void*>(outputDevicePtr), reinterpret_cast<cudaStream_t>(cuStream))
{}


void PyLdpcEncoder::setPuncturing(const uint8_t puncture) {
    m_ldpcEncoder.setPuncturing(puncture);
}


const cuda_array_t<uint32_t>& PyLdpcEncoder::encode(const cuda_array_t<uint32_t>& inputData,
                                                    const uint32_t tbSize,
                                                    const float codeRate,
                                                    const int rv) {
    static constexpr uint32_t ELEMENT_SIZE = sizeof(uint32_t) * CHAR_BIT;

    constexpr size_t numTbs = 1;
    m_tbParams.resize(numTbs);

    const int roundedK = inputData.get_shape()[0] * ELEMENT_SIZE;
    const int numCodeBlocks = inputData.get_shape()[1];

    // Input data is packed to 32-bit integers, assumed done when calling from Python.
    const cuphy::tensor_device dInputTensor = deviceFromCudaArray<uint32_t>(
        inputData,
        nullptr,
        CUPHY_R_32U,
        CUPHY_R_32U,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    // Conversion to CUPHY_BIT.
    cuphy::tensor_device dInputBits(CUPHY_BIT,
                                    roundedK,
                                    numCodeBlocks,
                                    cuphy::tensor_flags::align_tight);
    CUDA_CHECK(cudaMemcpyAsync(dInputBits.addr(),
                               dInputTensor.addr(),
                               dInputTensor.desc().get_size_in_bytes(),
                               cudaMemcpyDeviceToDevice,
                               m_cuStream));

    // Set the encoder parameters into the struct.
    setPdschPerTbParams(m_tbParams[0],
                        static_cast<uint8_t*>(dInputBits.addr()),
                        0,        // tbStartOffset
                        tbSize,
                        0,        // cumulativeTbSizePadding
                        codeRate,
                        0,        // rateMatchLen not used
                        2,        // qamMod not used
                        0,        // numCodedBits computed
                        rv,
                        1,        // numLayers not used
                        0);       // cinit not used

    const cuphy::tensor_device& dOutputTensor = m_ldpcEncoder.encode(dInputBits, m_tbParams);
    if(dOutputTensor.rank() != 2) {
        throw std::runtime_error("PyLdpcEncoder::encode: Invalid output data dimensions!");
    }

    const std::vector shape = {static_cast<size_t>(dOutputTensor.dimensions()[0] / ELEMENT_SIZE),
                               static_cast<size_t>(dOutputTensor.dimensions()[1])};
    m_encodedBits = deviceToCudaArrayPtr<uint32_t>(dOutputTensor.addr(), shape);
    return *m_encodedBits;
}


} // namespace pycuphy
