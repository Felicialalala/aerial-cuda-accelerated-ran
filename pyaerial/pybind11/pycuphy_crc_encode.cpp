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

#include "cuphy.hpp"
#include "cuphy.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_crc_encode.hpp"


namespace pycuphy {

CrcEncoder::CrcEncoder(const uint32_t maxUesPerCellGroup,
                       const cudaStream_t cuStream):
m_cuStream(cuStream) {

    // Allocate descriptors
    size_t dynDescrSizeBytes, dynDescrAlignBytes;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyCrcEncodeGetDescrInfo error {}", status);
        throw cuphy::cuphy_fn_exception(status, "cuphyCrcEncodeGetDescrInfo()");
    }
    dynDescrSizeBytes = ((dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrCrcEncodeCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(dynDescrSizeBytes);
    m_dynDescrCrcEncodeGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(dynDescrSizeBytes);

    status = cuphyPrepareCrcEncodeGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyPrepareCrcEncodeGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }
    dynDescrSizeBytes = ((dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrPrepareCrcEncodeCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(dynDescrSizeBytes);
    m_dynDescrPrepareCrcEncodeGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(dynDescrSizeBytes);

    // Allocate memory for outputs
    m_outputTbCrcs = cuphy::buffer<uint32_t, cuphy::device_alloc>(maxUesPerCellGroup);

    const uint32_t maxTotalCbBytes = PDSCH_MAX_UES_PER_CELL * MAX_N_CBS_PER_TB_SUPPORTED * div_round_up<uint32_t>(CUPHY_LDPC_BG1_INFO_NODES * CUPHY_LDPC_MAX_LIFTING_SIZE, sizeof(uint8_t) * CHAR_BIT);
    m_outputBuffer = cuphy::buffer<uint8_t, cuphy::device_alloc>(maxTotalCbBytes);

    // Input comes as given in a GPU buffer, but this is the intermediate buffer for storing
    // results from CRC prepare kernel, which is input to CRC encode kernel.
    // TODO: Times two as done in pdsch_tx.cpp, why?
    const uint32_t crcWorkspaceElems = 2 * div_round_up<uint32_t>(maxTotalCbBytes, sizeof(uint32_t));
    m_crcWorkspace = cuphy::buffer<uint32_t, cuphy::device_alloc>(crcWorkspaceElems);
}


const cuphy::tensor_device& CrcEncoder::encode(const cuphy::tensor_device& tbInput,
                                               const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
                                               const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
                                               const uint32_t numTbs) {
    if(!tbParams.size()) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Got empty transport block parameters!");
        throw std::runtime_error("CrcEncoder::encode: Got empty transport block parameters!");
    }

    uint32_t maxNumCbsPerTb = 0, maxTbBytes = 0, maxTbPaddedBytes = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {
        const uint32_t perCbCrcByteSize = (tbParams[tbIdx].num_CBs == 1) ? 0 : 3;
        const auto totalCbByteSize = div_round_up<uint32_t>(tbParams[tbIdx].K, sizeof(uint8_t) * CHAR_BIT); // K in bytes (includes CRC and filler bits)
        const auto cbDataByteSize = totalCbByteSize - perCbCrcByteSize - div_round_up<uint32_t>(tbParams[tbIdx].F, CHAR_BIT);  // CRC and filler bits removed
        const uint32_t tbPaddedByteSize = cbDataByteSize * tbParams[tbIdx].num_CBs;

        maxNumCbsPerTb = std::max(maxNumCbsPerTb, tbParams[tbIdx].num_CBs);
        maxTbPaddedBytes = std::max(maxTbPaddedBytes, tbPaddedByteSize);
        maxTbBytes = std::max(maxTbBytes, tbParams[tbIdx].tbSize);
    }

    constexpr uint8_t readTbCrc = 0;  // Hard-coded to compute also TB CRCs.
    constexpr uint8_t descAsyncCopy = 1;
    constexpr uint8_t reverseBytes = 1;
    cuphyCrcEncodeLaunchConfig crcEncodeLaunchCfg{};
    cuphyStatus_t status = cuphySetupCrcEncode(&crcEncodeLaunchCfg,
                                               nullptr,
                                               m_outputTbCrcs.addr(),
                                               m_crcWorkspace.addr(),
                                               m_outputBuffer.addr(),
                                               tbParamsDev.addr(),
                                               numTbs,
                                               maxNumCbsPerTb,
                                               maxTbPaddedBytes,
                                               reverseBytes,
                                               readTbCrc,
                                               m_dynDescrCrcEncodeCpu.addr(),
                                               m_dynDescrCrcEncodeGpu.addr(),
                                               descAsyncCopy,
                                               m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupCrcEncode error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    cuphyPrepareCrcEncodeLaunchConfig prepareCrcEncodeLaunchCfg{};
    status = cuphySetupPrepareCRCEncode(
            &prepareCrcEncodeLaunchCfg,
            nullptr,  // Input data provided in a GPU buffer.
            m_crcWorkspace.addr(),
            nullptr,  // Not in testing mode (hard-coded in PdschPerTbParams for pycuphy)
            tbParamsDev.addr(),
            numTbs,
            maxNumCbsPerTb,
            maxTbBytes,
            m_dynDescrPrepareCrcEncodeCpu.addr(),
            m_dynDescrPrepareCrcEncodeGpu.addr(),
            descAsyncCopy,
            m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupPrepareCRCEncode error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    // Run the kernels.
    if(CUresult prepareStatus = launch_kernel(prepareCrcEncodeLaunchCfg.m_kernelNodeParams, m_cuStream); prepareStatus != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Error in CRC encoder prepareCrcEncode ({})", prepareStatus);
        throw std::runtime_error("Error in CRC encoder prepareCrcEncode");
    }

    if(CUresult crcStatus1 = launch_kernel(crcEncodeLaunchCfg.m_kernelNodeParams[0], m_cuStream); crcStatus1 != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Error in CRC encoder crcEncode (kernel 1, {})", crcStatus1);
        throw std::runtime_error("Error in CRC encoder crcEncode (kernel 1)");
    }

    if(CUresult crcStatus2 = launch_kernel(crcEncodeLaunchCfg.m_kernelNodeParams[1], m_cuStream); crcStatus2 != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Error in CRC encoder crcEncode (kernel 2, {})", crcStatus2);
        throw std::runtime_error("Error in CRC encoder crcEncode (kernel 2)");
    }

    size_t numBytes = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {
        const auto& tbParam = tbParams[tbIdx];
        numBytes += ((tbParam.K + 31) >> 5) * sizeof(uint32_t) * tbParam.num_CBs;
    }

    m_outputCodeBlocks = cuphy::tensor_device(m_outputBuffer.addr(), CUPHY_R_8U, numBytes);
    return m_outputCodeBlocks;
}


PyCrcEncoder::PyCrcEncoder(const uint64_t cuStream, const uint32_t maxNumTbs):
m_crcEncoder(reinterpret_cast<cudaStream_t>(cuStream)),
m_cuStream(reinterpret_cast<cudaStream_t>(cuStream)) {

    m_tbParams = cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>(maxNumTbs);
    m_tbParamsDev = cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>(maxNumTbs);
    m_numInfoBits.reserve(maxNumTbs);
    m_outputBits.reserve(maxNumTbs);
}


const std::vector<cuda_array_uint8>& PyCrcEncoder::encode(const cuda_array_uint8& tbInput,
                                                          const std::vector<uint32_t>& tbSizes,
                                                          const std::vector<float>& codeRates) {
    const uint32_t numTbs = tbSizes.size();
    if(!numTbs) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Got empty transport block parameters!");
        throw std::runtime_error("PyCrcEncoder::encode: Got empty transport block parameters!");
    }

    if (codeRates.size() != numTbs) {
        throw std::runtime_error("PyCrcEncoder::encode: All vector arguments must have the same length!");
    }

    cuphy::tensor_device dTbInput = deviceFromCudaArray<uint8_t>(
        tbInput,
        nullptr,
        CUPHY_R_8U,
        CUPHY_R_8U,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    uint32_t tbStartOffset = 0;
    uint32_t cumulativeTbSizePadding = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {

        const auto tbSizeBytes = div_round_up<uint32_t>(tbSizes[tbIdx], CHAR_BIT);
        auto paddingBytes = div_round_up<uint32_t>(tbSizeBytes, sizeof(uint32_t)) * sizeof(uint32_t) - tbSizeBytes;
        paddingBytes += ((paddingBytes <= 2) ? sizeof(uint32_t) : 0);
        cumulativeTbSizePadding += (tbSizeBytes + paddingBytes);

        setPdschPerTbParams(
            m_tbParams[tbIdx],
            static_cast<uint8_t*>(dTbInput.addr()) + tbStartOffset,
            tbStartOffset,
            tbSizes[tbIdx],
            cumulativeTbSizePadding,
            codeRates[tbIdx],
            // The rest of the parameters are not used in CRC encoding.
            0,
            2,
            0,
            0,
            1,
            0
        );

        tbStartOffset += tbSizeBytes;
    }

    // Copy TB parameters from host to device.
    CUDA_CHECK(cudaMemcpyAsync(m_tbParamsDev.addr(),
                               m_tbParams.addr(),
                               sizeof(PdschPerTbParams) * numTbs,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    const cuphy::tensor_device& outputCbs = m_crcEncoder.encode(dTbInput, m_tbParams, m_tbParamsDev, numTbs);

    // Memorize the number of information bits.
    // Wrap the output code blocks into CUDA arrays.
    m_numInfoBits.clear();
    m_outputBits.clear();
    size_t tbOutputOffset = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {
        const auto& tbParam = m_tbParams[tbIdx];

        m_numInfoBits.push_back(tbParam.K);

        const size_t numCbBytes = ((tbParam.K + 31) >> 5) * sizeof(uint32_t);
        const size_t numBytes = numCbBytes * tbParam.num_CBs;
        const std::vector shape = {numCbBytes, static_cast<size_t>(tbParam.num_CBs)};
        uint8_t* tbOutputAddr = static_cast<uint8_t*>(outputCbs.addr()) + tbOutputOffset;
        m_outputBits.push_back(deviceToCudaArray<uint8_t>(tbOutputAddr, shape));
        tbOutputOffset += numBytes;
    }
    return m_outputBits;
}


} // namespace pycuphy