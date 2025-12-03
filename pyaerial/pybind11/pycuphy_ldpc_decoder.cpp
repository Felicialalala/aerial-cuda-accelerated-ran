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

#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"

namespace pycuphy {

LdpcDecoder::LdpcDecoder(const cudaStream_t cuStream):
m_ctx(),
m_decoder(m_ctx),
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()),
m_normalizationFactor(0.8125f) {}


size_t LdpcDecoder::getBufferSize() const {

    static constexpr uint32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    static constexpr size_t BYTES_PER_WORD = 4;

    static constexpr size_t ldpcOutputBufferSize = BYTES_PER_WORD * OUT_STRIDE_WORDS * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
    static constexpr size_t softOutputBufferSize = sizeof(__half) * MAX_DECODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
    static constexpr size_t nBytes = ldpcOutputBufferSize + softOutputBufferSize;
    return nBytes;
}


const std::vector<void*>& LdpcDecoder::getSoftOutputs() const {
    return m_ldpcSoftOutput;
}


void* LdpcDecoder::decode(void** deRmOutput, PuschParams& puschParams) {

    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;
    std::vector<void*> deRmLlr(deRmOutput, deRmOutput + nUes);

    const cuphyLDPCParams& ldpcParams = puschParams.getLdpcPrms();
    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    std::vector<PerTbParams> tbPrmsCpu(pTbPrmsCpu, pTbPrmsCpu + nUes);

    return decode(deRmLlr, tbPrmsCpu, ldpcParams);
}


void* LdpcDecoder::decode(const std::vector<void*>& deRmLlr,
                          const std::vector<PerTbParams>& tbPrmsCpu,
                          const cuphyLDPCParams& ldpcParams) {

    m_linearAlloc.reset();
    int nUes = tbPrmsCpu.size();
    m_ldpcSoftOutput.resize(nUes);

    int dims[2];
    int strides[2];
    const int32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const int32_t BYTES_PER_WORD = 4;

    // Allocate output memory.
    size_t lpdcDecodeOutSize = 0;
    for(uint32_t tbIdx = 0; tbIdx < nUes; tbIdx++) {
        lpdcDecodeOutSize += BYTES_PER_WORD * OUT_STRIDE_WORDS * tbPrmsCpu[tbIdx].num_CBs;
    }
    m_ldpcOutput = m_linearAlloc.alloc(lpdcDecodeOutSize);

    size_t outputBytes = 0;
    for(uint32_t tbIdx = 0; tbIdx < nUes; tbIdx++) {
        constexpr float clamp_value = 32.f; // TODO: Make this an API parameter.
        cuphy::LDPC_decode_config decoderConfig(ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F,
                                                ldpcParams.parityNodesArray[tbIdx],
                                                tbPrmsCpu[tbIdx].Zc,
                                                ldpcParams.fixedMaxNumItrs,
                                                clamp_value,
                                                ldpcParams.KbArray[tbIdx],
                                                m_normalizationFactor,
                                                ldpcParams.flags,
                                                tbPrmsCpu[tbIdx].bg,
                                                ldpcParams.algoIndex,
                                                nullptr);
        if(m_normalizationFactor <= 0.0f) {
            m_decoder.set_normalization(decoderConfig);
        }

        size_t numElems = round_up_to_next(ldpcParams.KbArray[tbIdx] * tbPrmsCpu[tbIdx].Zc, (unsigned int)2);
        size_t ldpcSoftOutputSize = sizeof(__half) * numElems * tbPrmsCpu[tbIdx].num_CBs;
        m_ldpcSoftOutput[tbIdx] = m_linearAlloc.alloc(ldpcSoftOutputSize);

        dims[0]    = tbPrmsCpu[tbIdx].Ncb_padded;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = tbPrmsCpu[tbIdx].Ncb_padded;

        cuphy::tensor_layout tlInput(2, dims, strides);
        cuphy::tensor_info   tiInput(ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F, tlInput);
        cuphy::tensor_desc   tdInput(tiInput, cuphy::tensor_flags::align_tight);

        dims[0]    = numElems;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = numElems;

        cuphy::tensor_layout tlSoftOutput(2, dims, strides);
        cuphy::tensor_info   tiSoftOutput(CUPHY_R_16F, tlSoftOutput);
        cuphy::tensor_desc   tdSoftOutput(tiSoftOutput, cuphy::tensor_flags::align_tight);

        dims[0]    = MAX_DECODED_CODE_BLOCK_BIT_SIZE;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = MAX_DECODED_CODE_BLOCK_BIT_SIZE;

        cuphy::tensor_layout tlOutput(2, dims, strides);
        cuphy::tensor_info   tiOutput(CUPHY_BIT, tlOutput);
        cuphy::tensor_desc   tdOutput(tiOutput, cuphy::tensor_flags::align_tight);

        cuphy::LDPC_decode_tensor_params decoderTensor(
            decoderConfig,                                          // LDPC configuration
            tdOutput.handle(),                                      // output descriptor
            static_cast<uint8_t*>(m_ldpcOutput) + outputBytes,      // output address
            tdInput.handle(),                                       // LLR descriptor
            deRmLlr[tbIdx],                                         // LLR address
            tdSoftOutput.handle(),                                  // Soft output descriptor
            m_ldpcSoftOutput[tbIdx]);                               // Soft output address

        // Run the decoder for this TB.
        m_decoder.decode(decoderTensor, m_cuStream);

        outputBytes += tdOutput.get_size_in_bytes();
    }

    return m_ldpcOutput;
}


PyLdpcDecoder::PyLdpcDecoder(const uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_decoder((cudaStream_t)cuStream),
m_linearAlloc(getBufferSize()) {
    m_ldpcParams.fixedMaxNumItrs = 10;
    m_ldpcParams.earlyTermination = false;
    m_ldpcParams.algoIndex = 0;
    m_ldpcParams.flags = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT | CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    m_ldpcParams.useHalf = true;
}

size_t PyLdpcDecoder::getBufferSize() const {
    static constexpr size_t ldpcOutputBufferSize = sizeof(__half) * MAX_DECODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
    static constexpr size_t softOutputBufferSize = sizeof(float) * MAX_DECODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
    static constexpr size_t nBytes = ldpcOutputBufferSize + softOutputBufferSize;
    return nBytes;
}

void PyLdpcDecoder::setNumIterations(const uint32_t numIterations) {
    m_ldpcParams.fixedMaxNumItrs = numIterations;
}


void PyLdpcDecoder::setThroughputMode(const uint8_t throughputMode) {
    uint8_t mode = throughputMode ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
    m_ldpcParams.flags = mode | CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
}


const std::vector<cuda_array_t<__half>>& PyLdpcDecoder::decode(const std::vector<cuda_array_t<__half>>& inputLlrs,
                                                               const std::vector<uint32_t>& tbSizes,
                                                               const std::vector<float>& codeRates,
                                                               const std::vector<uint32_t>& rvs,
                                                               const std::vector<uint32_t>& rateMatchLengths) {
    m_linearAlloc.reset();
    m_ldpcOutput.clear();
    m_softOutput.clear();
    int nUes = inputLlrs.size();
    m_tbParams.resize(nUes);
    std::vector<void*> dInputLlrs(nUes);

    m_inputLlrTensors.resize(nUes);
    m_ldpcParams.parityNodesArray.clear();
    m_ldpcParams.KbArray.clear();

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        m_inputLlrTensors[ueIdx] = deviceFromCudaArray<__half>(
            inputLlrs[ueIdx],
            nullptr,  // Use the same device buffer as no conversion needed.
            CUPHY_R_16F,
            CUPHY_R_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        dInputLlrs[ueIdx] = m_inputLlrTensors[ueIdx].addr();

        setPerTbParams(m_tbParams[ueIdx],
                       m_ldpcParams,
                       tbSizes[ueIdx],
                       codeRates[ueIdx],
                       2,              // qamModOrder not used here.
                       1,              // ndi not used here.
                       rvs[ueIdx],
                       rateMatchLengths[ueIdx],
                       0,  // cinit not used.
                       0,  // userGroupIndex not used.
                       1,  // Number of layers not used
                       1,
                       {0} // Layer mapping not used
                       );
    }

    void* ldpcOutput = m_decoder.decode(dInputLlrs, m_tbParams, m_ldpcParams);

    m_ldpcOutput.reserve(nUes);

    size_t outputBytes = 0;
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // LDPC output tensor layout.
        std::vector<size_t> shape = {MAX_DECODED_CODE_BLOCK_BIT_SIZE, m_tbParams[ueIdx].num_CBs};
        std::vector<size_t> strides = {sizeof(__half), sizeof(__half) * MAX_DECODED_CODE_BLOCK_BIT_SIZE};
        uint8_t* pDevAddr = static_cast<uint8_t*>(ldpcOutput) + outputBytes;
        m_ldpcOutput.push_back(deviceToCudaArray<__half>((void*)pDevAddr,
                                                         m_linearAlloc.alloc(MAX_DECODED_CODE_BLOCK_BIT_SIZE * m_tbParams[ueIdx].num_CBs * sizeof(__half)),
                                                         shape,
                                                         strides,
                                                         CUPHY_BIT,
                                                         CUPHY_R_16F,
                                                         cuphy::tensor_flags::align_tight,
                                                         m_cuStream));
        outputBytes += (MAX_DECODED_CODE_BLOCK_BIT_SIZE * m_tbParams[ueIdx].num_CBs) / (sizeof(uint8_t) * 8);
    }

    return m_ldpcOutput;
}


const std::vector<cuda_array_t<float>>& PyLdpcDecoder::getSoftOutputs() {

    m_softOutput.clear();
    const std::vector<void*> ldpcSoftOutput = m_decoder.getSoftOutputs();

    int nUes = ldpcSoftOutput.size();
    m_softOutput.reserve(nUes);

    size_t dims[2];
    size_t strides[2];
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // LDPC soft output tensor layout.
        size_t numElems = round_up_to_next(m_ldpcParams.KbArray[ueIdx] * m_tbParams[ueIdx].Zc, (unsigned int)2);
        std::vector<size_t> shape = {numElems, m_tbParams[ueIdx].num_CBs};

        m_softOutput.push_back(deviceToCudaArray<float>(ldpcSoftOutput[ueIdx],
                                                        m_linearAlloc.alloc(numElems * m_tbParams[ueIdx].num_CBs * sizeof(float)),
                                                        shape,
                                                        m_ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F,
                                                        CUPHY_R_32F,
                                                        cuphy::tensor_flags::align_tight,
                                                        m_cuStream));
    }

    return m_softOutput;
}


} // namespace pycuphy
