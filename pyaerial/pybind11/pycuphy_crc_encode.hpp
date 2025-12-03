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

#ifndef PYCUPHY_CRC_ENCODE_HPP
#define PYCUPHY_CRC_ENCODE_HPP

#include "cuphy.hpp"
#include "cuphy.h"
#include "cuda_array_interface.hpp"


namespace pycuphy {


class CrcEncoder final {

public:
    explicit CrcEncoder(uint32_t maxUesPerCellGroup, cudaStream_t cuStream);
    explicit CrcEncoder(cudaStream_t cuStream): CrcEncoder(PDSCH_MAX_UES_PER_CELL_GROUP, cuStream) {}

    [[nodiscard]] const cuphy::tensor_device& encode(const cuphy::tensor_device& tbInput,
                                                     const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
                                                     const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
                                                     uint32_t numTbs);

private:
    cudaStream_t m_cuStream{};

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrCrcEncodeCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrCrcEncodeGpu;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrPrepareCrcEncodeCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrPrepareCrcEncodeGpu;

    // Outputs
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_outputTbCrcs;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_outputBuffer;
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_crcWorkspace;
    cuphy::tensor_device m_outputCodeBlocks;
};


class __attribute__((visibility("default"))) PyCrcEncoder final {

public:
    explicit PyCrcEncoder(uint64_t cuStream, uint32_t maxNumTbs);

    [[nodiscard]] const std::vector<cuda_array_uint8>& encode(const cuda_array_uint8& tbInput,
                                                              const std::vector<uint32_t>& tbSizes,
                                                              const std::vector<float>& codeRates);

    [[nodiscard]] const std::vector<uint32_t>& getNumInfoBits() const { return m_numInfoBits; }

private:
    std::vector<uint32_t> m_numInfoBits;
    std::vector<cuda_array_uint8> m_outputBits;

    CrcEncoder m_crcEncoder;
    cudaStream_t m_cuStream{};

    cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc> m_tbParams;
    cuphy::buffer<PdschPerTbParams, cuphy::device_alloc> m_tbParamsDev;
};


}  // namespace pycuphy

#endif // PYCUPHY_CRC_ENCODE_HPP