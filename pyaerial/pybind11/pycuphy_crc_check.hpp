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

#ifndef PYCUPHY_CRC_CHECK_HPP
#define PYCUPHY_CRC_CHECK_HPP

#include <vector>
#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"


namespace pycuphy {

// This is the C++ API.
class CrcChecker {

public:
    CrcChecker(const cudaStream_t cuStream);
    ~CrcChecker();

    void checkCrc(void* ldpcOutput,
                  PuschParams& puschparams);

    void checkCrc(void* ldpcOutput,
                  const PerTbParams* tbPrmsCpu,
                  const PerTbParams* tbPrmsGpu,
                  const int nUes);

    const uint8_t* getOutputTbs() const { return m_outputTbs; }
    const uint32_t* getCbCrcs() const { return m_outputCbCrcs; }
    const uint32_t* getTbCrcs() const { return m_outputTbCrcs; }

    const std::vector<uint32_t>& getTbPayloadStartOffsets() const { return m_tbPayloadStartOffsets; }
    const std::vector<uint32_t>& getTbCrcStartOffsets() const { return m_tbCrcStartOffsets; }
    const std::vector<uint32_t>& getCbCrcStartOffsets() const { return m_cbCrcStartOffsets; }

    uint32_t getTotNumTbs() const { return m_totNumTbs; }
    uint32_t getTotNumCbs() const { return m_totNumCbs; }
    uint32_t getTotNumPayloadBytes() const { return m_totNumPayloadBytes; }

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void allocateDescr();
    size_t getBufferSize() const;
    void destroy();

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    // Output addresses on device.
    uint8_t* m_outputTbs;
    uint32_t* m_outputCbCrcs;
    uint32_t* m_outputTbCrcs;

    // Start offsets per UE.
    std::vector<uint32_t> m_tbPayloadStartOffsets;
    std::vector<uint32_t> m_tbCrcStartOffsets;
    std::vector<uint32_t> m_cbCrcStartOffsets;

    // Total number of TBs/CBs/payload bytes.
    uint32_t m_totNumTbs;
    uint32_t m_totNumCbs;
    uint32_t m_totNumPayloadBytes;

    cudaStream_t m_cuStream;
    cuphyPuschRxCrcDecodeHndl_t m_crcDecodeHndl;
};


// This is the Python API exposed to Python through pybind11.
class __attribute__((visibility("default"))) PyCrcChecker {

public:
    PyCrcChecker(const uint64_t cuStream);
    ~PyCrcChecker();

    const std::vector<cuda_array_t<uint8_t>>& checkCrc(const cuda_array_t<__half>& ldpcOutput,
                                                       const std::vector<uint32_t>& tbSizes,
                                                       const std::vector<float>& codeRates);

    const std::vector<cuda_array_t<uint8_t>>& getTbPayloads() const { return m_tbPayloads; }
    const std::vector<cuda_array_t<uint32_t>>& getCbCrcs() const { return m_cbCrcs; }
    const std::vector<cuda_array_t<uint32_t>>& getTbCrcs() const { return m_tbCrcs; }

private:
    CrcChecker m_crcChecker;
    cudaStream_t m_cuStream;

    // Pre-allocated memory for LDPC output / CRC input converted to cuPHY format.
    void* m_pCrcInput;

    std::vector<cuda_array_t<uint8_t>> m_tbPayloads;
    std::vector<cuda_array_t<uint32_t>> m_tbCrcs;
    std::vector<cuda_array_t<uint32_t>> m_cbCrcs;
};


} // namespace pycuphy


#endif // PYCUPHY_CRC_CHECK_HPP