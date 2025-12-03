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

#ifndef PYCUPHY_CFO_TA_EST_HPP
#define PYCUPHY_CFO_TA_EST_HPP

#include <vector>
#include <memory>

#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "cuda_array_interface.hpp"


namespace pycuphy {

// This is the interface when called from C++.
class CfoTaEstimator {
public:

    CfoTaEstimator(cudaStream_t cuStream);
    ~CfoTaEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Output getters.
    const std::vector<cuphy::tensor_ref>& getCfoEst() const { return m_tCfoEstVec; }
    const cuphy::tensor_ref& getCfoHz() const { return m_tCfoHz; }
    const cuphy::tensor_ref& getTaEst() const { return m_tTaEst; }
    const cuphy::tensor_ref& getCfoPhaseRot() const { return m_tCfoPhaseRot; }
    const cuphy::tensor_ref& getTaPhaseRot() const { return m_tTaPhaseRot; }

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxCfoTaEstHndl_t m_cfoTaEstHndl;

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;
    size_t m_statDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_statDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_statDescrBufGpu;

    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<cuphy::tensor_ref> m_tCfoEstVec;
    cuphy::tensor_ref m_tCfoHz;
    cuphy::tensor_ref m_tTaEst;
    cuphy::tensor_ref m_tCfoPhaseRot;
    cuphy::tensor_ref m_tTaPhaseRot;
    cuphy::tensor_ref m_tCfoTaEstInterCtaSyncCnt;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyCfoTaEstimator {
public:
    PyCfoTaEstimator(uint64_t cuStream);

    const std::vector<cuda_array_t<std::complex<float>>>& estimate(const std::vector<cuda_array_t<std::complex<float>>>& chEst,
                                                                   PuschParams& puschParams);

    const cuda_array_t<float>& getCfoHz() const { return *m_cfoEstHz; }
    const cuda_array_t<float>& getTaEst() const { return *m_taEst; }
    const cuda_array_t<std::complex<float>>& getCfoPhaseRot() const { return *m_cfoPhaseRot; }
    const cuda_array_t<std::complex<float>>& getTaPhaseRot() const { return *m_taPhaseRot; }

private:

    CfoTaEstimator m_cfoTaEstimator;

    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;

    // Outputs.
    std::vector<cuda_array_t<std::complex<float>>> m_cfoEstVec;
    std::unique_ptr<cuda_array_t<float>> m_cfoEstHz;
    std::unique_ptr<cuda_array_t<float>> m_taEst;
    std::unique_ptr<cuda_array_t<std::complex<float>>> m_cfoPhaseRot;
    std::unique_ptr<cuda_array_t<std::complex<float>>> m_taPhaseRot;
};

} // pycuphy

#endif // PYCUPHY_CFO_TA_EST_HPP
