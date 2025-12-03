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

#ifndef PYCUPHY_RSRP_HPP
#define PYCUPHY_RSRP_HPP

#include <vector>
#include <memory>


#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_params.hpp"
#include "cuda_array_interface.hpp"


namespace pycuphy {

// This is the interface when called from C++.
class RsrpEstimator {
public:

    RsrpEstimator(cudaStream_t cuStream);
    ~RsrpEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Access the results once the estimator has been run.
    const cuphy::tensor_ref& getRsrp() const { return m_tRsrp; }
    const cuphy::tensor_ref& getNoiseIntVarPostEq() const { return m_tNoiseIntfVarPostEq; }
    const cuphy::tensor_ref& getSinrPreEq() const { return m_tSinrPreEq; }
    const cuphy::tensor_ref& getSinrPostEq() const { return m_tSinrPostEq; }

    void debugDump(H5DebugDump& debugDump, cudaStream_t cuStream = 0);

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxRssiHndl_t m_puschRxRssiHndl;

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    cudaStream_t m_cuStream;

    // Outputs.
    cuphy::tensor_ref m_tRsrp;
    cuphy::tensor_ref m_tNoiseIntfVarPostEq;
    cuphy::tensor_ref m_tSinrPreEq;
    cuphy::tensor_ref m_tSinrPostEq;
    cuphy::tensor_ref m_tInterCtaSyncCnt;

};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyRsrpEstimator {
public:
    PyRsrpEstimator(uint64_t cuStream);

    const cuda_array_t<float>& estimate(const std::vector<cuda_array_t<std::complex<float>>>& chEst,
                                        const std::vector<cuda_array_t<float>>& reeDiagInv,
                                        const cuda_array_t<float>& infoNoiseVarPreEq,
                                        PuschParams& puschParams);

    // output getters to be used after the estimation has been run.
    const cuda_array_t<float>& getInfoNoiseVarPostEq() const { return *m_infoNoiseVarPostEq; }
    const cuda_array_t<float>& getSinrPreEq() const { return *m_sinrPreEq; }
    const cuda_array_t<float>& getSinrPostEq() const { return *m_sinrPostEq; }

private:

    RsrpEstimator m_rsrpEstimator;

    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;
    std::vector<cuphy::tensor_device> m_tReeDiagInv;
    cuphy::tensor_device m_tInfoNoiseVarPreEq;

    // Outputs.
    std::unique_ptr<cuda_array_t<float>> m_rsrp;
    std::unique_ptr<cuda_array_t<float>> m_infoNoiseVarPostEq;
    std::unique_ptr<cuda_array_t<float>> m_sinrPreEq;
    std::unique_ptr<cuda_array_t<float>> m_sinrPostEq;
};

} // pycuphy

#endif // PYCUPHY_RSRP_HPP