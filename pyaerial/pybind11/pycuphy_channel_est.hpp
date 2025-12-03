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

#ifndef PYCUPHY_CHANNEL_EST_HPP
#define PYCUPHY_CHANNEL_EST_HPP

#include <vector>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_params.hpp"
#include "ch_est/IModule.hpp"
#include "ch_est/ch_est_settings.hpp"


namespace pycuphy {


// This is the interface when called from C++.
class ChannelEstimator {
public:

    enum DescriptorTypes {
        CH_EST               = 0,
        N_CH_EST_DESCR_TYPES = CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + 1
    };

    ChannelEstimator(const PuschParams& puschParams, cudaStream_t cuStream);
    ~ChannelEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Full channel estimate. Note that this does not get populated if only LS estimates are requested.
    const std::vector<cuphy::tensor_ref>& getChEst() const { return m_tChannelEst; };

    // LS channel estimates. Note that this only gets populated if using the LS+MMSE multi-stage algorithm,
    // or LS channel estimation only.
    const std::vector<cuphy::tensor_ref>& getLsChEst() const { return m_tDmrsLSEst; }

    const std::vector<cuphy::tensor_ref>& getDmrsDelayMean() const { return m_tDmrsDelayMean; }

    // For debugging purposes, dump channel estimates.
    void debugDump(H5DebugDump& debugDump, uint16_t numUeGrps, cudaStream_t cuStream = 0);

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void init(const PuschParams& puschParams);
    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    // Descriptor variables.
    cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelStatDescr;
    cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelDynDescr;

    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<cuphy::tensor_ref> m_tChannelEst, m_tDbg, m_tDmrsLSEst, m_tDmrsDelayMean, m_tDmrsAccum;

    std::unique_ptr<cuphyChEstSettings> m_cuphyChEstSettings;
    std::unique_ptr<ch_est::IKernelBuilder> m_chestKernelBuilder;
    std::unique_ptr<ch_est::IModule> m_chest;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyChannelEstimator {
public:
    PyChannelEstimator(const PuschParams& puschParams, uint64_t cuStream);

    const std::vector<cuda_array_t<std::complex<float>>>& estimate(PuschParams& puschParams);

private:
    ChannelEstimator m_chEstimator;
    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<cuda_array_t<std::complex<float>>> m_chEst;
};


} // pycuphy

#endif // PYCUPHY_CHANNEL_EST_HPP
