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

#ifndef PYCUPHY_TEST_PUSCH_RX_HPP
#define PYCUPHY_TEST_PUSCH_RX_HPP

#include "pycuphy_params.hpp"
#include "pycuphy_channel_est.hpp"
#include "pycuphy_channel_eq.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "pycuphy_rsrp.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_crc_check.hpp"

namespace pycuphy {

class TestPuschRxPipeline {
public:
    TestPuschRxPipeline(PuschParams& puschParams, const cudaStream_t cudaStream);

    bool runTest(PuschParams& puschParams, std::string& errMsg);

private:

    cudaStream_t m_cudaStream;

    // PUSCH Rx pipeline components
    ChannelEstimator    m_chEstimator;
    NoiseIntfEstimator  m_noiseIntfEstimator;
    ChannelEqualizer    m_chEqualizer;
    RsrpEstimator       m_rsrpEstimator;
    LdpcDerateMatch     m_derateMatch;
    LdpcDecoder         m_decoder;
    CrcChecker          m_crcChecker;
};

} // namespace pycuphy

#endif // PYCUPHY_TEST_PUSCH_RX_HPP