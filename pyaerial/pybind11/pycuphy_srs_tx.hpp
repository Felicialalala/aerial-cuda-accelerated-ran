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

#ifndef PYCUPHY_SRS_TX_HPP
#define PYCUPHY_SRS_TX_HPP

#include <vector>
#include <memory>
#include "cuda_fp16.h"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"

namespace pycuphy {

class SrsTx final {

public:
    explicit SrsTx(cudaStream_t cuStream, cuphySrsTxStatPrms_t* srsStatPrms);
    ~SrsTx();

    SrsTx(const SrsTx&) = delete;
    SrsTx& operator=(const SrsTx&) = delete;

    void run(cuphySrsTxDynPrms_t* srsDynPrms) const;

private:
    cudaStream_t m_cuStream{};
    cuphySrsTxHndl_t m_srsTxHndl{};
};


class __attribute__((visibility("default"))) PySrsTx final {

public:
    explicit PySrsTx(uint16_t nMaxSrsUes,
                     uint16_t nSlotsPerFrame,
                     uint16_t nSymbsPerSlot,
                     uint64_t cuStream);

    [[nodiscard]] const std::vector<cuda_array_t<std::complex<float>>>& run(uint16_t idxSlotInFrame,
                                                                            uint16_t idxFrame,
                                                                            const std::vector<pybind11::object>& srsPrms);

private:
    static constexpr size_t m_nElemsPerUe = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * 4;

    cudaStream_t m_cuStream{};

    std::unique_ptr<SrsTx> m_srsTx;

    cuphyTracker_t m_tracker{};
    cuphySrsTxStatPrms_t m_srsTxStatPrms{};

    std::vector<cuphyTensorPrm_t> m_pTDataSrsTxPrm;
    std::vector<cuphyUeSrsTxPrm_t> m_ueSrsPrms;
    cuphySrsTxDynPrms_t m_srsTxDynPrms{};

    // Tx buffers from cuPHY in complex half-precision.
    cuphy::unique_device_ptr<__half2> m_dLargeBufferHalf;
    std::vector<cuphy::tensor_device> m_txBuffersHalf;

    // Tx buffers to CuPy.
    // TODO: Not needed when CuPy supports cp.complex32.
    cuphy::unique_device_ptr<std::complex<float>> m_dLargeBuffer;
    std::vector<cuda_array_t<std::complex<float>>> m_txBuffers;

};

}  // namespace pycuphy

#endif // PYCUPHY_SRS_TX_HPP
