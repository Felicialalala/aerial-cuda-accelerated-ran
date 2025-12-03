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

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuda_fp16.h"
#include "cuphy.h"
#include "cuphy_api.h"
#include "pycuphy_srs_util.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_srs_tx.hpp"


namespace py = pybind11;

namespace pycuphy {

SrsTx::SrsTx(const cudaStream_t cuStream, cuphySrsTxStatPrms_t* srsStatPrms):
m_cuStream(cuStream) {

    if(cuphyStatus_t status = cuphyCreateSrsTx(&m_srsTxHndl, srsStatPrms); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyCreateSrsTx error {}", status);
        throw std::runtime_error("SrsTx::SrsTx: cuphyCreateSrsTx error!");
    }
}


void SrsTx::run(cuphySrsTxDynPrms_t* srsDynPrms) const {
    if(cuphyStatus_t status = cuphySetupSrsTx(m_srsTxHndl, srsDynPrms); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupSrsTx error {}", status);
        throw std::runtime_error("SrsTx::run: cuphySetupSrsTx error!");
    }

    if(cuphyStatus_t status = cuphyRunSrsTx(m_srsTxHndl); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyRunSrsTx error {}", status);
        throw std::runtime_error("SrsTx::run: cuphyRunSrsTx error!");
    }
}


SrsTx::~SrsTx() {
    if(cuphyStatus_t status = cuphyDestroySrsTx(m_srsTxHndl); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyDestroySrsTx error {}", status);
    }
}


PySrsTx::PySrsTx(const uint16_t nMaxSrsUes,
                 const uint16_t nSlotsPerFrame,
                 const uint16_t nSymbsPerSlot,
                 const uint64_t cuStream):
m_cuStream(reinterpret_cast<cudaStream_t>(cuStream)) {

    m_srsTxStatPrms.nMaxSrsUes = nMaxSrsUes;
    m_srsTxStatPrms.nSlotsPerFrame = nSlotsPerFrame;
    m_srsTxStatPrms.nSymbsPerSlot = nSymbsPerSlot;
    m_tracker.pMemoryFootprint = nullptr;
    m_srsTxStatPrms.pOutInfo = &m_tracker;
    m_srsTx = std::make_unique<SrsTx>(m_cuStream, &m_srsTxStatPrms);

    // Memory allocation for outputs. Half precision for internal computation, full precision
    // for CuPy outputs. A single large buffer for all UEs.
    m_dLargeBufferHalf = cuphy::make_unique_device<__half2>(m_nElemsPerUe * nMaxSrsUes);
    m_dLargeBuffer = cuphy::make_unique_device<std::complex<float>>(m_nElemsPerUe * nMaxSrsUes);
}


const std::vector<cuda_array_t<std::complex<float>>>& PySrsTx::run(const uint16_t idxSlotInFrame,
                                                                   const uint16_t idxFrame,
                                                                   const std::vector<pybind11::object>& srsPrms) {
    const uint16_t nUes = srsPrms.size();
    m_pTDataSrsTxPrm.resize(nUes);
    m_ueSrsPrms.resize(nUes);
    m_txBuffersHalf.resize(nUes);
    m_txBuffers.clear();

    // Reset buffers.
    CUDA_CHECK(cudaMemsetAsync(m_dLargeBufferHalf.get(), 0, nUes * m_nElemsPerUe * sizeof(__half2), m_cuStream));

    // Set SRS dynamic parameters.
    readUeSrsTxParams(m_ueSrsPrms, idxSlotInFrame, idxFrame, srsPrms);

    constexpr uint16_t numTxAnt = 4;
    for (int ueIdx = 0; auto& buf : m_txBuffersHalf) {
        buf = cuphy::tensor_device(m_dLargeBufferHalf.get() + ueIdx * m_nElemsPerUe,
                                   CUPHY_C_16F,
                                   MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB,
                                   OFDM_SYMBOLS_PER_SLOT,
                                   numTxAnt,
                                   cuphy::tensor_flags::align_tight);
        m_pTDataSrsTxPrm[ueIdx].desc = buf.desc().handle();
        m_pTDataSrsTxPrm[ueIdx].pAddr = buf.addr();
        ueIdx++;
    }

    m_srsTxDynPrms.cuStream = m_cuStream;
    m_srsTxDynPrms.chan_graph = nullptr;
    m_srsTxDynPrms.nSrsUes = nUes;
    m_srsTxDynPrms.pUeSrsTxPrms = m_ueSrsPrms.data();
    m_srsTxDynPrms.pTDataSrsTx = m_pTDataSrsTxPrm.data();

    // Run SRS Tx.
    m_srsTx->run(&m_srsTxDynPrms);

    // Convert back to full precision, write output.
    m_txBuffers.reserve(nUes);
    for (int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        const std::vector shape = {static_cast<size_t>(m_txBuffersHalf[ueIdx].dimensions()[0]),
                                   static_cast<size_t>(m_txBuffersHalf[ueIdx].dimensions()[1]),
                                   static_cast<size_t>(m_txBuffersHalf[ueIdx].dimensions()[2])};
        m_txBuffers.push_back(deviceToCudaArray<std::complex<float>>(
            m_txBuffersHalf[ueIdx].addr(),
            m_dLargeBuffer.get() + ueIdx * m_nElemsPerUe,
            shape,
            CUPHY_C_16F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream));
    }

    return m_txBuffers;
}


} // namespace pycuphy