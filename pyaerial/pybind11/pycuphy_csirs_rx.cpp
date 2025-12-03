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

#include <bit>
#include <ranges>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "tensor_desc.hpp"
#include "cuphy_api.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_csirs_rx.hpp"
#include "pycuphy_csirs_util.hpp"
#include "pycuphy_util.hpp"

namespace py = pybind11;

namespace pycuphy {

CsiRsRx::CsiRsRx(cuphyCsirsStatPrms_t const* pStatPrms) {
    if(cuphyStatus_t status = cuphyCreateCsirsRx(&m_csirsRxHndl, pStatPrms); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyCreateCsirsRx error {}", status);
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCsirsRx()");
    }
}


CsiRsRx::~CsiRsRx() {
    if(cuphyStatus_t status = cuphyDestroyCsirsRx(m_csirsRxHndl); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyDestroyCsirsRx error {}", status);
    }
}


cuphyStatus_t CsiRsRx::run(cuphyCsirsRxDynPrms_t* pDynPrms) {
    if(cuphyStatus_t status = cuphySetupCsirsRx(m_csirsRxHndl, pDynPrms); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupCsirsRx error {}", status);
        return status;
    }

    return cuphyRunCsirsRx(m_csirsRxHndl);
}


PyCsiRsRx::PyCsiRsRx(const std::vector<uint16_t>& numPrbDlBwp) {
    const auto numCells = numPrbDlBwp.size();

    m_tracker.pMemoryFootprint = nullptr;

    m_cellStatPrms.reserve(numCells);
    std::ignore = std::ranges::for_each(numPrbDlBwp, [this](const uint16_t num) {
        m_cellStatPrms.emplace_back().nPrbDlBwp = num;
    });

    m_statPrms.pOutInfo = &m_tracker;
    m_statPrms.pCellStatPrms = m_cellStatPrms.data();
    m_statPrms.nCells = numCells;
    m_statPrms.nMaxCellsPerSlot = numCells;

    m_csiRsRx = std::make_unique<CsiRsRx>(&m_statPrms);
}


const std::vector<std::vector<cuda_array_complex_float>>& PyCsiRsRx::run(
    const py::list& pyCsiRsConfigs,
    const std::vector<cuda_array_complex_float>& rxData,
    const std::vector<int>& ueCellAssociation,
    const uint64_t cudaStream) {

    const auto numUes = rxData.size();
    m_chEst.clear();

    populateDynPrms(pyCsiRsConfigs, rxData, ueCellAssociation, cudaStream);

    // Run CSI-RS reception.
    if(cuphyStatus_t status = m_csiRsRx->run(&m_dynPrms); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Failed to run CSI-RS Rx (status code: {})", status);
        throw std::runtime_error("Failed to run CSI-RS Rx (CsiRxRx::run)!");
    }

    // Outputs to Python.
    m_chEst.resize(numUes);
    for(int ueIdx = 0; ueIdx < numUes; ueIdx++) {
        const auto& chEstTensors = m_chEstOutput[ueIdx].chEstTensor;
        const uint16_t nRrcParams = chEstTensors.size();
        m_chEst[ueIdx].reserve(nRrcParams);
        for(const auto& chEstTensor : chEstTensors) {
            const std::vector shape = {static_cast<size_t>(chEstTensor.dimensions()[0]),
                                       static_cast<size_t>(chEstTensor.dimensions()[1]),
                                       static_cast<size_t>(chEstTensor.dimensions()[2])};
            m_chEst[ueIdx].push_back(deviceToCudaArray<std::complex<float>>(chEstTensor.addr(), shape));
        }
    }

    return m_chEst;
}


uint16_t PyCsiRsRx::populateCellDynPrms(const py::list& pyCsiRsConfigs) {
    const auto numCells = pyCsiRsConfigs.size();

    uint16_t nTotalRrcPrms = 0;
    m_csiRsCellDynPrms.resize(numCells);
    for(int i = 0; i < numCells; i++) {
        const uint16_t nRrcPrms = [&pyCsiRsConfigs, i]()
        {
            const py::list& cellCsiRsConfigs = pyCsiRsConfigs[i];
            return cellCsiRsConfigs.size();
        }();

        m_csiRsCellDynPrms[i].rrcParamsOffset = nTotalRrcPrms;
        m_csiRsCellDynPrms[i].nRrcParams = nRrcPrms;
        m_csiRsCellDynPrms[i].slotBufferIdx = i;
        m_csiRsCellDynPrms[i].cellPrmStatIdx = i;

        nTotalRrcPrms += nRrcPrms;
    }

    return nTotalRrcPrms;
}


void PyCsiRsRx::populateUeDynPrms(
    const std::vector<cuda_array_complex_float>& rxData,
    const std::vector<int>& ueCellAssociation) {

    m_csiRsUeDynPrms.resize(rxData.size());
    for(int ueIdx = 0; auto& ueDynPrms : m_csiRsUeDynPrms) {
        if(const int numDims = rxData[ueIdx].get_ndim(); numDims != 3) {
            throw std::runtime_error("Invalid Rx data dimensions!");
        }
        const uint8_t numRxAnt = rxData[ueIdx].get_shape()[2];  // Shape is [nSubcarriers, nSymbols, nRxAnt]
        ueDynPrms.nRxAnt = numRxAnt;
        ueDynPrms.cellPrmStatIdx = ueCellAssociation[ueIdx];
        ueIdx++;
    }
}


void PyCsiRsRx::populateRrcDynPrms(const py::list& pyCsiRsConfigs, const uint16_t nTotalRrcPrms) {
    const auto numCells = pyCsiRsConfigs.size();
    m_csiRsRrcDynPrms.resize(nTotalRrcPrms);
    for(int i = 0, count = 0; i < numCells; i++) {
        const py::list& cellCsiRsConfigs = pyCsiRsConfigs[i];
        for(int j = 0; j < cellCsiRsConfigs.size(); j++, count++) {
            readCsiRsRrcDynPrms(cellCsiRsConfigs[j], m_csiRsRrcDynPrms[count]);
        }
    }
}


void PyCsiRsRx::populateRxData(const std::vector<cuda_array_complex_float>& rxData) {

    const auto numUes = rxData.size();
    m_dataRx.resize(numUes);
    m_dataRxTensorPrm.resize(numUes);

    for(int ueIdx = 0; const auto& data : rxData) {
        m_dataRx[ueIdx] = deviceFromCudaArray<std::complex<float>>(
            data,
            nullptr,
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_dynPrms.cuStream);
        m_dataRxTensorPrm[ueIdx].desc = m_dataRx[ueIdx].desc().handle();
        m_dataRxTensorPrm[ueIdx].pAddr = m_dataRx[ueIdx].addr();
        ueIdx++;
    }
}


void PyCsiRsRx::populateChEstBuffInfo() {

    static constexpr uint8_t nPortsLUT[] = {1, 1, 2, 4, 4, 8, 8, 8, 12, 12, 16, 16, 24, 24, 24, 32, 32, 32};

    const auto numUes = m_csiRsUeDynPrms.size();
    m_chEstOutput.resize(numUes);
    m_chEstBuffInfo.resize(numUes);

    for(int ueIdx = 0; auto& ueChEstOutput : m_chEstOutput) {

        const cuphyCsirsUeDynPrm_t& ueDynPrms = m_csiRsUeDynPrms[ueIdx];
        const cuphyCsirsCellDynPrm_t& cellDynPrms = m_csiRsCellDynPrms[ueDynPrms.cellPrmStatIdx];
        const uint16_t nRrcParams = cellDynPrms.nRrcParams;

        ueChEstOutput.startPrbChEst.resize(nRrcParams);
        ueChEstOutput.sizePrbChEst.resize(nRrcParams);
        ueChEstOutput.chEstTensor.resize(nRrcParams);
        ueChEstOutput.chEstTensorPrm.resize(nRrcParams);

        for(int rrcIdx = 0; rrcIdx < nRrcParams; rrcIdx++) {

            const auto& rrcDynPrms = m_csiRsRrcDynPrms[cellDynPrms.rrcParamsOffset + rrcIdx];

            uint16_t nRb               = rrcDynPrms.nRb;
            const uint8_t row          = rrcDynPrms.row;
            const uint8_t freqDensity  = rrcDynPrms.freqDensity;
            const uint8_t nPort        = nPortsLUT[row - 1];

            if((row==1) && (freqDensity==3)) {
                nRb = nRb * 3;
            }
            else if((freqDensity==0) || (freqDensity==1)) {
                nRb = (nRb>>1);
            }

            ueChEstOutput.startPrbChEst[rrcIdx] = rrcDynPrms.startRb;
            ueChEstOutput.sizePrbChEst[rrcIdx] = nRb;
            // TODO: Pre-allocate memory for the tensors
            ueChEstOutput.chEstTensor[rrcIdx] = cuphy::tensor_device(CUPHY_C_32F,
                                                                     nRb,
                                                                     nPort,
                                                                     ueDynPrms.nRxAnt,
                                                                     cuphy::tensor_flags::align_tight);
            CUDA_CHECK(cudaMemsetAsync(ueChEstOutput.chEstTensor[rrcIdx].addr(),
                                       0,
                                       nRb * nPort * ueDynPrms.nRxAnt * sizeof(std::complex<float>),
                                       m_dynPrms.cuStream));
            ueChEstOutput.chEstTensorPrm[rrcIdx].desc = ueChEstOutput.chEstTensor[rrcIdx].desc().handle();
            ueChEstOutput.chEstTensorPrm[rrcIdx].pAddr = ueChEstOutput.chEstTensor[rrcIdx].addr();
        }

        m_chEstBuffInfo[ueIdx].nCsirs = nRrcParams;
        m_chEstBuffInfo[ueIdx].startPrb = ueChEstOutput.startPrbChEst.data();
        m_chEstBuffInfo[ueIdx].sizePrb = ueChEstOutput.sizePrbChEst.data();
        m_chEstBuffInfo[ueIdx].tChEstBuffer = ueChEstOutput.chEstTensorPrm.data();
        ueIdx++;
    }
}


void PyCsiRsRx::populateDynPrms(
    const py::list& pyCsiRsConfigs,
    const std::vector<cuda_array_complex_float>& rxData,
    const std::vector<int>& ueCellAssociation,
    const uint64_t cudaStream) {

    const auto numUes = rxData.size();
    if(numUes != ueCellAssociation.size()) {
        throw std::runtime_error("Number of UEs does not match with the given cell association!");
    }
    m_dynPrms.nUes = numUes;

    const auto numCells = pyCsiRsConfigs.size();
    m_dynPrms.nCells = numCells;
    m_dynPrms.cuStream = std::bit_cast<cudaStream_t>(cudaStream);

    const uint16_t nTotalRrcPrms = populateCellDynPrms(pyCsiRsConfigs);
    m_dynPrms.pCellParam = m_csiRsCellDynPrms.data();

    populateRrcDynPrms(pyCsiRsConfigs, nTotalRrcPrms);
    m_dynPrms.pRrcDynPrm = m_csiRsRrcDynPrms.data();

    populateUeDynPrms(rxData, ueCellAssociation);
    m_dynPrms.pUeParam = m_csiRsUeDynPrms.data();

    populateRxData(rxData);
    m_dataIn.pTDataRx = m_dataRxTensorPrm.data();
    m_dynPrms.pDataIn = &m_dataIn;

    populateChEstBuffInfo();
    m_dataOut.pChEstBuffInfo = m_chEstBuffInfo.data();
    m_dynPrms.pDataOut = &m_dataOut;

    m_dynPrms.procModeBmsk = CSIRS_PROC_MODE_GRAPHS;
    m_dynPrms.chan_graph = nullptr;

}

}  // namespace pycuphy