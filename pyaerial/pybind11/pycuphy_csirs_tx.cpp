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

#include "cuphy.h"
#include "cuphy.hpp"
#include "tensor_desc.hpp"
#include "cuphy_api.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_csirs_tx.hpp"
#include "pycuphy_csirs_util.hpp"
#include "pycuphy_util.hpp"

namespace py = pybind11;

namespace pycuphy {


CsiRsTx::CsiRsTx(cuphyCsirsStatPrms_t const* pStatPrms) {
    cuphyStatus_t status = cuphyCreateCsirsTx(&m_csirsTxHndl, pStatPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyCreateCsirsTx error {}", status);
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCsirsTx()");
    }
}


cuphyStatus_t CsiRsTx::run(cuphyCsirsDynPrms_t* pDynPrms) {
    cuphyStatus_t status = cuphySetupCsirsTx(m_csirsTxHndl, pDynPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupCsirsTx error {}", status);
        return status;
    }

    status = cuphyRunCsirsTx(m_csirsTxHndl);
    return status;
}


CsiRsTx::~CsiRsTx() {
    cuphyStatus_t status = cuphyDestroyCsirsTx(m_csirsTxHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyDestroyCsirsTx error {}", status);
    }
}


PyCsiRsTx::PyCsiRsTx(const std::vector<uint16_t>& numPrbDlBwp, const std::vector<uint16_t>& numAntDl) {
    uint16_t numCells = numPrbDlBwp.size();

    m_tracker.pMemoryFootprint = nullptr;

    m_cellStatPrms.resize(numCells);
    for(int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        memset(&m_cellStatPrms[cellIdx], 0, sizeof(cuphyCellStatPrm_t));  // Other parameters not used, set to zero.
        m_cellStatPrms[cellIdx].nPrbDlBwp = numPrbDlBwp[cellIdx];
        m_cellStatPrms[cellIdx].nTxAnt = numAntDl[cellIdx];
    }

    m_statPrms.pOutInfo = &m_tracker;
    m_statPrms.pCellStatPrms = m_cellStatPrms.data();
    m_statPrms.nCells = numCells;
    m_statPrms.nMaxCellsPerSlot = numCells;

    m_csiRsTx = std::make_unique<CsiRsTx>(&m_statPrms);

    // Pre-allocate memory for half-precision transmit buffers.
    m_txBufHalf.resize(MAX_CELLS_PER_SLOT);
    // z-dimension cannot be MAX_DL_LAYERS, as there are TVs with 24 or 32 ports.
    constexpr size_t nElems =  MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * CUPHY_CSIRS_MAX_ANTENNA_PORTS;
    for(auto& buf : m_txBufHalf) {
        buf = cuphy::make_unique_device<__half2>(nElems);
    }
}


void PyCsiRsTx::run(const py::list& pyCsiRsConfigs,
                    const py::list& precodingMatrices,
                    std::vector<cuda_array_t<std::complex<float>>>& txBuffers,
                    uint64_t cudaStream) {

    // Create the dynamic parameters object.
    m_dynPrms.cuStream = reinterpret_cast<cudaStream_t>(cudaStream);

    const uint16_t numCells = pyCsiRsConfigs.size();
    m_dynPrms.nCells = numCells;

    uint16_t nTotalRrcPrms = 0;
    m_csiRsCellDynPrms.resize(numCells);
    for(int i = 0; i < numCells; i++) {
        const py::list& cellCsiRsConfigs = pyCsiRsConfigs[i];
        const uint16_t nRrcPrms = cellCsiRsConfigs.size();

        m_csiRsCellDynPrms[i].rrcParamsOffset = nTotalRrcPrms;
        m_csiRsCellDynPrms[i].nRrcParams = nRrcPrms;
        m_csiRsCellDynPrms[i].slotBufferIdx = i;
        m_csiRsCellDynPrms[i].cellPrmStatIdx = i;

        nTotalRrcPrms += nRrcPrms;
    }
    m_dynPrms.pCellParam = m_csiRsCellDynPrms.data();

    m_csiRsRrcDynPrms.resize(nTotalRrcPrms);
    for(int i = 0, count = 0; i < numCells; i++) {
        const py::list& cellCsiRsConfigs = pyCsiRsConfigs[i];
        for(int j = 0; j < cellCsiRsConfigs.size(); j++, count++) {
            readCsiRsRrcDynPrms(cellCsiRsConfigs[j], m_csiRsRrcDynPrms[count]);
        }
    }
    m_dynPrms.pRrcDynPrm = m_csiRsRrcDynPrms.data();
    m_dynPrms.procModeBmsk = CSIRS_PROC_MODE_GRAPHS;
    m_dynPrms.chan_graph = nullptr;

    // Data output.
    m_txBuffer.resize(numCells);
    m_txBufferTensorPrm.resize(numCells);
    if(txBuffers.size() != numCells) {
        throw std::runtime_error("The number of Tx buffers does not match with CSI-RS cell dynamic parameters!");
    }

    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        const cuda_array_t<std::complex<float>> cellTxBuffer = txBuffers[cellIdx];
        m_txBuffer[cellIdx] = deviceFromCudaArray<std::complex<float>>(
            cellTxBuffer,
            static_cast<void*>(m_txBufHalf[cellIdx].get()),  // Pre-allocated memory.
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            m_dynPrms.cuStream);

        m_txBufferTensorPrm[cellIdx].desc = m_txBuffer[cellIdx].desc().handle();
        m_txBufferTensorPrm[cellIdx].pAddr = m_txBuffer[cellIdx].addr();
    }

    m_csiRsDataOut.pTDataTx = m_txBufferTensorPrm.data();
    m_dynPrms.pDataOut = &m_csiRsDataOut;

    // Precoding matrices.
    m_dynPrms.nPrecodingMatrices = precodingMatrices.size();
    m_csiRsPmW.resize(m_dynPrms.nPrecodingMatrices);
    if(m_dynPrms.nPrecodingMatrices > 0) {
        for(int pmwIdx = 0; pmwIdx < m_dynPrms.nPrecodingMatrices; pmwIdx++) {
            const py::array& temp = precodingMatrices[pmwIdx];
            const py::array_t<std::complex<float>>& pmwArray = temp;
            const py::buffer_info& buf = pmwArray.request();
            m_csiRsPmW[pmwIdx].nPorts = buf.shape[1];  // Precoding matrix shape is [nLayers, nPorts]
            const std::complex<float> *ptr = static_cast<std::complex<float> *>(buf.ptr);
            for (size_t idx = 0; idx < buf.size; idx++){
                m_csiRsPmW[pmwIdx].matrix[idx].x = __float2half(ptr[idx].real());
                m_csiRsPmW[pmwIdx].matrix[idx].y = __float2half(ptr[idx].imag());
            }
        }
        m_dynPrms.pPmwParams = m_csiRsPmW.data();
    }
    else
        m_dynPrms.pPmwParams = nullptr;

    // Run CSI-RS transmission.
    cuphyStatus_t status = m_csiRsTx->run(&m_dynPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Failed to run CSI-RS Tx (status code: {})", status);
        throw std::runtime_error("Failed to run CSI-RS Tx (CsiRxTx::run)!");
    }

    // Convert back to full precision, write output.
    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        const std::vector shape = txBuffers[cellIdx].get_shape();
        txBuffers[cellIdx] = deviceToCudaArray<std::complex<float>>(
            m_txBuffer[cellIdx].addr(),
            txBuffers[cellIdx].get_device_ptr(),
            shape,
            CUPHY_C_16F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_dynPrms.cuStream
        );
    }
}



}  // namespace pycuphy