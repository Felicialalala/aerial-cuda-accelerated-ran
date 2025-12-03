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

#ifndef PYCUPHY_CSIRS_TX_HPP
#define PYCUPHY_CSIRS_TX_HPP
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuda_array_interface.hpp"


namespace pycuphy {


// This is the C++ wrapper around cuPHY.
class CsiRsTx final {

public:
    explicit CsiRsTx(cuphyCsirsStatPrms_t const* pStatPrms);
    ~CsiRsTx();
    CsiRsTx(const CsiRsTx&) = delete;
    CsiRsTx& operator=(const CsiRsTx&) = delete;

    [[nodiscard]] cuphyStatus_t run(cuphyCsirsDynPrms_t* pDynPrms);

private:
    cuphyCsirsTxHndl_t m_csirsTxHndl{};
};


// This is for the Python bindings.
class __attribute__((visibility("default"))) PyCsiRsTx final {

public:
    explicit PyCsiRsTx(const std::vector<uint16_t>& numPrbDlBwp, const std::vector<uint16_t>& numAntDl);

    void run(const pybind11::list& pyCsiRsConfigs,
             const pybind11::list& precodingMatrices,
             std::vector<cuda_array_t<std::complex<float>>>& txBuffers,
             uint64_t cudaStream);

private:
    cuphyCsirsStatPrms_t    m_statPrms{};
    cuphyCsirsDynPrms_t     m_dynPrms{};

    cuphyTracker_t                      m_tracker{};
    std::vector<cuphyCellStatPrm_t>     m_cellStatPrms;
    std::vector<cuphyCsirsRrcDynPrm_t>  m_csiRsRrcDynPrms;
    std::vector<cuphyCsirsCellDynPrm_t> m_csiRsCellDynPrms;
    cuphyCsirsDataOut_t                 m_csiRsDataOut{};
    std::vector<cuphyPmWOneLayer_t>     m_csiRsPmW;

    // Transmit buffer tensors (in/out).
    // cuPHY works on 16-bit+16-bit complex floats, needs conversion.
    // TODO: Remove conversions once cuPy supports cp.complex32.
    std::vector<cuphy::tensor_device>               m_txBuffer;
    std::vector<cuphyTensorPrm_t>                   m_txBufferTensorPrm;
    std::vector<cuphy::unique_device_ptr<__half2>>  m_txBufHalf;

    std::unique_ptr<CsiRsTx> m_csiRsTx;
};


}  // namespace pycuphy


#endif // PYCUPHY_CSIRS_TX_HPP