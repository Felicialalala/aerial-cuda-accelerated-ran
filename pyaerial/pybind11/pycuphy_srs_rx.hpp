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

#ifndef PYCUPHY_SRS_RX_HPP
#define PYCUPHY_SRS_RX_HPP

#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_srs_util.hpp"

namespace pycuphy {


class SrsRx final {

public:
    explicit SrsRx(cudaStream_t cuStream, cuphySrsStatPrms_t* statPrms);

    void run(cuphySrsDynPrms_t* dynPrms);

private:
    cudaStream_t m_cuStream{};

    cuphySrsRxHndl_t m_srsRxHndl{};
    cuphySrsBatchPrmHndl_t m_batchPrmHndl{};
};


class __attribute__((visibility("default"))) PySrsRx final {

public:
    explicit PySrsRx(uint16_t nCells,
                     const std::vector<uint16_t>& nRxAnt,
                     uint8_t chEstAlgoIdx,
                     uint8_t enableDelayOffsetCorrection,
                     const pybind11::dict& chEstParams,
                     uint16_t nMaxSrsUes,
                     uint64_t cuStream);

    [[nodiscard]] const std::vector<pybind11::array_t<std::complex<float>>>& run(const std::vector<cuda_array_t<std::complex<float>>>& rxData,
                                                                                 const std::vector<pybind11::object>& pySrsRxUeConfigs,
                                                                                 const std::vector<pybind11::object>& pySrsRxCellConfigs);

    [[nodiscard]] const std::vector<pybind11::array_t<std::complex<float>>>& getChEstToL2();
    [[nodiscard]] const std::vector<cuphySrsReport_t>& getSrsReport() const { return m_srsReports; }
    [[nodiscard]] const pybind11::array_t<float>& getRbSnrBuffer();
    [[nodiscard]] const std::vector<uint32_t>& getRbSnrBufferOffsets() const { return m_rbSnrBuffOffsets; }

private:

    static constexpr size_t m_nRxDataElemsPerCell = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * MAX_N_ANTENNAS_SUPPORTED * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI;
    static constexpr size_t m_nChEstElemsPerUe = MAX_N_PRBS_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * 4;

    uint16_t m_nMaxSrsUes{};
    uint16_t m_nSrsUes{};

    cudaStream_t m_cuStream{};

    std::unique_ptr<SrsRx> m_srsRx;

    cuphy::unique_device_ptr<__half2> m_dRxData;
    std::vector<cuphy::tensor_device> m_dRxDataTensor;

    // Static SRS parameters.
    cuphySrsStatPrms_t m_statPrms{};
    cuphyTracker_t m_tracker{};
    std::vector<cuphyCellStatPrm_t> m_cellStatPrms;
    cuphySrsStatDbgPrms_t m_srsStatDbgPrms{};

    // Dynamic SRS parameters.
    cuphySrsDynPrms_t m_dynPrms{};

    std::vector<cuphySrsCellDynPrm_t> m_srsCellDynPrms;
    std::vector<cuphyUeSrsPrm_t> m_ueSrsPrms;
    cuphySrsCellGrpDynPrm_t m_srsCellGrpDynPrms{};

    // Inputs.
    cuphySrsDataIn_t m_dataIn{};
    std::vector<cuphyTensorPrm_t> m_tDataRx;

    // Outputs.
    cuphySrsDataOut_t m_dataOut{};
    cuphy::unique_device_ptr<std::complex<float>> m_dChEsts;
    cuphy::unique_pinned_ptr<std::complex<float>> m_hChEsts;
    std::vector<float> m_rbSnrs;
    std::vector<uint32_t> m_rbSnrBuffOffsets;
    std::vector<cuphySrsChEstBuffInfo_t> m_srsChEstBuffInfo;
    std::vector<cuphySrsReport_t> m_srsReports;
    std::vector<cuphy::tensor_device> m_tSrsChEsts;
    std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>> m_chEstCpuBuff;
    std::vector<cuphySrsChEstToL2_t> m_srsChEstToL2;

    // Status.
    cuphySrsStatusOut_t m_srsStatusOut{};

    // Debug parameters.
    cuphySrsDynDbgPrms_t m_srsDynDbgPrms{};

    // Filter tensors and parameters.
    SrsTensorPrms m_tensorPrms;

    // Output to Python.
    std::vector<pybind11::array_t<std::complex<float>>> m_chEsts;
    std::vector<pybind11::array_t<std::complex<float>>> m_chEstsToL2;
    pybind11::array_t<float> m_rbSnrArray;

};

} // namespace pycuphy


#endif // PYCUPHY_SRS_RX_HPP