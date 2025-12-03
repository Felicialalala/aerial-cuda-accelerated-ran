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

#ifndef PYCUPHY_SRS_UTIL_HPP
#define PYCUPHY_SRS_UTIL_HPP

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cuphy.hpp"
#include "cuphy_api.h"

namespace pycuphy {

void readSrsCellParams(std::vector<cuphySrsCellPrms_t>& srsCellPrms,
                       const std::vector<pybind11::object>& pySrsCellPrms);

void readSrsCellDynParams(std::vector<cuphySrsCellDynPrm_t>& srsCellPrms,
                          const std::vector<pybind11::object>& pySrsRxCellConfigs);

void readUeSrsParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms,
                     const std::vector<pybind11::object>& pyUeSrsPrms);

void readUeSrsRxParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms,
                       std::vector<int>& numPrgs,
                       const std::vector<pybind11::object>& pySrsRxUeConfigs);

void readUeSrsTxParams(std::vector<cuphyUeSrsTxPrm_t>& ueSrsPrms,
                       uint16_t slotIdx,
                       uint16_t frameIdx,
                       const std::vector<pybind11::object>& pySrsConfigs);


class SrsTensorPrms final {

public:
    explicit SrsTensorPrms(const pybind11::dict& chEstParams, cudaStream_t cuStream);

    [[nodiscard]] const cuphySrsFilterPrms_t& getSrsFilterPrms() const { return m_srsFilterPrms; }
    [[nodiscard]] const cuphySrsRkhsPrms_t& getSrsRkhsPrms() const { return m_srsRkhsPrms; }

private:

    // Filter tensors and parameters.
    void readChEstParams();

    cuphySrsFilterPrms_t m_srsFilterPrms{};
    cuphySrsRkhsPrms_t m_srsRkhsPrms{};

    cuphy::tensor_device m_tPrmFocc_table;
    cuphy::tensor_device m_tPrmFocc_comb2_table;
    cuphy::tensor_device m_tPrmFocc_comb4_table;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts8_wide;

    cuphy::tensor_device m_tPrmW_comb4_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts6_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts12_wide;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts8_narrow;

    cuphy::tensor_device m_tPrmW_comb4_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts6_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts12_narrow;

    cuphy::tensor_device m_eigenVecTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device m_eigenValueTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device m_eigenCorrTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device m_secondStageTwiddleFactorsTable;
    cuphy::tensor_device m_secondStageFourierPermTable;

    cuphy::buffer<cuphyRkhsGridPrms_t, cuphy::pinned_alloc> m_cpuBufferRkhsGridPrms;
};


}

#endif // PYCUPHY_SRS_TX_HPP