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

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "pycuphy_csirs_util.hpp"


namespace py = pybind11;

namespace pycuphy {

void readCsiRsRrcDynPrms(const py::object& pyCsiRsRrcDynPrms,
                         cuphyCsirsRrcDynPrm_t& csiRsRrcDynPrms) {
    csiRsRrcDynPrms.startRb = pyCsiRsRrcDynPrms.attr("start_prb").cast<uint16_t>();
    csiRsRrcDynPrms.nRb = pyCsiRsRrcDynPrms.attr("num_prb").cast<uint16_t>();
    const py::list freqAlloc = pyCsiRsRrcDynPrms.attr("freq_alloc");
    uint16_t freqDomain = 0;
    for(int i = 0; i < 16; i++) {
        const auto bit = freqAlloc[freqAlloc.size() - i - 1].cast<uint16_t>();
        freqDomain |= ((bit & 0x001) << i);
    }
    csiRsRrcDynPrms.freqDomain = freqDomain;
    csiRsRrcDynPrms.row = pyCsiRsRrcDynPrms.attr("row").cast<uint8_t>();
    csiRsRrcDynPrms.symbL0 = pyCsiRsRrcDynPrms.attr("symb_L0").cast<uint8_t>();
    csiRsRrcDynPrms.symbL1 = pyCsiRsRrcDynPrms.attr("symb_L1").cast<uint8_t>();
    csiRsRrcDynPrms.freqDensity = pyCsiRsRrcDynPrms.attr("freq_density").cast<uint8_t>();
    csiRsRrcDynPrms.scrambId = pyCsiRsRrcDynPrms.attr("scramb_id").cast<uint16_t>();
    csiRsRrcDynPrms.idxSlotInFrame = pyCsiRsRrcDynPrms.attr("idx_slot_in_frame").cast<uint8_t>();
    csiRsRrcDynPrms.csiType = NZP_CSI_RS;
    csiRsRrcDynPrms.cdmType = static_cast<cuphyCdmType_t>(pyCsiRsRrcDynPrms.attr("cdm_type").cast<uint8_t>());
    csiRsRrcDynPrms.beta = pyCsiRsRrcDynPrms.attr("beta").cast<float>();
    csiRsRrcDynPrms.enablePrcdBf = pyCsiRsRrcDynPrms.attr("enable_precoding").cast<uint8_t>();
    if(csiRsRrcDynPrms.enablePrcdBf) {
        csiRsRrcDynPrms.pmwPrmIdx = pyCsiRsRrcDynPrms.attr("precoding_matrix_index").cast<uint16_t>();
    }
    else {
        csiRsRrcDynPrms.pmwPrmIdx = 0;
    }

}

}  // namespace pycuphy