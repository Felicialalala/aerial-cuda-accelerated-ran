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

#ifndef PYCUPHY_CSIRS_UTIL_HPP
#define PYCUPHY_CSIRS_UTIL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"


namespace pycuphy {

/**
 * @brief Read CSI-RS RRC dynamic parameters from a Python object
 *
 * This function extracts CSI-RS RRC dynamic parameters from a Python object
 * and populates the corresponding C++ structure. The function handles the
 * conversion of various parameter types and performs necessary bit operations
 * for fields like frequency domain allocation.
 *
 * @param pyCsiRsRrcDynPrms Python object containing CSI-RS RRC parameters
 * @param csiRsRrcDynPrms Output C++ structure to populate
 *
 * The Python object should have the following attributes:
 * - start_prb (uint16_t): Starting PRB index
 * - num_prb (uint16_t): Number of PRBs
 * - prb_bitmap (list): 16-bit bitmap for frequency domain allocation
 * - row (uint8_t): Row index for CSI-RS configuration
 * - symb_L0 (uint8_t): First symbol index
 * - symb_L1 (uint8_t): Second symbol index
 * - freq_density (uint8_t): Frequency density
 * - scramb_id (uint16_t): Scrambling ID
 * - idx_slot_in_frame (uint8_t): Slot index in frame
 * - cdm_type (uint8_t): CDM type
 * - beta (float): Power control parameter
 * - enable_precoding (uint8_t): Whether precoding is enabled
 * - precoding_matrix_index (uint16_t): Precoding matrix index (if enabled)
 */
void readCsiRsRrcDynPrms(const pybind11::object& pyCsiRsRrcDynPrms,
                         cuphyCsirsRrcDynPrm_t& csiRsRrcDynPrms);

}  // namespace pycuphy


#endif // PYCUPHY_CSIRS_UTIL_HPP