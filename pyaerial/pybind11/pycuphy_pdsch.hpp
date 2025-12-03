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

#ifndef PYCUPHY_PDSCH_HPP
#define PYCUPHY_PDSCH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

class PdschPipeline {

public:
    PdschPipeline(const py::object& statPrms);
    ~PdschPipeline();

    void setupPdschTx(const py::object& dynPrms);
    void runPdschTx();

    // Return LDPC encoder output bits as NumPy array (on host).
    const py::array_t<float>& getLdpcOutputPerTbPerCell(int cellIdx, int tbIdx);

private:
    PdschParams m_pdschParams;
    cuphyPdschTxHndl_t m_pdschHandle;

    py::array_t<float> m_ldpcOutput;

    void createPdschTx();
};


}  // namespace pycuphy

#endif