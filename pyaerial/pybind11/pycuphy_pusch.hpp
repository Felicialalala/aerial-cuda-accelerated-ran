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

#ifndef PYCUPHY_PUSCH_HPP
#define PYCUPHY_PUSCH_HPP

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "hdf5hpp.hpp"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

class PuschPipeline {

public:

    PuschPipeline(const py::object& statPrms, uint64_t cuStream);
    ~PuschPipeline();

    void setupPuschRx(const py::object& dynPrms);
    void runPuschRx();
    void writeDbgBufSynch();

private:

    void createPuschRx(cudaStream_t cuStream);
    void destroyPuschRx();

    PuschParams                          m_puschParams;
    cuphyPuschRxHndl_t                   m_puschHandle;
    std::unique_ptr<hdf5hpp::hdf5_file>  m_debugFile;
};

} // namespace pycuphy

#endif // PYCUPHY_PUSCH_HPP