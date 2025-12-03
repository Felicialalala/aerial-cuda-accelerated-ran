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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "cuphy.hpp"
#include "pdsch_tx.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_pdsch.hpp"

namespace py = pybind11;

namespace pycuphy {

PdschPipeline::PdschPipeline(const py::object& statPrms) :
m_pdschParams(statPrms) {
    createPdschTx();
}


PdschPipeline::~PdschPipeline() {
    cuphyStatus_t status = cuphyDestroyPdschTx(m_pdschHandle);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,
                   "PdschPipeline::~PdschPipeline() failed to call cuphyDestroyPdschTx()");
    }
}


void PdschPipeline::createPdschTx() {
    cuphyStatus_t status = cuphyCreatePdschTx(&m_pdschHandle, &m_pdschParams.m_pdschStatPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePdschTx()");
    }
}


void PdschPipeline::setupPdschTx(const py::object& dynPrms) {
    m_pdschParams.setDynPrms(dynPrms);

    // Setup pipeline.
    cuphyStatus_t status = cuphySetupPdschTx(m_pdschHandle, &m_pdschParams.m_pdschDynPrms, nullptr);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphySetupPdschTx()");
    }
}


void PdschPipeline::runPdschTx() {

    cuphyPdschProcMode_t pdschProcMode = static_cast<cuphyPdschProcMode_t>(
        (uint32_t)PDSCH_PROC_MODE_NO_GRAPHS | (uint32_t)PDSCH_INTER_CELL_BATCHING
    );

    cuphyStatus_t status = cuphyRunPdschTx(m_pdschHandle, pdschProcMode);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyRunPdschTx()");
    }
}


const py::array_t<float>& PdschPipeline::getLdpcOutputPerTbPerCell(int cellIdx, int tbIdx) {

    PdschTx* pipelinePtr = static_cast<PdschTx*>(m_pdschHandle);

    // Outputs.
    int numCbsForTbIdx = 0;
    int numBitsForTbIdx = 0;

    cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> hostOutTensor = pipelinePtr->getHostOutputLDPCTBPerCell(
        cellIdx, tbIdx, &numCbsForTbIdx, &numBitsForTbIdx, pipelinePtr->dynamic_params->cuStream
    );

    // Convert to float for Numpy and return the Numpy array.
    uint32_t dim0 = hostOutTensor.dimensions()[0];
    uint32_t dim1 = hostOutTensor.dimensions()[1];
    cuphy::tensor_pinned outTensor(CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    outTensor.convert(hostOutTensor, pipelinePtr->dynamic_params->cuStream);
    CUDA_CHECK(cudaStreamSynchronize(pipelinePtr->dynamic_params->cuStream));
    m_ldpcOutput = hostToNumpy<float>((float*)outTensor.addr(), dim0, dim1);
    return m_ldpcOutput;
}


}  // namespace pycuphy
