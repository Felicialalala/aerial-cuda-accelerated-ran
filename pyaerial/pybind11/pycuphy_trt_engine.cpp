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
#include <vector>
#include <numeric>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "tensor_desc.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_trt_engine.hpp"


namespace py = pybind11;


namespace pycuphy {

TrtEngine::TrtEngine(const std::string& trtModelFile,
                     const uint32_t maxBatchSize,
                     const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
                     const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
                     cudaStream_t cuStream):
m_cuStream(cuStream) {
    cuphyStatus_t status = cuphyCreateTrtEngine(&m_trtEngineHndl,
                                                trtModelFile.c_str(),
                                                maxBatchSize,
                                                const_cast<cuphyTrtTensorPrms_t*>(inputTensorPrms.data()),
                                                inputTensorPrms.size(),
                                                const_cast<cuphyTrtTensorPrms_t*>(outputTensorPrms.data()),
                                                outputTensorPrms.size(),
                                                m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateTrtEngine()");
    }
}


TrtEngine::~TrtEngine() {
    cuphyStatus_t status = cuphyDestroyTrtEngine(m_trtEngineHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,
                   "TrtEngine::~TrtEngine() failed to call cuphyDestroyTrtEngine()");
    }
}


cuphyStatus_t TrtEngine::run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors) {
    return run(inputTensors, outputTensors, 0);
}


cuphyStatus_t TrtEngine::run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors, const uint32_t batchSize) {
    cuphyStatus_t setupStatus = cuphySetupTrtEngine(m_trtEngineHndl,
                                                    (void**)inputTensors.data(),
                                                    inputTensors.size(),
                                                    (void**)outputTensors.data(),
                                                    outputTensors.size(),
                                                    batchSize);
    if(setupStatus != CUPHY_STATUS_SUCCESS) {
        return setupStatus;
    }

    cuphyStatus_t runStatus = cuphyRunTrtEngine(m_trtEngineHndl, m_cuStream);
    return runStatus;
}


PyTrtEngine::PyTrtEngine(const std::string& trtModelFile,
                         uint32_t maxBatchSize,
                         const std::vector<std::string>& inputNames,
                         const std::vector<std::vector<int>>& inputShapes,
                         const std::vector<cuphyDataType_t>& inputDataTypes,
                         const std::vector<std::string>& outputNames,
                         const std::vector<std::vector<int>>& outputShapes,
                         const std::vector<cuphyDataType_t>& outputDataTypes,
                         uint64_t cuStream):
m_linearAlloc(getBufferSize(maxBatchSize, outputShapes)),
m_inputNames(inputNames),
m_outputNames(outputNames),
m_cuStream((cudaStream_t)cuStream) {

    getTensorPrms(m_inputNames, inputShapes, inputDataTypes, m_inputTensorPrms);
    getTensorPrms(m_outputNames, outputShapes, outputDataTypes, m_outputTensorPrms);

    m_trtEngine = std::make_unique<TrtEngine>(trtModelFile, maxBatchSize, m_inputTensorPrms, m_outputTensorPrms, m_cuStream);

    m_inputBuffers.resize(inputNames.size());

    // Allocate memory for outputs.
    m_outputBuffers.resize(outputNames.size());
    for(int i = 0; i < outputNames.size(); i++)  {
        size_t nBytes = get_cuphy_type_storage_element_size(outputDataTypes[i]);
        nBytes *= std::accumulate(outputShapes[i].cbegin(), outputShapes[i].cend(), 1, std::multiplies<int>{});
        nBytes *= maxBatchSize;
        m_outputBuffers[i] = m_linearAlloc.alloc(nBytes);
    }
}


const py::dict& PyTrtEngine::run(const py::dict& inputTensors) {

    if(m_inputTensorPrms.size() != inputTensors.size()) {
        throw std::runtime_error("Invalid number TRT inputs!");
    }

    uint32_t batchSize{};
    for(int index = 0; index < inputTensors.size(); index++) {

        std::string tensorName = std::string(m_inputTensorPrms[index].name);
        switch(m_inputTensorPrms[index].dataType) {

        case CUPHY_R_32F:
            {
                const cuda_array_t<float>& inputTensor = inputTensors[py::str(tensorName)].cast<cuda_array_t<float>>();
                m_inputBuffers[index] = inputTensor.get_device_ptr();
                batchSize = inputTensor.get_shape()[0];
                break;
            }
        case CUPHY_R_32I:
            {
                const cuda_array_t<int>& inputTensor = inputTensors[py::str(tensorName)].cast<cuda_array_t<int>>();
                m_inputBuffers[index] = inputTensor.get_device_ptr();
                batchSize = inputTensor.get_shape()[0];
                break;
            }
        default:
            throw std::runtime_error("Invalid input data type!");
        }
    }

    cuphyStatus_t status = m_trtEngine->run(m_inputBuffers, m_outputBuffers, batchSize);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyRunTrtEngine()");
    }

    // Move outputs to CuPy arrays.
    for(int i = 0; i < m_outputBuffers.size(); i++) {

        // Set batch size correctly.
        std::vector<size_t> tensorShape(m_outputTensorPrms[i].dims, m_outputTensorPrms[i].dims + m_outputTensorPrms[i].nDims);
        tensorShape.insert(tensorShape.begin(), batchSize);

        switch(m_outputTensorPrms[i].dataType) {
        case CUPHY_R_32F:
            m_outputTensors[py::str(m_outputTensorPrms[i].name)] = deviceToCudaArray<float>(const_cast<void*>(m_outputBuffers[i]), tensorShape);
            break;
        case CUPHY_R_32I:
            m_outputTensors[py::str(m_outputTensorPrms[i].name)] = deviceToCudaArray<int>(const_cast<void*>(m_outputBuffers[i]), tensorShape);
            break;
        default:
            throw std::runtime_error("Invalid output data type!");
        }
    }
    return m_outputTensors;
}


size_t PyTrtEngine::getBufferSize(uint32_t maxBatchSize, const std::vector<std::vector<int>>& outputShapes) const {

    // Allocate internal buffers.
    static constexpr uint32_t N_BYTES_PER_ELEM = 4;
    static constexpr uint32_t EXTRA_PADDING = LINEAR_ALLOC_PAD_BYTES;

    size_t nBytesBuffer = 0;
    for(const auto& shape : outputShapes) {
        nBytesBuffer += N_BYTES_PER_ELEM * maxBatchSize * std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>{}) + EXTRA_PADDING;
    }

    return nBytesBuffer;
}


void getTensorPrms(const std::vector<std::string>& names,
                   const std::vector<std::vector<int>>& shapes,
                   const std::vector<cuphyDataType_t>& dataTypes,
                   std::vector<cuphyTrtTensorPrms_t>& tensorPrmsVec) {
    int numTensors = names.size();
    tensorPrmsVec.resize(numTensors);
    for(int i = 0; i < numTensors; i++)  {
        cuphyTrtTensorPrms_t tensorPrms;
        tensorPrms.name = names[i].c_str();
        tensorPrms.nDims = shapes[i].size();
        tensorPrms.dataType = dataTypes[i];
        memcpy(tensorPrms.dims, shapes[i].data(), tensorPrms.nDims * sizeof(int));
        tensorPrmsVec[i] = tensorPrms;
    }
}


} // namespace pycuphy
