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

#ifndef PYCUPHY_TRT_ENGINE_HPP
#define PYCUPHY_TRT_ENGINE_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"

namespace py = pybind11;


namespace pycuphy {

class TrtEngine {
public:
    TrtEngine(
        const std::string& trtModelFile,
        const uint32_t maxBatchSize,
        const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
        const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
        cudaStream_t cuStream);
    ~TrtEngine();

    // Use with fixed batch size which in this case is equal to maximum batch size.
    cuphyStatus_t run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors);

    // Use with dynamic batch size.
    cuphyStatus_t run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors, const uint32_t batchSize);

private:
    cuphyTrtEngineHndl_t m_trtEngineHndl;

    cudaStream_t m_cuStream;
};


void getTensorPrms(const std::vector<std::string>& names,
                   const std::vector<std::vector<int>>& shapes,
                   const std::vector<cuphyDataType_t>& dataTypes,
                   std::vector<cuphyTrtTensorPrms_t>& tensorPrmsVec);


class __attribute__((visibility("default"))) PyTrtEngine {

public:
    PyTrtEngine(
        const std::string& trtModelFile,
        uint32_t maxBatchSize,
        const std::vector<std::string>& inputNames,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<cuphyDataType_t>& inputDataTypes,
        const std::vector<std::string>& outputNames,
        const std::vector<std::vector<int>>& outputShapes,
        const std::vector<cuphyDataType_t>& outputDataTypes,
        uint64_t cuStream);

    // Run using Python inputs/outputs.
    const py::dict& run(const py::dict& inputTensors);

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;
    size_t getBufferSize(uint32_t maxBatchSize, const std::vector<std::vector<int>>& outputShapes) const;

    // Input and output tensors.
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
    std::vector<cuphyTrtTensorPrms_t> m_inputTensorPrms;
    std::vector<void*> m_inputBuffers;
    std::vector<cuphyTrtTensorPrms_t> m_outputTensorPrms;
    std::vector<void*> m_outputBuffers;

    cudaStream_t m_cuStream;

    std::unique_ptr<TrtEngine> m_trtEngine;

    // Python outputs.
    py::dict m_outputTensors;
};


} // pycuphy

#endif // PYCUPHY_TRT_ENGINE_HPP
