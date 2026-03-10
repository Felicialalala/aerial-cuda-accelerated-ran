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

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include "trtEngine.h"

namespace cumac_ml {
namespace {

bool ensureTrtPluginsInitialized(nvinfer1::ILogger& logger)
{
    static bool initialized = false;
    static bool initOk = false;
    if (!initialized) {
        initOk = initLibNvInferPlugins(&logger, "");
        initialized = true;
    }
    return initOk;
}

} // namespace

void trtLogger::log(Severity severity, const char *msg) noexcept {
    // Only log warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}


trtEngine::trtEngine(const char* modelPath,
                     const bool parseFromOnnx,
                     const uint32_t maxBatchSize,
                     const std::vector<trtTensorPrms_t>& inputTensorPrms,
                     const std::vector<trtTensorPrms_t>& outputTensorPrms):
m_maxBatchSize(maxBatchSize),
m_inputTensorPrms(inputTensorPrms),
m_outputTensorPrms(outputTensorPrms),
m_numInputs(inputTensorPrms.size()),
m_numOutputs(outputTensorPrms.size())
{
    if (!ensureTrtPluginsInitialized(m_logger)) {
        throw std::runtime_error("Failed to initialize TensorRT plugin registry");
    }
    if(parseFromOnnx)
        buildFromOnnx(modelPath);
    else
        buildFromTrt(modelPath);
}


void trtEngine::buildFromTrt(const char* trtModelPath)
{
    std::ifstream engineFile;
    try {
        engineFile.open(trtModelPath, std::ios::binary);
        engineFile.exceptions(std::ifstream::failbit);
    }
    catch(std::ifstream::failure e) {
        throw std::runtime_error(std::string("Model file not found: ") + trtModelPath);
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    if (fsize <= 0) {
        throw std::runtime_error(std::string("Empty TRT engine file: ") + trtModelPath);
    }
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (m_engine == nullptr) {
        throw std::runtime_error(std::string("Failed to deserialize TensorRT engine: ") + trtModelPath);
    }
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (m_context == nullptr) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }
}


void trtEngine::buildFromOnnx(const char* onnxModelPath)
{
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (builder == nullptr) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }

    // TensorRT 10 deprecates kEXPLICIT_BATCH; create network with default flags.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (network == nullptr) {
        throw std::runtime_error("Failed to create TensorRT network");
    }

    // Create a builder config.
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (config == nullptr) {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }

    // Add two optimization profiles.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (profile == nullptr) {
        throw std::runtime_error("Failed to create TensorRT optimization profile");
    }
    for(const auto& inputTensorPrm : m_inputTensorPrms) {

        nvinfer1::Dims dims;
        toNvInferDims(inputTensorPrm.dims, dims);

        bool ok = profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims);
        if (!ok) {
            throw std::runtime_error(std::string("Failed to set OPT dims for input tensor: ") + inputTensorPrm.name);
        }

        dims.d[0] = 1;
        ok = profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims);
        if (!ok) {
            throw std::runtime_error(std::string("Failed to set MIN dims for input tensor: ") + inputTensorPrm.name);
        }

        dims.d[0] = m_maxBatchSize;
        ok = profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims);
        if (!ok) {
            throw std::runtime_error(std::string("Failed to set MAX dims for input tensor: ") + inputTensorPrm.name);
        }
    }
    config->addOptimizationProfile(profile);

    // Create a parser for reading the ONNX file and parse it.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (parser == nullptr) {
        throw std::runtime_error("Failed to create TensorRT ONNX parser");
    }
    const bool parsed = parser->parseFromFile(onnxModelPath, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        std::string errMsg = std::string("Failed to parse ONNX: ") + onnxModelPath;
        const int numErr = parser->getNbErrors();
        for (int i = 0; i < numErr; ++i) {
            const auto* err = parser->getError(i);
            if (err != nullptr) {
                errMsg += "\n  - ";
                errMsg += err->desc();
            }
        }
        throw std::runtime_error(errMsg);
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (plan == nullptr) {
        throw std::runtime_error(std::string("Failed to build TensorRT engine from ONNX: ") + onnxModelPath);
    }
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (m_engine == nullptr) {
        throw std::runtime_error(std::string("Failed to deserialize TensorRT engine built from ONNX: ") + onnxModelPath);
    }
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (m_context == nullptr) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }
}


bool trtEngine::setup(const std::vector<void*>& inputDeviceBuf,
                      const std::vector<void*>& outputDeviceBuf,
                      const uint32_t batchSize)
{
    if (m_context == nullptr) {
        std::cerr << "TensorRT context is not initialized." << std::endl;
        return false;
    }
    if (static_cast<int>(inputDeviceBuf.size()) != m_numInputs ||
        static_cast<int>(outputDeviceBuf.size()) != m_numOutputs) {
        std::cerr << "TensorRT setup input/output buffer count mismatch." << std::endl;
        return false;
    }

    // Optional value - if not given, use maximum batch size.
    uint32_t currentBatchSize = m_maxBatchSize;
    if(batchSize) {
        currentBatchSize = batchSize;
    }

    bool status;

    // Set correct batch size everywhere.
    for(auto& inputTensorPrm : m_inputTensorPrms) {
        inputTensorPrm.dims[0] = currentBatchSize;

        nvinfer1::Dims dims;
        toNvInferDims(inputTensorPrm.dims, dims);
        status = m_context->setInputShape(inputTensorPrm.name.c_str(), dims);
        if(!status) {
            std::cerr << "Failed to set input tensor shape for tensor " << inputTensorPrm.name << "!" << std::endl;
            return false;
        }
    }

    for(auto& outputTensorPrm : m_outputTensorPrms) {
        outputTensorPrm.dims[0] = currentBatchSize;
    }

    // Set input and output tensor addresses.
    for(int i = 0; i < m_numInputs; i++) {
        std::string inputName = m_inputTensorPrms[i].name;
        status = m_context->setTensorAddress(inputName.c_str(), inputDeviceBuf[i]);
        if(!status) {
            std::cerr << "Failed to set input tensor address for tensor " << inputName << "!" << std::endl;
            return false;
        }
    }
    for(int i = 0; i < m_numOutputs; i++) {
        std::string outputName = m_outputTensorPrms[i].name;
        status = m_context->setTensorAddress(outputName.c_str(), outputDeviceBuf[i]);
        if(!status) {
            std::cerr << "Failed to set output tensor address for tensor " << outputName << "!" << std::endl;
            return false;
        }
    }

    if (!m_context->allInputDimensionsSpecified()) {
        return false;
    }

    return true;
}


void trtEngine::toNvInferDims(const std::vector<int>& shape, nvinfer1::Dims& dims)
{
    dims.nbDims = shape.size();
    std::copy(shape.begin(), shape.end(), dims.d);
}


bool trtEngine::run(cudaStream_t cuStream)
{
    if (m_context == nullptr) {
        return false;
    }
    return m_context->enqueueV3(cuStream);
}

} // namespace cumac_ml
