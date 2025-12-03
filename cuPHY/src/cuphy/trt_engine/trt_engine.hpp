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

#if !defined(TRT_ENGINE_HPP_INCLUDED_)
#define TRT_ENGINE_HPP_INCLUDED_

#include <memory>
#include <string>
#include <vector>
#include <NvInfer.h>
#include "cuphy.h"

#include "cuphy.hpp" // cuphy::linear_alloc

#include "trt_engine_params.hpp"
#include "trt_engine_interfaces.hpp"

struct cuphyTrtEngine
{};


namespace trt_engine {

/**
 * @brief Logger concrete implementation that is needed for nvinfer1::ILogger
 *
 * @class trtLogger: implement log() function.
 */
class trtLogger final : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept final;
};

/**
 * @brief Specific implementation of interface IPrePostTrtEngEnqueue
 *
 * @class CaptureStreamPrePostTrtEngEnqueue: implements stream capture for
 *  interface IPrePostTrtEngEnqueue
 */
class CaptureStreamPrePostTrtEngEnqueue final : public IPrePostTrtEngEnqueue {
public:
    // API

    /**
     * Start Capture/End capture of stream
     * @param cuStream stream to use
     * @return cuphyStatus_t for SUCCESS or any failure
     */
    [[nodiscard]]
    cuphyStatus_t preEnqueue(cudaStream_t cuStream) final;
    [[nodiscard]]
    cuphyStatus_t postEnqueue(cudaStream_t cuStream) final;
    [[nodiscard]] auto* getGraph()const { return m_pGraph; }
private:
    cudaGraph_t m_pGraph{};
};

/**
 * @brief this implementation of interface IPrePostEnqueueTensorConversion calls the vanilla
 * implemenration/logic of converting tensors.
 * There is an internal, anonymous namespace function named convertLayout that will take care
 * of changing the tensors. This includes changing the format and memcpy-ing from/to the device.
 */
class PrePostEnqueueTensorConversion final : public IPrePostEnqueueTensorConversion {
public:
    [[nodiscard]] cuphyStatus_t preEnqueueConvert(const std::vector<TrtParams> &inputTensorPrms,
                                                  const std::vector<std::vector<int>> &inputStridesTrt,
                                                  const std::vector<std::vector<int>> &inputStridesCuphy,
                                                  const std::vector<void *> &inputBuffers,
                                                  const std::vector<void *> &inputInternalBuf,
                                                  cudaStream_t cuStream) final;
    [[nodiscard]] cuphyStatus_t postEnqueueConvert(const std::vector<TrtParams> &outputTensorPrms,
                                                   const std::vector<std::vector<int>> &outputStridesCuphy,
                                                   const std::vector<std::vector<int>> &outputStridesTrt,
                                                   const std::vector<void *> &outputInternalBuf,
                                                   const std::vector<void *> &outputBuffers,
                                                   cudaStream_t cuStream) final;
    [[nodiscard]]
    cuphyStatus_t setup([[maybe_unused]] gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
                        [[maybe_unused]] gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
                        [[maybe_unused]] const std::vector<std::vector<int>> &inputStridesTrt,
                        [[maybe_unused]] const std::vector<void *> &inputInternalBuf,
                        [[maybe_unused]] const std::vector<std::vector<int>> &outputStridesTrt,
                        [[maybe_unused]] const std::vector<void *> &outputInternalBuf,
                        [[maybe_unused]] cudaStream_t                       strm) final { return CUPHY_STATUS_SUCCESS; }
};

// A generic TRT engine class supporting multiple inputs.

/**
 * @brief Implement wrapper over Trt inference engine.
 *
 * @class trtEngine: Wraps functionality of Trt inference engine.
 *  Loads, warmup and run the loaded blob.
 *  Has few interfaces to call pre/post enqueueV3 for graph purposes,
 *  pre/post for convert tensors/copy to/from device
 */
class trtEngine final : public cuphyTrtEngine
{
public:
    /**
     * @brief Make a copy of the I/O tensor parameters.
     * @param maxBatchSize mac batch size to cache
     * @param inputTensorPrms Input to copy to a local member
     * @param outputTensorPrms Output to copy to a local member
     * @param prePostTrtEngEnqueue Is an interception point for pre/post enqueue.
     *        client can inject any operations for pre/post enqueueV3 like stream capture commands.
     * @param prePostRunTensorConversion An interface to be invoked pre/post enqueue calls for the purpose
     *        of tensor conversion and memcpy to/from device.
     * @throws std::out_of_range if number of dims is too big to hold in array.
     */
    trtEngine(uint32_t maxBatchSize,
              std::vector<cuphyTrtTensorPrms_t> inputTensorPrms,
              std::vector<cuphyTrtTensorPrms_t> outputTensorPrms,
              std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue = nullptr,
              std::unique_ptr<IPrePostEnqueueTensorConversion> prePostRunTensorConversion = nullptr);

    /**
     * @brief Make a copy of the I/O tensor parameters.
     * @param maxBatchSize mac batch size to cache
     * @param inputTensorPrms Input to copy to a local member
     * @param outputTensorPrms Output to copy to a local member
     * @param prePostTrtEngEnqueue Is an interception point for pre/post enqueue.
     *        client can inject any operations for pre/post enqueueV3 like stream capture commands.
     * @param prePostRunTensorConversion An interface to be invoked pre/post enqueue calls for the purpose
     *        of tensor conversion and memcpy to/from device.
     * @throws std::out_of_range if number of dims is too big to hold in array.
     */
    trtEngine(uint32_t maxBatchSize,
              std::vector<TrtParams> inputTensorPrms,
              std::vector<TrtParams> outputTensorPrms,
              std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue = nullptr,
              std::unique_ptr<IPrePostEnqueueTensorConversion> prePostRunTensorConversion = nullptr);
    /* Disable move/copy */
    trtEngine(trtEngine&& engine)            = delete;
    trtEngine& operator=(trtEngine&& engine) = delete;

    /**
     * @brief Load the model from the filesystem.
     * Init Trt Engine.
     * Create an instance of
     *  nvinfer1::IRuntime,
     *  nvinfer1::ICudaEngine
     *  nvinfer1::IExecutionContext
     * Allocate m_input and output InternalBuf-fers with m_linearAlloc.alloc()
     * call m_context->setTensorAddress(name, addr);
     * @param trtModelPath Path to the model
     * @return Error or success cuphyStatus_t
     */
    [[nodiscard]]
    cuphyStatus_t init(const char* trtModelPath);

    /**
     * @brief Runs TRT enqueue once to get resources allocated etc.
     * Avoid the first run latency.
     * Calls m_context->setInputShape(name, dims) exactly numInputs.
     * Calls m_context->enqueueV3(stream) + sync stream.
     * @param cuStream stream to use for enqueueV3
     * @return success/error of cuphyStatus_t
     */
    [[nodiscard]]
    cuphyStatus_t warmup(cudaStream_t cuStream);

    /**
     * @brief Setup input/output tensor buffer addresses.
     * Cache the Inputs/Outputs buffers. (vector<void*>).
     * Setup the Strides (vector<vector<int>>)
     *  First dims is the number of inputs, second dimension holds number of dims.
     *  For each slot in the second dim, compute the strides.
     * Once done with Trt strides, it happens with the same logic but instead of starting from the end
     *  like in trt, it will start from the beginning.
     * The logic is happening for both input and output.
     * @param inputBuffers vector of addressees to cache. Used in ::run
     * @param outputBuffers vector of addresses to cache. Used in ::run
     * @param batchSize batch size
     * @return success or error of cuphyStatus_t type.
     */
    [[nodiscard]]
    cuphyStatus_t setup(const std::vector<void*>& inputBuffers,
                        const std::vector<void*>& outputBuffers,
                        uint32_t batchSize = 0);

    // @TODO - to avoid exposing what exactly is passed as I/Os, we could make these 2
    //         as simple template parameters, just forwarding it to the next function.
    //         And providing a default/vanilla impl, that others can specialize.
    [[nodiscard]]
    cuphyStatus_t setup(gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
                        gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
                        cudaStream_t                       strm);

    /**
     * @brief Run inference.
     * Given a stream, invoke ctx::enqueueV3(st)
     * Convert inputs to tensor_layout using tensor params/dims and computed strides - as src
     * Convert inputs to tensor_layout using tensor params/dims and computed strides - as dst
     * Same for tensor descriptor.
     * cuphyConvertTensor - >copy from input buffers (cached in setup from function parameter),
     *  into internal buffers (allocated during init).
     * calls m_context->enqueueV3(cuStream);
     * cuphyConvertTensor - copy from internal buffers (allocated during init)
     *  into output buffers (vec<void*>) (cached in setup from function parameter)
     * @param cuStream Stream to use for enqueueV3
     * @return success/failure in the shape of cuphyStatus_t
     */
    [[nodiscard]]
    cuphyStatus_t run(cudaStream_t cuStream) const;

    // Used for computing total bytes/buffer needed for m_linearAlloc
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;

private:
    /// @brief Common functionality for Constructor body.
    void commonCtorBody(std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue,
                        std::unique_ptr<IPrePostEnqueueTensorConversion> prePostEnqueueTensorConversion);

    uint32_t m_maxBatchSize{};

    // Model inputs and outputs.
    std::vector<TrtParams> m_inputTensorPrms;
    std::vector<TrtParams> m_outputTensorPrms;

    // Internal buffers.
    std::vector<void*> m_inputInternalBuf;
    std::vector<void*> m_outputInternalBuf;

    // Memory layout used by TRT.
    std::vector<std::vector<int>> m_inputStridesTrt;
    std::vector<std::vector<int>> m_outputStridesTrt;

    // Memory layout used by cuPHY.
    std::vector<std::vector<int>> m_inputStridesCuphy;
    std::vector<std::vector<int>> m_outputStridesCuphy;

    // Inputs/outputs to this engine. There is a conversion between these and the internal buffers.
    std::vector<void*> m_inputBuffers;
    std::vector<void*> m_outputBuffers;

    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    // TensorRT components.
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    trtLogger m_logger;
    std::unique_ptr<IPrePostTrtEngEnqueue>        m_prePostTrtEngEnqueue;
    std::unique_ptr<IPrePostEnqueueTensorConversion>  m_prePostEnqueueTensorConversion;
};


}  // namespace trt_engine

#endif // !defined(TRT_ENGINE_HPP_INCLUDED_)
