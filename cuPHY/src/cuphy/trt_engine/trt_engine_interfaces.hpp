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

#ifndef TRT_ENGINE_INTERFACES_HPP
#define TRT_ENGINE_INTERFACES_HPP

#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include "trt_engine_params.hpp"

namespace trt_engine {

/**
 * @brief Define an interface for Pre/Post Trt Engine EnqueueV3
 *
 * @class IPrePostTrtEngEnqueue: for pre and post enqueueV3() in trtEngine
 */
class IPrePostTrtEngEnqueue {
public:
    IPrePostTrtEngEnqueue() = default;
    virtual ~IPrePostTrtEngEnqueue() = default;
    IPrePostTrtEngEnqueue(const IPrePostTrtEngEnqueue& prePostTrtEngEnqueue) = default;
    IPrePostTrtEngEnqueue& operator=(const IPrePostTrtEngEnqueue& prePostTrtEngEnqueue) = default;
    IPrePostTrtEngEnqueue(IPrePostTrtEngEnqueue&& prePostTrtEngEnqueue) = default;
    IPrePostTrtEngEnqueue& operator=(IPrePostTrtEngEnqueue&& prePostTrtEngEnqueue) = default;

    // API

    /**
     * @brief Pre Enqueue activity before calling enqueueV3()
     * @param cuStream stream to use
     * @return cuphyStatus_t SUCCESS or error
     */
    [[nodiscard]]
    virtual cuphyStatus_t preEnqueue(cudaStream_t cuStream) = 0;

    /**
     * @brief Post Enqueue activity after calling enqueueV3()
     * @param cuStream stream to use
     * @return cuphyStatus_t SUCCESS or error
     */
    [[nodiscard]]
    virtual cuphyStatus_t postEnqueue(cudaStream_t cuStream) = 0;
};

/**
 * @brief NullObject for pre/post Trt Engine EnqueueV3.
 *
 * @class NullPrePostTrtEngEnqueue trt_engine.hpp
 *        implements NullObject for the interface IPrePostTrtEngEnqueue.
 * Implementation just return value of CUPHY_STATUS_SUCCESS
 */
class NullPrePostTrtEngEnqueue final : public IPrePostTrtEngEnqueue {
public:
    // API
    [[nodiscard]]
    cuphyStatus_t preEnqueue([[maybe_unused]] cudaStream_t cuStream) final { return CUPHY_STATUS_SUCCESS; }
    [[nodiscard]]
    cuphyStatus_t postEnqueue([[maybe_unused]] cudaStream_t cuStream) final { return CUPHY_STATUS_SUCCESS; }
};

/**
 * @brief Interface to be used for any Tensor Conversions.
 *
 * @class IPrePostEnqueueTensorConversion: A set of pure interfaces that is invoked
 *  pre enqueueV3 and post enqueueV3 calls during run of the Trt model.
 */
class IPrePostEnqueueTensorConversion {
public:
    IPrePostEnqueueTensorConversion() = default;
    virtual ~IPrePostEnqueueTensorConversion() = default;
    IPrePostEnqueueTensorConversion(const IPrePostEnqueueTensorConversion &prePostTrtEngEnqueue) = default;
    IPrePostEnqueueTensorConversion &operator=(const IPrePostEnqueueTensorConversion &prePostTrtEngEnqueue) = default;
    IPrePostEnqueueTensorConversion(IPrePostEnqueueTensorConversion &&prePostTrtEngEnqueue) = default;
    IPrePostEnqueueTensorConversion &operator=(IPrePostEnqueueTensorConversion &&prePostTrtEngEnqueue) = default;

    /**
     * @brief invoked before calling enqueueV3
     * @param inputTensorPrms tensor parameters for inputs
     * @param inputStridesTrt Strides of inputs as required by Trt
     * @param inputStridesCuphy Strides of inputs as require by cuPHY
     * @param inputBuffers Input buffers address on the device
     * @param inputInternalBuf Input buffers allocated for Trt (inputs will be converted and copy to be consumed by trt)
     * @param cuStream stream to operate on
     * @return cuphyStatus_t for success or failure
     */
    [[nodiscard]]
    virtual cuphyStatus_t preEnqueueConvert(const std::vector<TrtParams> &       inputTensorPrms,
                                            const std::vector<std::vector<int>>& inputStridesTrt,
                                            const std::vector<std::vector<int>>& inputStridesCuphy,
                                            const std::vector<void*>&            inputBuffers,
                                            const std::vector<void*>&            inputInternalBuf,
                                            cudaStream_t                         cuStream) = 0;

    /**
     * @brief invoked before after enqueueV3
     * @param outputTensorPrms tensor parameters for outputs
     * @param outputStridesCuphy Strides of output as require by cuPHY
     * @param outputStridesTrt Strides of outputs as required by Trt
     * @param outputInternalBuf Output buffers allocated for Trt
     * @param outputBuffers Output buffers address on the device (Output will be converted and copy to be consumed by cuPHY)
     * @param cuStream stream to operate on
     * @return cuphyStatus_t for success or failure
     */
    [[nodiscard]]
    virtual cuphyStatus_t postEnqueueConvert(const std::vector<TrtParams>&        outputTensorPrms,
                                             const std::vector<std::vector<int>>& outputStridesCuphy,
                                             const std::vector<std::vector<int>>& outputStridesTrt,
                                             const std::vector<void*>&            outputInternalBuf,
                                             const std::vector<void*>&            outputBuffers,
                                             cudaStream_t                         cuStream) = 0;

    /**
     * @brief invoked for every Slot setup.
     * @param pDrvdUeGrpPrmsCpu CPU host pinned allocation to use on host side
     * @param pDrvdUeGrpPrmsGpu GPU device allocation to use in Kernel code
     * @param inputStridesTrt Strides of inputs as required by Trt
     * @param inputInternalBuf Input buffers allocated for Trt (inputs will be converted and copy to be consumed by trt)
     * @param outputStridesTrt Strides of outputs as required by Trt
     * @param outputInternalBuf Output buffers allocated for Trt
     * @param cuStream stream to operate on
     * @return cuphyStatus_t for success or failure
     */
    [[nodiscard]]
    virtual cuphyStatus_t setup(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>   pDrvdUeGrpPrmsCpu,
                                gsl_lite::span<cuphyPuschRxUeGrpPrms_t>   pDrvdUeGrpPrmsGpu,
                                const std::vector<std::vector<int>>& inputStridesTrt,
                                const std::vector<void*>&            inputInternalBuf,
                                const std::vector<std::vector<int>>& outputStridesTrt,
                                const std::vector<void*>&            outputInternalBuf,
                                cudaStream_t                         cuStream) = 0;
};

/**
 * @brief NullObject concreate implementation of interface IPrePostEnqueueTensorConversion
 *
 * @class NullPrePostEnqueueTensorConversion: has no-op implementation of interface
 *  IPrePostEnqueueTensorConversion
 */
class NullPrePostEnqueueTensorConversion final : public IPrePostEnqueueTensorConversion {
public:
    [[nodiscard]]
    cuphyStatus_t preEnqueueConvert([[maybe_unused]] const std::vector<TrtParams> &inputTensorPrms,
                                    [[maybe_unused]] const std::vector<std::vector<int>> &inputStridesTrt,
                                    [[maybe_unused]] const std::vector<std::vector<int>> &inputStridesCuphy,
                                    [[maybe_unused]] const std::vector<void *> &inputBuffers,
                                    [[maybe_unused]] const std::vector<void *> &inputInternalBuf,
                                    [[maybe_unused]] cudaStream_t cuStream) final { return CUPHY_STATUS_SUCCESS; }
    [[nodiscard]]
    cuphyStatus_t postEnqueueConvert([[maybe_unused]] const std::vector<TrtParams> &outputTensorPrms,
                                     [[maybe_unused]] const std::vector<std::vector<int>> &outputStridesCuphy,
                                     [[maybe_unused]] const std::vector<std::vector<int>> &outputStridesTrt,
                                     [[maybe_unused]] const std::vector<void *> &outputInternalBuf,
                                     [[maybe_unused]] const std::vector<void *> &outputBuffers,
                                     [[maybe_unused]] cudaStream_t cuStream) final { return CUPHY_STATUS_SUCCESS; }

    [[nodiscard]]
    cuphyStatus_t setup([[maybe_unused]] gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
                        [[maybe_unused]] gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
                        [[maybe_unused]] const std::vector<std::vector<int>> &inputStridesTrt,
                        [[maybe_unused]] const std::vector<void *> &inputInternalBuf,
                        [[maybe_unused]] const std::vector<std::vector<int>> &outputStridesTrt,
                        [[maybe_unused]] const std::vector<void *> &outputInternalBuf,
                        [[maybe_unused]] cudaStream_t                       strm) final { return CUPHY_STATUS_SUCCESS; }
};

} // namespace trt_engine

#endif //TRT_ENGINE_INTERFACES_HPP
