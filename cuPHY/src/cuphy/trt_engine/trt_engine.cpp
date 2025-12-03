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
#include <iostream>
#include <fstream>
#include <iterator>
#include <NvInfer.h>
#include "cuphy.h"
#include "cuphy.hpp"
#include "tensor_desc.hpp"
#include "trt_engine.hpp"

namespace {

/**
 * @brief Given maxBatchSize, and I/O tensor parameters, compute the total required buffer size.
 * This will iterate over dims, returns product of the dims, query the storage element size by type,
 * add any needed padding and compute the total size.
 * Non-member function.
 * @param maxBatchSize Max Batch Size
 * @param inputTensorPrms Input Trt Tensor parameters
 * @param outputTensorPrms Output Trt Tensor parameters
 *        name, dataType, number of dims and up to 5 dims
 *        (array of 5 int describing it)
 * @return
 */
size_t getBufferSize(const uint32_t maxBatchSize,
                     const std::vector<trt_engine::TrtParams> &inputTensorPrms,
                     const std::vector<trt_engine::TrtParams> &outputTensorPrms) {

    // Note: Input and output tensor prms are without batch dimension here. Buffer size allocated
    // based on the maximum batch size.

    // Allocate internal buffers.
    constexpr uint32_t EXTRA_PADDING = trt_engine::trtEngine::LINEAR_ALLOC_PAD_BYTES;

    const auto accTensor{[maxBatchSize](const auto& tensorPrms){
        size_t nBytesBuffer = 0;
        for (const auto &tensor: tensorPrms) {
            auto numElems = std::accumulate(tensor.params.dims,
                                            tensor.params.dims + tensor.params.nDims,
                                            1ULL,
                                            std::multiplies<int>{});
            numElems *= maxBatchSize;
            nBytesBuffer += get_cuphy_type_storage_element_size(tensor.params.dataType) * numElems + EXTRA_PADDING;
        }
        return nBytesBuffer;
    }};

    const auto nBytesBuffer = accTensor(inputTensorPrms) + accTensor(outputTensorPrms);
    return nBytesBuffer;
}

/**
 * Max number of dims allowed is N-1 (e.g. 4 )since we are +=1 it in the init()
 * phase. So if we had (e.g.) > 4, +1 means 6 or above but we have dims
 * defined as array of 5: dims[5]
 */
void verifyDims(const std::size_t nDims, const std::size_t N) {
    if (nDims >= N) {
        throw std::out_of_range(fmt::format("Failed dims limit verification: Number of dims {} >= N {}",
                                            nDims, N));
    }
}

std::vector<trt_engine::TrtParams> from(std::vector<cuphyTrtTensorPrms_t> params) {
    std::vector<trt_engine::TrtParams> out;
    out.reserve(params.size());
    std::ignore = std::transform(params.begin(), params.end(),
                                 std::back_inserter(out),
                                 [](auto p) {
                                     // Copy everything in case the caller disappears.
                                     // Name is allocated on a separate std::string
                                     return trt_engine::TrtParams(std::move(p));
                                 });
    return out;
}


// Convert layout in TRT format.
/**
 * @brief Convert Tensor and memcpy to/from device.
 * The function loops over the incoming vector as input and write
 * @param tensorPrms Tensor parameters as dimensions, data type etc...
 * @param stridesDst Strides layout that need to be on the destination tensor.
 * @param stridesSrc Strides layout that exist in the source tensor.
 * @param src vector of the source as inputs.
 * @param dst vector of the destination as outputs.
 * @param cuStream stream to be used when running memcpy from/to device.
 * @return
 */
[[nodiscard]]
auto convertLayout(const std::vector<trt_engine::TrtParams>& tensorPrms,
                   const std::vector<std::vector<int>>& stridesDst,
                   const std::vector<std::vector<int>>& stridesSrc,
                   const std::vector<void *>& src,
                   const std::vector<void *>& dst,
                   cudaStream_t cuStream) {
    const auto numInputs = tensorPrms.size();
    for (int i = 0; i < numInputs; i++) {

        cuphy::tensor_layout srcLayout = cuphy::tensor_layout(tensorPrms[i].params.nDims, tensorPrms[i].params.dims,
                                                              stridesSrc[i].data());
        cuphy::tensor_layout dstLayout = cuphy::tensor_layout(tensorPrms[i].params.nDims, tensorPrms[i].params.dims,
                                                              stridesDst[i].data());

        cuphy::tensor_desc srcDesc = cuphy::tensor_desc(tensorPrms[i].params.dataType, srcLayout,
                                                        cuphy::tensor_flags::align_default);
        cuphy::tensor_desc dstDesc = cuphy::tensor_desc(tensorPrms[i].params.dataType, dstLayout,
                                                        cuphy::tensor_flags::align_default);

        const cuphyStatus_t convertStatus = cuphyConvertTensor(dstDesc.handle(),
                                                               dst[i],
                                                               srcDesc.handle(),
                                                               src[i],
                                                               cuStream);
        if (convertStatus != CUPHY_STATUS_SUCCESS) {
            return convertStatus;
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

} // anonymous namespace

namespace trt_engine {

cuphyStatus_t CaptureStreamPrePostTrtEngEnqueue::preEnqueue(cudaStream_t cuStream) {
    CUDA_CHECK_EXCEPTION(cudaStreamBeginCapture(cuStream, cudaStreamCaptureModeGlobal));
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CaptureStreamPrePostTrtEngEnqueue::postEnqueue(cudaStream_t cuStream) {
    CUDA_CHECK_EXCEPTION(cudaStreamEndCapture(cuStream, &m_pGraph));
    return CUPHY_STATUS_SUCCESS;
}

void trtLogger::log(Severity severity, const char *msg) noexcept {
    // Only log warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

trtEngine::trtEngine(const uint32_t maxBatchSize,
                     std::vector<cuphyTrtTensorPrms_t> inputTensorPrms,
                     std::vector<cuphyTrtTensorPrms_t> outputTensorPrms,
                     std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue,
                     std::unique_ptr<IPrePostEnqueueTensorConversion> prePostEnqueueTensorConversion) :
        m_maxBatchSize(maxBatchSize),
        m_inputTensorPrms(from(std::move(inputTensorPrms))),
        m_outputTensorPrms(from(std::move(outputTensorPrms))),
        m_linearAlloc(getBufferSize(maxBatchSize, m_inputTensorPrms, m_outputTensorPrms))
{
    commonCtorBody(std::move(prePostTrtEngEnqueue), std::move(prePostEnqueueTensorConversion));
}

trtEngine::trtEngine(uint32_t maxBatchSize,
                     std::vector<TrtParams> inputTensorPrms,
                     std::vector<TrtParams> outputTensorPrms,
                     std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue,
                     std::unique_ptr<IPrePostEnqueueTensorConversion> prePostEnqueueTensorConversion) :
        m_maxBatchSize(maxBatchSize),
        m_inputTensorPrms(std::move(inputTensorPrms)),
        m_outputTensorPrms(std::move(outputTensorPrms)),
        m_linearAlloc(getBufferSize(maxBatchSize, m_inputTensorPrms, m_outputTensorPrms))
{
    commonCtorBody(std::move(prePostTrtEngEnqueue), std::move(prePostEnqueueTensorConversion));
}

void trtEngine::commonCtorBody(std::unique_ptr<IPrePostTrtEngEnqueue> prePostTrtEngEnqueue,
                               std::unique_ptr<IPrePostEnqueueTensorConversion> prePostEnqueueTensorConversion) {
    m_prePostTrtEngEnqueue = std::move(prePostTrtEngEnqueue);
    if (!m_prePostTrtEngEnqueue) {
        m_prePostTrtEngEnqueue = std::make_unique<NullPrePostTrtEngEnqueue>();
    }
    m_prePostEnqueueTensorConversion = std::move(prePostEnqueueTensorConversion);
    if(!m_prePostEnqueueTensorConversion) {
        m_prePostEnqueueTensorConversion = std::make_unique<NullPrePostEnqueueTensorConversion>();
    }

    const auto verify{[](const auto& tensorParams){
        for(const auto& tprms : tensorParams) {
            verifyDims(tprms.params.nDims, std::size(tprms.params.dims));
        }
    }};

    verify(m_inputTensorPrms);
    verify(m_outputTensorPrms);
}

cuphyStatus_t trtEngine::init(const char* const trtModelPath) {

    std::ifstream engineFile(trtModelPath, std::ios::binary);
    if (!engineFile)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    std::vector<char> engineData(std::istreambuf_iterator<char>(engineFile), {});
    if (engineData.empty())
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!m_engine) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    // Build the execution context.
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    // Allocate internal input and output buffers.
    // Note: These are for the tensors that actually get fed into TensorRT.
    // They're different from the ones that the user calls trtEngine with.

    const auto compute{[this](const auto numElems, auto& tensorPrms, auto& internalBuf){
        for(int index = 0; index < numElems; index++) {
            auto& tensorPrmsIdx = tensorPrms[index];
            // Add maximum batch size as the first dimension.
            // We make sure in the constructor that we do not have nDims > 4
            // See the `+= 1` below for more description.
            for(int i = tensorPrmsIdx.params.nDims; i > 0; i--) {
                tensorPrmsIdx.params.dims[i] = tensorPrmsIdx.params.dims[i - 1];
            }
            tensorPrmsIdx.params.dims[0] = m_maxBatchSize;
            // Since dims is defined as [5], we cannot have nDims > 4.
            // For example, we cannot have 5. If we do, nDims will be 6
            // but only have dims[5] defined.
            tensorPrmsIdx.params.nDims += 1;

            size_t totalNumElems = 1;
            // FIXME std::accumulate with std::multiply<int>{}
            for(int dim = 0; dim < tensorPrmsIdx.params.nDims; dim++) {
                totalNumElems *= tensorPrmsIdx.params.dims[dim];
            }
            const size_t nBytes = totalNumElems * get_cuphy_type_storage_element_size(tensorPrmsIdx.params.dataType);
            internalBuf[index] = m_linearAlloc.alloc(nBytes);

            const bool status = m_context->setTensorAddress(tensorPrmsIdx.name.c_str(), internalBuf[index]);
            if(!status) {
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
        }
        return CUPHY_STATUS_SUCCESS;
    }};

    const auto numInputs = m_inputTensorPrms.size();
    m_inputStridesTrt.resize(numInputs);
    m_inputStridesCuphy.resize(numInputs);
    m_inputInternalBuf.resize(numInputs);
    if (const auto ret = compute(numInputs, m_inputTensorPrms, m_inputInternalBuf);
        ret != CUPHY_STATUS_SUCCESS ){
        return ret;
    }

    const auto numOutputs = m_outputTensorPrms.size();
    m_outputStridesTrt.resize(numOutputs);
    m_outputStridesCuphy.resize(numOutputs);
    m_outputInternalBuf.resize(numOutputs);
    if (const auto ret = compute(numOutputs, m_outputTensorPrms, m_outputInternalBuf);
            ret != CUPHY_STATUS_SUCCESS ){
        return ret;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t trtEngine::warmup(cudaStream_t cuStream)
{
    const auto numInputs = m_inputTensorPrms.size();
    // Run with maximum batch size.
    for(int inpIndex = 0; inpIndex < numInputs; inpIndex++) {
        m_inputTensorPrms[inpIndex].params.dims[0] = m_maxBatchSize;
        nvinfer1::Dims dims;
        dims.nbDims = m_inputTensorPrms[inpIndex].params.nDims;
        std::copy(m_inputTensorPrms[inpIndex].params.dims, m_inputTensorPrms[inpIndex].params.dims + m_inputTensorPrms[inpIndex].params.nDims, dims.d);
        m_context->setInputShape(m_inputTensorPrms[inpIndex].name.c_str(), dims);
    }

    // Run the network once for warmup
    if (const auto ret = m_prePostTrtEngEnqueue->preEnqueue(cuStream); ret != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    const bool status = m_context->enqueueV3(cuStream);
    if(!status) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if (const auto ret = m_prePostTrtEngEnqueue->postEnqueue(cuStream); ret != CUPHY_STATUS_SUCCESS) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    CUDA_CHECK_EXCEPTION(cudaStreamSynchronize(cuStream));
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t trtEngine::setup(gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
                               gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
                               cudaStream_t                       strm)
{
    const std::vector inputs{pDrvdUeGrpPrmsCpu[0].tInfoDmrsLSEst.pAddr};
    const std::vector outputs{pDrvdUeGrpPrmsCpu[0].tInfoHEst.pAddr};
    CUPHY_CHECK(setup(inputs, outputs, m_maxBatchSize));
    return m_prePostEnqueueTensorConversion->setup(pDrvdUeGrpPrmsCpu, pDrvdUeGrpPrmsGpu,
        m_inputStridesTrt, m_inputInternalBuf,
        m_outputStridesTrt, m_outputInternalBuf, strm);
}

cuphyStatus_t trtEngine::setup(const std::vector<void*>& inputDeviceBuf,
                               const std::vector<void*>& outputDeviceBuf,
                               const uint32_t batchSize)
{
    m_inputBuffers = inputDeviceBuf;
    m_outputBuffers = outputDeviceBuf;

    // Optional value - if not given, use maximum batch size.
    auto currentBatchSize = m_maxBatchSize;
    if(batchSize) {
        currentBatchSize = batchSize;
    }
    
    const auto setupTensors{[currentBatchSize](auto& tensorPrms, auto& stridesTrt, auto& stridesCuphy,
            const auto& func) {
        const auto numElems = tensorPrms.size();
        for(int index = 0; index < numElems; index++) {
            // Input dimensions with batch size.
            auto& tensorPrmsIdx = tensorPrms[index];
            tensorPrmsIdx.params.dims[0] = currentBatchSize;

            const int nDims = tensorPrmsIdx.params.nDims;
            stridesTrt[index].resize(nDims);
            stridesTrt[index].back() = 1;
            for(int i = nDims - 1; i > 0; i--) {
                stridesTrt[index][i - 1] = stridesTrt[index][i] * tensorPrmsIdx.params.dims[i];
            }

            auto& stridesCuphyIdx = stridesCuphy[index];
            stridesCuphyIdx.resize(nDims);
            stridesCuphyIdx[0] = 1;
            for(int i = 1; i < nDims; i++) {
                stridesCuphyIdx[i] = stridesCuphyIdx[i - 1] * tensorPrmsIdx.params.dims[i - 1];
            }
            func(tensorPrms, index);
        }  
    }};

    const auto inputOnlyEveryLoop{[this](const auto& tensorPrms, const std::size_t inpIndex){
        nvinfer1::Dims dims;
        dims.nbDims = tensorPrms[inpIndex].params.nDims;
        std::copy(tensorPrms[inpIndex].params.dims, tensorPrms[inpIndex].params.dims + tensorPrms[inpIndex].params.nDims, dims.d);
        m_context->setInputShape(tensorPrms[inpIndex].name.c_str(), dims);
    }};

    // Setup input and output strides.
    setupTensors(m_inputTensorPrms, m_inputStridesTrt, m_inputStridesCuphy, inputOnlyEveryLoop);
    setupTensors(m_outputTensorPrms, m_outputStridesTrt, m_outputStridesCuphy, []([[maybe_unused]] const auto& tensorPrms,
            [[maybe_unused]] const std::size_t inpIndex){});

    if (!m_context->allInputDimensionsSpecified()) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t
PrePostEnqueueTensorConversion::preEnqueueConvert(const std::vector<TrtParams> &inputTensorPrms,
                                                  const std::vector<std::vector<int>> &inputStridesTrt,
                                                  const std::vector<std::vector<int>> &inputStridesCuphy,
                                                  const std::vector<void *> &inputBuffers,
                                                  const std::vector<void *> &inputInternalBuf,
                                                  cudaStream_t cuStream) {
    return convertLayout(inputTensorPrms, inputStridesTrt,
                         inputStridesCuphy, inputBuffers, inputInternalBuf, cuStream);
}

cuphyStatus_t PrePostEnqueueTensorConversion::postEnqueueConvert(const std::vector<TrtParams> &outputTensorPrms,
                                                                 const std::vector<std::vector<int>> &outputStridesCuphy,
                                                                 const std::vector<std::vector<int>> &outputStridesTrt,
                                                                 const std::vector<void *> &outputInternalBuf,
                                                                 const std::vector<void *> &outputBuffers,
                                                                 cudaStream_t cuStream) {
    return convertLayout(outputTensorPrms, outputStridesCuphy,
                         outputStridesTrt, outputInternalBuf, outputBuffers, cuStream);
}

cuphyStatus_t trtEngine::run(cudaStream_t cuStream) const
{
    if (const auto ret = m_prePostEnqueueTensorConversion->preEnqueueConvert(m_inputTensorPrms,
                                                                             m_inputStridesTrt,
                                                                             m_inputStridesCuphy,
                                                                             m_inputBuffers,
                                                                             m_inputInternalBuf,
                                                                             cuStream);
            ret != CUPHY_STATUS_SUCCESS) {
        return ret;
    }

    if (const bool status = m_context->enqueueV3(cuStream); !status) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(const auto ret = m_prePostEnqueueTensorConversion->postEnqueueConvert(m_outputTensorPrms,
                                                                             m_outputStridesCuphy,
                                                                             m_outputStridesTrt,
                                                                             m_outputInternalBuf,
                                                                             m_outputBuffers,
                                                                             cuStream);
            ret != CUPHY_STATUS_SUCCESS) {
        return ret;
    }

    return CUPHY_STATUS_SUCCESS;
}

} // namespace trt_engine
