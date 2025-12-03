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

#include <gsl-lite/gsl-lite.hpp>

#include "tensor_desc.hpp"
#include "type_convert.hpp"
#include "cuphy.hpp"

#include "ch_est_trtengine_pre_post_conversion.hpp"

namespace ch_est {

template <typename TElem>
struct tensor_ref
{
    TElem*     pAddr{};
    const int* strides{};

    [[nodiscard]]
    CUDA_BOTH
    tensor_ref(void* pAddr, const int* pStrides) :
        pAddr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }
    template<std::size_t... Is, typename... Args>
    [[nodiscard]]
    CUDA_BOTH long offset_impl(std::index_sequence<Is...>, Args... args) const {
        return ((strides[Is] * static_cast<long>(args)) + ...);
    }

    /**
     * @brief Variadic template function to calculate the offset.
     * example: return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2) + ...;
     * @tparam Args the integral type denoting the offset in strides_[] array
     * @param args the actual integral instance denoting the offset in strides_[] array
     * @return offset value in the buffer to use in operator() with the pAddr.
     */
    template<typename... Args>
    [[nodiscard]]
    CUDA_BOTH long offset(Args... args) const {
        return offset_impl(std::make_index_sequence<sizeof...(Args)>{}, args...);
    }

    // clang-format off
    template <typename ... Args>
    [[nodiscard]]
    CUDA_BOTH TElem&       operator()(Args ... i0)                          { return const_cast<TElem&>(const_cast<const tensor_ref&>(*this)(i0...)); }

    /**
     * @brief compute the element in the correct offset of the multiple dims
     * starting from address.
     * e.g. *(pAddr + offset(i0, i1, i2, i3));
     * is equivalent to addr[i0][i1][i2][i3]
     * and is invoked with operator(i0, i1, i2, i3)
     * @tparam Args the integral type denoting the offset in strides_[] array
     * @param i0 the actual integral instance denoting the offset in strides_[] array
     * @return the element in the correct offset.
     */
    template <typename ... Args>
    [[nodiscard]]
    CUDA_BOTH const TElem& operator()(Args ... i0) const                    { return *(pAddr + offset(i0 ...)); }
    // clang-format on
};


static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }

template <typename Tdst, typename Tsrc> __global__ void prepareChestMlInputsKernel(ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionDynDescr_t* pDynDescr)
{
    const uint16_t                                                               layerIdx      = threadIdx.x;
    const uint16_t                                                               rxAntIdx      = threadIdx.y;
    const uint16_t                                                               dmrsSymIdx    = threadIdx.z;
    const uint32_t                                                               scIdx         = blockIdx.x;
    const uint32_t                                                               ueGrpIdx      = blockIdx.y;
    ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionDynDescr_t& dynDescr      = *pDynDescr;
    cuphyPuschRxUeGrpPrms_t&                                                     drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    uint16_t nLayers = drvdUeGrpPrms.nLayers;
    if(layerIdx >= nLayers) { return; }

    uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
    if(rxAntIdx >= nRxAnt) { return; }

    uint16_t nSc = drvdUeGrpPrms.nPrb * 6;
    if(scIdx >= nSc) { return; }

    uint16_t dmrsAddlnPos = drvdUeGrpPrms.dmrsAddlnPos;
    if(dmrsSymIdx > dmrsAddlnPos) { return; }

    tensor_ref<const Tsrc> tLsChEst(drvdUeGrpPrms.tInfoDmrsLSEst.pAddr, drvdUeGrpPrms.tInfoDmrsLSEst.strides);
    tensor_ref<Tdst>       tInputInternalBuf(dynDescr.tInputInternalBuf.pAddr, dynDescr.tInputInternalBuf.strides);

    tInputInternalBuf(ueGrpIdx, scIdx, layerIdx, rxAntIdx, dmrsSymIdx, 0) = type_convert<Tdst>(cuReal(tLsChEst(scIdx, layerIdx, rxAntIdx, dmrsSymIdx)));
    tInputInternalBuf(ueGrpIdx, scIdx, layerIdx, rxAntIdx, dmrsSymIdx, 1) = type_convert<Tdst>(cuImag(tLsChEst(scIdx, layerIdx, rxAntIdx, dmrsSymIdx)));
}

template <typename Tdst, typename Tsrc> __global__ void extractChestMlOutputsKernel(ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionDynDescr_t* pDynDescr)
{
    const uint16_t                                                               layerIdx      = threadIdx.x;
    const uint16_t                                                               rxAntIdx      = threadIdx.y;
    const uint16_t                                                               dmrsSymIdx    = threadIdx.z;
    const uint32_t                                                               scIdx         = blockIdx.x;
    const uint32_t                                                               ueGrpIdx      = blockIdx.y;
    ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionDynDescr_t& dynDescr      = *pDynDescr;
    cuphyPuschRxUeGrpPrms_t&                                                     drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    uint16_t nLayers = drvdUeGrpPrms.nLayers;
    if(layerIdx >= nLayers) { return; }

    uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
    if(rxAntIdx >= nRxAnt) { return; }

    uint16_t nSc = drvdUeGrpPrms.nPrb * 12;
    if(scIdx >= nSc) { return; }

    uint16_t dmrsAddlnPos = drvdUeGrpPrms.dmrsAddlnPos;
    if(dmrsSymIdx > dmrsAddlnPos) { return; }

    tensor_ref<Tdst> tInfoHEst(drvdUeGrpPrms.tInfoHEst.pAddr, drvdUeGrpPrms.tInfoHEst.strides);
    tensor_ref<Tsrc> tOutputInternalBuf(dynDescr.tOutputInternalBuf.pAddr, dynDescr.tOutputInternalBuf.strides);

    tInfoHEst(rxAntIdx, layerIdx, scIdx, dmrsSymIdx).x = type_convert<Tsrc>(tOutputInternalBuf(ueGrpIdx, rxAntIdx, layerIdx, scIdx, dmrsSymIdx, 0));
    tInfoHEst(rxAntIdx, layerIdx, scIdx, dmrsSymIdx).y = type_convert<Tsrc>(tOutputInternalBuf(ueGrpIdx, rxAntIdx, layerIdx, scIdx, dmrsSymIdx, 1));
}

cuphyStatus_t ChestPrePostEnqueueTensorConversion::setup(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>   pDrvdUeGrpPrmsCpu,
                                                         gsl_lite::span<cuphyPuschRxUeGrpPrms_t>   pDrvdUeGrpPrmsGpu,
                                                         const std::vector<std::vector<int>>& inputStridesTrt,
                                                         const std::vector<void*>&            inputInternalBuf,
                                                         const std::vector<std::vector<int>>& outputStridesTrt,
                                                         const std::vector<void*>&            outputInternalBuf,
                                                         cudaStream_t                         strm)
{
    puschPrePostTensorConversionDynDescr_t dynDescrCpu{};
    dynDescrCpu.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu.data();
    cuphyTensorInfo6_t& inputInternalBufVal = dynDescrCpu.tInputInternalBuf;
    inputInternalBufVal.pAddr = inputInternalBuf[0];
    inputInternalBufVal.elemType = CUPHY_R_32F;
    inputInternalBufVal.strides[0] = inputStridesTrt[0][0];
    inputInternalBufVal.strides[1] = inputStridesTrt[0][1];
    inputInternalBufVal.strides[2] = inputStridesTrt[0][2];
    inputInternalBufVal.strides[3] = inputStridesTrt[0][3];
    inputInternalBufVal.strides[4] = inputStridesTrt[0][4];
    inputInternalBufVal.strides[5] = inputStridesTrt[0][5];


    cuphyTensorInfo6_t& outputInternalBufVal = dynDescrCpu.tOutputInternalBuf;
    outputInternalBufVal.pAddr = outputInternalBuf[0];
    outputInternalBufVal.elemType = CUPHY_R_32F;
    outputInternalBufVal.strides[0] = outputStridesTrt[0][0];
    outputInternalBufVal.strides[1] = outputStridesTrt[0][1];
    outputInternalBufVal.strides[2] = outputStridesTrt[0][2];
    outputInternalBufVal.strides[3] = outputStridesTrt[0][3];
    outputInternalBufVal.strides[4] = outputStridesTrt[0][4];
    outputInternalBufVal.strides[5] = outputStridesTrt[0][5];


    m_pDynDescrGpu = cuphy::buffer<puschPrePostTensorConversionDynDescr_t, cuphy::device_alloc>(1);
    CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_pDynDescrGpu.addr(), &dynDescrCpu, sizeof(puschPrePostTensorConversionDynDescr_t), cudaMemcpyHostToDevice, strm));
    m_kernelArgs.pDynDescr = m_pDynDescrGpu.addr();

    /////////////////////////////////////////////////////////////////
    // Setup input preparation kernel
    CUDA_KERNEL_NODE_PARAMS& preTensorConversionKernelNodeParamsDriver = m_pPreTensorConversionKernelLaunchCfg.kernelNodeParamsDriver;

    typedef data_type_traits<CUPHY_R_32F>::type r_32f_type_t;
    typedef data_type_traits<CUPHY_C_32F>::type c_32f_type_t;
    void* preTensorConversionKernelFunc = reinterpret_cast<void*>(prepareChestMlInputsKernel<r_32f_type_t, c_32f_type_t>);
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&preTensorConversionKernelNodeParamsDriver.func, preTensorConversionKernelFunc));

    // @todo: Optimize??
    const uint32_t nUeGrps = 1;//pDrvdUeGrpPrmsCpu.size(); //only support ONE UEGRP
    const uint32_t N_BS_ANT = pDrvdUeGrpPrmsCpu[0].nRxAnt;
    const uint32_t N_PUSCH_LAYER = pDrvdUeGrpPrmsCpu[0].nLayers;
    const uint32_t N_DMRS_SYM = (pDrvdUeGrpPrmsCpu[0].dmrsAddlnPos + 1);
    const uint32_t N_PRB = pDrvdUeGrpPrmsCpu[0].nPrb;
    dim3 blockDim(N_PUSCH_LAYER, N_BS_ANT, N_DMRS_SYM);
    dim3 gridDim(N_PRB * CUPHY_N_TONES_PER_PRB/2, nUeGrps);

    preTensorConversionKernelNodeParamsDriver.blockDimX = blockDim.x;
    preTensorConversionKernelNodeParamsDriver.blockDimY = blockDim.y;
    preTensorConversionKernelNodeParamsDriver.blockDimZ = blockDim.z;
    preTensorConversionKernelNodeParamsDriver.gridDimX = gridDim.x;
    preTensorConversionKernelNodeParamsDriver.gridDimY = gridDim.y;
    preTensorConversionKernelNodeParamsDriver.gridDimZ = gridDim.z;
    preTensorConversionKernelNodeParamsDriver.extra = nullptr;
    preTensorConversionKernelNodeParamsDriver.sharedMemBytes = 0;

    m_pPreTensorConversionKernelLaunchCfg.kernelArgs[0] = &m_kernelArgs.pDynDescr;
    preTensorConversionKernelNodeParamsDriver.kernelParams = &(m_pPreTensorConversionKernelLaunchCfg.kernelArgs[0]);

    /////////////////////////////////////////////////////////////////
    // Setup output LLR extraction kernel
    CUDA_KERNEL_NODE_PARAMS& postTensorConversionKernelNodeParamsDriver = m_pPostTensorConversionKernelLaunchCfg.kernelNodeParamsDriver;

    void* postTensorConversionKernelFunc = reinterpret_cast<void*>(extractChestMlOutputsKernel<c_32f_type_t, r_32f_type_t>);
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&postTensorConversionKernelNodeParamsDriver.func, postTensorConversionKernelFunc));

    dim3 postKernelBlockDim(N_PUSCH_LAYER, N_BS_ANT, N_DMRS_SYM);
    dim3 postKernelGridDim(N_PRB * CUPHY_N_TONES_PER_PRB, nUeGrps);

    postTensorConversionKernelNodeParamsDriver.blockDimX = postKernelBlockDim.x;
    postTensorConversionKernelNodeParamsDriver.blockDimY = postKernelBlockDim.y;
    postTensorConversionKernelNodeParamsDriver.blockDimZ = postKernelBlockDim.z;
    postTensorConversionKernelNodeParamsDriver.gridDimX = postKernelGridDim.x;
    postTensorConversionKernelNodeParamsDriver.gridDimY = postKernelGridDim.y;
    postTensorConversionKernelNodeParamsDriver.gridDimZ = postKernelGridDim.z;
    postTensorConversionKernelNodeParamsDriver.extra = nullptr;
    postTensorConversionKernelNodeParamsDriver.sharedMemBytes = 0;

    m_pPostTensorConversionKernelLaunchCfg.kernelArgs[0] = &m_kernelArgs.pDynDescr;
    postTensorConversionKernelNodeParamsDriver.kernelParams = &(m_pPostTensorConversionKernelLaunchCfg.kernelArgs[0]);

    return CUPHY_STATUS_SUCCESS;
}


cuphyStatus_t ChestPrePostEnqueueTensorConversion::preEnqueueConvert([[maybe_unused]] const std::vector<trt_engine::TrtParams>& inputTensorPrms,
                                                                     [[maybe_unused]] const std::vector<std::vector<int>>&      inputStridesTrt,
                                                                     [[maybe_unused]] const std::vector<std::vector<int>>&      inputStridesCuphy,
                                                                     [[maybe_unused]] const std::vector<void*>&                 inputBuffers,
                                                                     [[maybe_unused]] const std::vector<void*>&                 inputInternalBuf,
                                                                     cudaStream_t                                               cuStream)
{
    const CUDA_KERNEL_NODE_PARAMS& preTensorConversionKernelNodeParamsDriver = m_pPreTensorConversionKernelLaunchCfg.kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION(launch_kernel(preTensorConversionKernelNodeParamsDriver, cuStream));
    return CUPHY_STATUS_SUCCESS;
}

[[nodiscard]]
cuphyStatus_t ChestPrePostEnqueueTensorConversion::postEnqueueConvert([[maybe_unused]] const std::vector<trt_engine::TrtParams>& outputTensorPrms,
                                                                      [[maybe_unused]] const std::vector<std::vector<int>>&      outputStridesCuphy,
                                                                      [[maybe_unused]] const std::vector<std::vector<int>>&      outputStridesTrt,
                                                                      [[maybe_unused]] const std::vector<void*>&                 outputInternalBuf,
                                                                      [[maybe_unused]] const std::vector<void*>&                 outputBuffers,
                                                                      cudaStream_t                                               cuStream)
{
    const CUDA_KERNEL_NODE_PARAMS& postTensorConversionKernelNodeParamsDriver = m_pPostTensorConversionKernelLaunchCfg.kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION(launch_kernel(postTensorConversionKernelNodeParamsDriver, cuStream));
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ch_est
