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

#include <vector>
#include <memory>

#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "pycuphy_cfo_ta_est.hpp"
#include "tensor_desc.hpp"
#include "cuda_array_interface.hpp"


namespace pycuphy {

CfoTaEstimator::~CfoTaEstimator() {
    destroy();
}


CfoTaEstimator::CfoTaEstimator(cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_cuStream(cuStream) {

    // Allocate descriptors.
    allocateDescr();

    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t status = cuphyCreatePuschRxCfoTaEst(&m_cfoTaEstHndl,
                                                      enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                      static_cast<void*>(m_statDescrBufCpu.addr()),
                                                      static_cast<void*>(m_statDescrBufGpu.addr()),
                                                      m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCfoTaEst()");
    }
}


size_t CfoTaEstimator::getBufferSize() const {

    static constexpr uint32_t N_BYTES_R32 = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32 = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t N_BYTES_PER_UINT32 = 4;
    static constexpr uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    static constexpr uint32_t MAX_N_UE = MAX_N_TBS_SUPPORTED;

    size_t nBytesBuffer = 0;

    static constexpr uint32_t maxBytesCfoEst = N_BYTES_C32 * MAX_ND_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoEst + EXTRA_PADDING;

    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  // CFO Hz
    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  // TA

    static constexpr uint32_t maxBytesCfoPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoPhaseRot + EXTRA_PADDING;

    static constexpr uint32_t maxBytesTaPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesTaPhaseRot + EXTRA_PADDING;

    static constexpr uint32_t maxBytesCfoTaEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoTaEstInterCtaSyncCnt + EXTRA_PADDING;

    return nBytesBuffer;
}


void CfoTaEstimator::allocateDescr() {

    size_t statDescrAlignBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxCfoTaEstGetDescrInfo(&m_statDescrSizeBytes,
                                                                            &statDescrAlignBytes,
                                                                            &m_dynDescrSizeBytes,
                                                                            &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxCfoTaEstGetDescrInfo()");
    }

    m_dynDescrSizeBytes = ((m_dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);

    m_statDescrSizeBytes = ((m_statDescrSizeBytes + (statDescrAlignBytes - 1)) / statDescrAlignBytes) * statDescrAlignBytes;
    m_statDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_statDescrSizeBytes);
    m_statDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_statDescrSizeBytes);
}


void CfoTaEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    m_tCfoEstVec.resize(nUeGrps);

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    // Allocate output tensor arrays in device memory.
    m_tCfoPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoPhaseRot);

    m_tTaPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tTaPhaseRot);

    m_tCfoHz.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoHz);

    m_tTaEst.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tTaEst);

    m_tCfoTaEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoTaEstInterCtaSyncCnt);

    for(int i = 0; i < nUeGrps; ++i) {
        copyTensorData(m_tCfoPhaseRot, pPuschRxUeGrpPrmsCpu[i].tInfoCfoPhaseRot);
        copyTensorData(m_tTaPhaseRot, pPuschRxUeGrpPrmsCpu[i].tInfoTaPhaseRot);
        copyTensorData(m_tCfoHz, pPuschRxUeGrpPrmsCpu[i].tInfoCfoHz);
        copyTensorData(m_tTaEst, pPuschRxUeGrpPrmsCpu[i].tInfoTaEst);
        copyTensorData(m_tCfoTaEstInterCtaSyncCnt, pPuschRxUeGrpPrmsCpu[i].tInfoCfoTaEstInterCtaSyncCnt);

        m_tCfoEstVec[i].desc().set(CUPHY_C_32F, MAX_ND_SUPPORTED, pPuschRxUeGrpPrmsCpu[i].nUes, cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tCfoEstVec[i]);
        copyTensorData(m_tCfoEstVec[i], pPuschRxUeGrpPrmsCpu[i].tInfoCfoEst);
    }
    m_linearAlloc.memset(0, m_cuStream);

    // Run setup.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyPuschRxCfoTaEstLaunchCfgs_t m_cfoTaEstLaunchCfgs;
    m_cfoTaEstLaunchCfgs.nCfgs = 0;  // Setup within the component.
    cuphy::buffer<float*, cuphy::pinned_alloc>  bFoCompensationBufferPtrs = std::move(cuphy::buffer<float*, cuphy::pinned_alloc>(1));
    cuphyStatus_t cfoTaEstSetupStatus = cuphySetupPuschRxCfoTaEst(m_cfoTaEstHndl,
                                                                  pPuschRxUeGrpPrmsCpu,
                                                                  pPuschRxUeGrpPrmsGpu,
                                                                  bFoCompensationBufferPtrs.addr(), // pFoCompensationBuffers
                                                                  nUeGrps,
                                                                  puschParams.getMaxNumPrb(),
                                                                  0,  // pDbg
                                                                  enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                  static_cast<void*>(m_dynDescrBufCpu.addr()),
                                                                  static_cast<void*>(m_dynDescrBufGpu.addr()),
                                                                  &m_cfoTaEstLaunchCfgs,
                                                                  m_cuStream);
    if(cfoTaEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(cfoTaEstSetupStatus, "cuphySetupPuschRxCfoTaEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
        puschParams.copyPuschRxUeGrpPrms();
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_cfoTaEstLaunchCfgs.nCfgs; ++hetCfgIdx) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_cfoTaEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                                         kernelNodeParamsDriver.gridDimX,
                                                         kernelNodeParamsDriver.gridDimY,
                                                         kernelNodeParamsDriver.gridDimZ,
                                                         kernelNodeParamsDriver.blockDimX,
                                                         kernelNodeParamsDriver.blockDimY,
                                                         kernelNodeParamsDriver.blockDimZ,
                                                         kernelNodeParamsDriver.sharedMemBytes,
                                                         static_cast<CUstream>(m_cuStream),
                                                         kernelNodeParamsDriver.kernelParams,
                                                         kernelNodeParamsDriver.extra));
    }
}


void CfoTaEstimator::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxCfoTaEst(m_cfoTaEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxCfoTaEst()");
    }
}


PyCfoTaEstimator::PyCfoTaEstimator(uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_cfoTaEstimator((cudaStream_t)cuStream) {
    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
}


const std::vector<cuda_array_t<std::complex<float>>>& PyCfoTaEstimator::estimate(const std::vector<cuda_array_t<std::complex<float>>>& chEst,
                                                                                 PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    m_cfoEstVec.clear();

    // Read inputs.
    for(int i = 0; i < nUeGrps; ++i) {
        m_tChannelEst[i] = deviceFromCudaArray<std::complex<float>>(
            chEst[i],
            nullptr,
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);
    }

    // Run the estimator.
    m_cfoTaEstimator.estimate(puschParams);

    // Get the return values.
    const std::vector<cuphy::tensor_ref>& dCfoEst = m_cfoTaEstimator.getCfoEst();
    const cuphy::tensor_ref& dCfoHz = m_cfoTaEstimator.getCfoHz();
    const cuphy::tensor_ref& dTaEst = m_cfoTaEstimator.getTaEst();
    const cuphy::tensor_ref& dCfoPhaseRot = m_cfoTaEstimator.getCfoPhaseRot();
    const cuphy::tensor_ref& dTaPhaseRot = m_cfoTaEstimator.getTaPhaseRot();

    // Move the return values to Python.
    m_cfoEstVec.reserve(nUeGrps);
    for(int i = 0; i < nUeGrps; ++i) {
        std::vector<size_t> cfoEstVecShape = {MAX_ND_SUPPORTED, pPuschRxUeGrpPrmsCpu[i].nUes};
        m_cfoEstVec.push_back(deviceToCudaArray<std::complex<float>>(const_cast<void*>(dCfoEst[i].addr()), cfoEstVecShape));
    }

    std::vector estShape = {static_cast<size_t>(nUes)};
    m_cfoEstHz = deviceToCudaArrayPtr<float>(const_cast<void*>(dCfoHz.addr()), estShape);
    m_taEst = deviceToCudaArrayPtr<float>(const_cast<void*>(dTaEst.addr()), estShape);

    std::vector cfoPhaseRotShape = {static_cast<size_t>(CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST),
                                    static_cast<size_t>(CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP),
                                    static_cast<size_t>(nUeGrps)};
    m_cfoPhaseRot = deviceToCudaArrayPtr<std::complex<float>>(const_cast<void*>(dCfoPhaseRot.addr()), cfoPhaseRotShape);

    std::vector taPhaseRotShape = {static_cast<size_t>(CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP),
                                   static_cast<size_t>(nUeGrps)};
    m_taPhaseRot = deviceToCudaArrayPtr<std::complex<float>>(const_cast<void*>(dTaPhaseRot.addr()), taPhaseRotShape);

    return m_cfoEstVec;
}



} // namespace pycuphy
