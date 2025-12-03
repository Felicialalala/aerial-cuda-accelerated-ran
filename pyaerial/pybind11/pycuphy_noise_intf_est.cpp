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
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "tensor_desc.hpp"

namespace pycuphy {

NoiseIntfEstimator::~NoiseIntfEstimator() {
    destroy();
}


NoiseIntfEstimator::NoiseIntfEstimator(cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_cuStream(cuStream) {

    m_tInfoLwInv.resize(MAX_N_USER_GROUPS_SUPPORTED);

    // Allocate descriptors.
    allocateDescr();

    cuphyStatus_t status = cuphyCreatePuschRxNoiseIntfEst(&m_puschRxNoiseIntfEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxNoiseIntfEst()");
    }
}


size_t NoiseIntfEstimator::getBufferSize() const {
    static constexpr uint32_t N_BYTES_C16 = sizeof(data_type_traits<CUPHY_C_16F>::type);
    static constexpr uint32_t N_BYTES_R16 = sizeof(data_type_traits<CUPHY_R_16F>::type);
    static constexpr uint32_t N_BYTES_R32 = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32 = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t N_BYTES_U32 = 4;
    static constexpr uint32_t MAX_N_UE = MAX_N_TBS_SUPPORTED; // 1 UE per TB for PUSCH

    static constexpr uint32_t NF = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB;
    static constexpr uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    static constexpr uint32_t N_MAX_LAYERS = MAX_N_ANTENNAS_SUPPORTED;

#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    static constexpr uint32_t maxBytesNoiseVarPreEq = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
#else
    static constexpr uint32_t maxBytesNoiseVarPreEq = N_BYTES_R32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
#endif
    static constexpr uint32_t maxBytesNoiseIntfEstInterCtaSyncCnt = N_BYTES_U32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
    static constexpr uint32_t maxBytesNoiseIntfEstLwInv = N_BYTES_C32 * MAX_N_ANTENNAS_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * MAX_N_PRBS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ + (MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES);
    static constexpr size_t nBytesBuffer = ((2*maxBytesNoiseVarPreEq) + maxBytesNoiseIntfEstInterCtaSyncCnt + maxBytesNoiseIntfEstLwInv);
    return nBytesBuffer;
}


void NoiseIntfEstimator::debugDump(H5DebugDump& debugDump, cudaStream_t cuStream) {
    debugDump.dump("PreEqNoiseVar", m_tInfoNoiseVarPreEq, cuStream);
}


void NoiseIntfEstimator::allocateDescr() {

    size_t dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxNoiseIntfEstGetDescrInfo(&m_dynDescrSizeBytes, &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxNoiseIntfEstGetDescrInfo()");
    }

    m_dynDescrSizeBytes = ((m_dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);
}


void NoiseIntfEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.getNumUes();

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    std::vector<cuphy::tensor_device>& tDataRx = puschParams.getDataTensor();

    // Allocate output tensor arrays in device memory.
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    m_tInfoNoiseVarPreEq.desc().set(CUPHY_R_32F,
                                    nUes,
                                    cuphy::tensor_flags::align_tight);
#else
    m_tInfoNoiseVarPreEq.desc().set(CUPHY_R_32F,
                                    nUeGrps,
                                    cuphy::tensor_flags::align_tight);
#endif
    m_linearAlloc.alloc(m_tInfoNoiseVarPreEq);

    m_tInfoNoiseIntfEstInterCtaSyncCnt.desc().set(CUPHY_R_32U,
                                                  nUeGrps,
                                                  cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tInfoNoiseIntfEstInterCtaSyncCnt);

    for(int i = 0; i < nUeGrps; ++i) {
        uint16_t cellPrmDynIdx = puschParams.m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[i].pCellPrm->cellPrmDynIdx;
        copyTensorData(tDataRx[cellPrmDynIdx], pPuschRxUeGrpPrmsCpu[i].tInfoDataRx);

        m_tInfoLwInv[i].desc().set(CUPHY_C_32F,
                                   pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                   pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                   pPuschRxUeGrpPrmsCpu[i].nPrb,
                                   cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tInfoLwInv[i]);
        copyTensorData(m_tInfoLwInv[i], pPuschRxUeGrpPrmsCpu[i].tInfoLwInv);

        copyTensorData(m_tInfoNoiseVarPreEq, pPuschRxUeGrpPrmsCpu[i].tInfoNoiseVarPreEq);
        copyTensorData(m_tInfoNoiseIntfEstInterCtaSyncCnt, pPuschRxUeGrpPrmsCpu[i].tInfoNoiseIntfEstInterCtaSyncCnt);
    }
    m_linearAlloc.memset(0, m_cuStream);

    // Run setup.
    const uint8_t enableDftSOfdm = 0;  // DFT-S-OFDM not supported by pyAerial.
    const uint8_t dmrsSymbolIdx = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t noiseIntfEstLaunchCfgs;
    noiseIntfEstLaunchCfgs.nCfgs = CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS;
    cuphyStatus_t noiseIntfEstSetupStatus = cuphySetupPuschRxNoiseIntfEst(m_puschRxNoiseIntfEstHndl,
                                                                          pPuschRxUeGrpPrmsCpu,
                                                                          pPuschRxUeGrpPrmsGpu,
                                                                          nUeGrps,
                                                                          puschParams.getMaxNumPrb(),
                                                                          enableDftSOfdm,
                                                                          dmrsSymbolIdx,
                                                                          enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                          static_cast<void*>(m_dynDescrBufCpu.addr()),
                                                                          static_cast<void*>(m_dynDescrBufGpu.addr()),
                                                                          &noiseIntfEstLaunchCfgs,
                                                                          m_cuStream);
    if(noiseIntfEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(noiseIntfEstSetupStatus, "cuphySetupPuschRxNoiseIntfEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
        puschParams.copyPuschRxUeGrpPrms();

    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < noiseIntfEstLaunchCfgs.nCfgs; ++hetCfgIdx) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = noiseIntfEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
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


void NoiseIntfEstimator::destroy() {
    // Destroy the PUSCH noise and interference estimation handle.
    cuphyStatus_t status = cuphyDestroyPuschRxNoiseIntfEst(m_puschRxNoiseIntfEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxNoiseIntfEst()");
    }
}


PyNoiseIntfEstimator::PyNoiseIntfEstimator(uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_noiseIntfEstimator((cudaStream_t)cuStream) {
    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
}


const std::vector<cuda_array_t<std::complex<float>>>& PyNoiseIntfEstimator::estimate(const std::vector<cuda_array_t<std::complex<float>>>& chEst,
                                                                                     PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.getNumUes();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();

    m_infoLwInv.clear();

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
    m_noiseIntfEstimator.estimate(puschParams);

    // Get the return values.
    const cuphy::tensor_ref& dInfoNoiseVarPreEq = m_noiseIntfEstimator.getInfoNoiseVarPreEq();
    const std::vector<cuphy::tensor_ref>& dInfoLwInv = m_noiseIntfEstimator.getInfoLwInv();

    // Move the return values to Python.
    m_infoLwInv.reserve(nUeGrps);
    for(int i = 0; i < nUeGrps; ++i) {
        std::vector<size_t> lwInvShape = {pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                          pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                          pPuschRxUeGrpPrmsCpu[i].nPrb};
        m_infoLwInv.push_back(deviceToCudaArray<std::complex<float>>(const_cast<void*>(dInfoLwInv[i].addr()), lwInvShape));
    }
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    std::vector estShape = {static_cast<size_t>(nUes)};
#else
    std::vector estShape = {static_cast<size_t>(nUeGrps)};
#endif
    m_infoNoiseVarPreEq = deviceToCudaArrayPtr<float>(const_cast<void*>(dInfoNoiseVarPreEq.addr()), estShape);

    return m_infoLwInv;
}



} // namespace pycuphy
