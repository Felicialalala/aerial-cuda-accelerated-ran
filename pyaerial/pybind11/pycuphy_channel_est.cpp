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

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_channel_est.hpp"
#include "tensor_desc.hpp"

#include "ch_est/chest_factory.hpp"

namespace pycuphy {


void ChannelEstimator::allocateDescr() {

    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphyPuschRxChEstGetDescrInfo(&pStatDescrSizeBytes[CH_EST],
                                                         &pStatDescrAlignBytes[CH_EST],
                                                         &pDynDescrSizeBytes[CH_EST],
                                                         &pDynDescrAlignBytes[CH_EST]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEstGetDescrInfo()");
    }

    for(uint32_t chEstTimeIdx = 1; chEstTimeIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeIdx) {
        pStatDescrSizeBytes[CH_EST + chEstTimeIdx]  = pStatDescrSizeBytes[CH_EST];
        pStatDescrAlignBytes[CH_EST + chEstTimeIdx] = pStatDescrAlignBytes[CH_EST];
        pDynDescrSizeBytes[CH_EST + chEstTimeIdx]   = pDynDescrSizeBytes[CH_EST];
        pDynDescrAlignBytes[CH_EST + chEstTimeIdx]  = pDynDescrAlignBytes[CH_EST];
    }

    // Allocate descriptors (CPU and GPU).
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes);
}


ChannelEstimator::ChannelEstimator(const PuschParams& puschParams, cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_kernelStatDescr("ChEstStatDescr"),
m_kernelDynDescr("ChEstDynDescr"),
m_cuStream(cuStream) {
    init(puschParams);
}


ChannelEstimator::~ChannelEstimator() {
    destroy();
}


void ChannelEstimator::debugDump(H5DebugDump& debugDump, uint16_t numUeGrps, cudaStream_t cuStream) {
    for(int i = 0; i < numUeGrps; i++) {
        debugDump.dump(std::string("LsChEst" + std::to_string(i)), m_tDmrsLSEst[i], cuStream);
        debugDump.dump(std::string("ChEst" + std::to_string(i)), m_tChannelEst[i], cuStream);
    }
}


size_t ChannelEstimator::getBufferSize() const {
    static constexpr uint32_t N_BYTES_R32 = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32 = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t NF = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB;
    static constexpr uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    static constexpr uint32_t MAX_NUM_DMRS_LAYERS = 8;

    size_t nBytesBuffer = 0;

    nBytesBuffer += N_BYTES_C32 * MAX_N_ANTENNAS_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * NF * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + EXTRA_PADDING;
    // m_tDmrsDelayMean
    nBytesBuffer += N_BYTES_R32 * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + EXTRA_PADDING;
    // m_tDmrsAccum
    nBytesBuffer += N_BYTES_C32 * 2 + EXTRA_PADDING;
    // m_tDmrsLSEst
    nBytesBuffer += N_BYTES_C32 * (NF / 2) * MAX_NUM_DMRS_LAYERS * MAX_N_ANTENNAS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST  + EXTRA_PADDING;

    // Debug buffer
    static constexpr uint32_t maxBytesChEstDbg = N_BYTES_C32 * (NF / 2) * MAX_N_DMRSSYMS_SUPPORTED;
    nBytesBuffer += CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * maxBytesChEstDbg + EXTRA_PADDING;

    return nBytesBuffer;
}


void ChannelEstimator::init(const PuschParams& puschParams) {

    // Channel estimation algorithm. RKHS not yet supported by pyAerial.
    cuphyPuschChEstAlgoType_t algoType = puschParams.m_puschStatPrms.chEstAlgo;
    if(algoType == PUSCH_CH_EST_ALGO_TYPE_RKHS) {
        throw std::invalid_argument("RKHS not supported by pyAerial yet.");
    }

    // Same as in pusch_utils.hpp
    const auto nMaxChEstHetCfgs = (algoType == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
        CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS : CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS;

    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDbg.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsLSEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsDelayMean.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsAccum.resize(MAX_N_USER_GROUPS_SUPPORTED);

    // Allocate descriptors.
    allocateDescr();

    // Create the channel estimator object.
    auto statCpuDescrStartAddrs = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs = m_kernelStatDescr.getGpuStartAddrs();
    constexpr bool enableCpuToGpuDescrAsyncCpy = false;
    m_chestKernelBuilder = ch_est::factory::createPuschRxChEstKernelBuilder();
    m_cuphyChEstSettings = std::make_unique<cuphyChEstSettings>(&puschParams.m_puschStatPrms, m_cuStream);
    /**
     * We cannot let span<> just wrap statCpuDescrStartAddrs and GPU as is since
     * these are array<5> and we have only 4 launch configs. So we must limit it when passing it as span.
     */
    auto [puschRxChEst, status] = ch_est::factory::createPuschRxChEst(m_chestKernelBuilder.get(),
                                                   *m_cuphyChEstSettings,
                                                   puschParams.m_puschStatPrms.enableEarlyHarq,
                                                   enableCpuToGpuDescrAsyncCpy,
                                                   gsl_lite::span(statCpuDescrStartAddrs.data(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST),
                                                   gsl_lite::span(statGpuDescrStartAddrs.data(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST),
                                                   m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEst()");
    }

    m_chest = std::move(puschRxChEst);

    if(!enableCpuToGpuDescrAsyncCpy){
        m_kernelStatDescr.asyncCpuToGpuCpy(m_cuStream);
    }
}


void ChannelEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    /**
     * @note @warning:
     *  We cannot leave it as is and let span<> ctor take this array.
     *  Why? Because this is an array<uint8_t*, 5>
     *  Why 5? because N_CH_EST_DESCR_TYPES = CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + 1
     *   - which is 0 + 4 + 1 == 5
     *  However, in ch_est.cu, we are iterating over launchcfs using the size of ppDynDescrsCpu.size()
     *  It used to be hardcoded to CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST but now it takes it from the ppDynDescrsCpu.size()
     *
     *  Second things - we cannot wrap get().data() w/ span as is here when getCpuStartAddrs() is used.
     *  Because it is returning a const But this is not really const since we are casting it in ch_est:
     *   e.g.  puschRxChEstDynDescr_t* pDynDescrVecGpu = reinterpret_cast<puschRxChEstDynDescr_t*>(ppDynDescrsGpu[chEstTimeInstIdx]);
     */
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    const uint16_t nUeGrps = puschParams.getNumUeGrps();
    gsl_lite::span pPuschRxUeGrpPrmsCpu(puschParams.getPuschRxUeGrpPrmsCpuPtr(), nUeGrps);
    gsl_lite::span pPuschRxUeGrpPrmsGpu(puschParams.getPuschRxUeGrpPrmsGpuPtr(), nUeGrps);

    std::vector<cuphy::tensor_device>& tDataRx = puschParams.getDataTensor();

    // Allocate output tensor arrays in device memory.
    for(int i = 0; i < nUeGrps; ++i) {
        int numCh = pPuschRxUeGrpPrmsCpu[i].dmrsAddlnPos + 1;

        uint16_t cellPrmDynIdx = puschParams.m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[i].pCellPrm->cellPrmDynIdx;
        copyTensorData(tDataRx[cellPrmDynIdx], pPuschRxUeGrpPrmsCpu[i].tInfoDataRx);

        m_tChannelEst[i].desc().set(CUPHY_C_32F,
                                    pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                    pPuschRxUeGrpPrmsCpu[i].nLayers,
                                    CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                    numCh,
                                    cuphy::tensor_flags::align_default);
        m_linearAlloc.alloc(m_tChannelEst[i]);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);

        m_tDbg[i].desc().set(CUPHY_C_32F,
                             CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2,
                             pPuschRxUeGrpPrmsCpu[i].nDmrsSyms,
                             1,
                             1,
                             cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDbg[i]);
        copyTensorData(m_tDbg[i], pPuschRxUeGrpPrmsCpu[i].tInfoChEstDbg);

        m_tDmrsDelayMean[i].desc().set(CUPHY_R_32F,
                                       numCh,
                                       cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsDelayMean[i]);
        copyTensorData(m_tDmrsDelayMean[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsDelayMean);

        m_tDmrsLSEst[i].desc().set(CUPHY_C_32F,
                                   CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2,
                                   pPuschRxUeGrpPrmsCpu[i].nLayers,
                                   pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                   numCh,
                                   cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsLSEst[i]);
        copyTensorData(m_tDmrsLSEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsLSEst);

        m_tDmrsAccum[i].desc().set(CUPHY_C_32F, 2, cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsAccum[i]);
        copyTensorData(m_tDmrsAccum[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsAccum);
    }
    m_linearAlloc.memset(0.f, m_cuStream);

    // Run setup.
    const uint8_t enableDftSOfdm = 0;
    const uint8_t chEstAlgo = static_cast<uint8_t>(puschParams.m_puschStatPrms.chEstAlgo);
    const uint16_t waitTimeOutPreEarlyHarqUs = 0;
    const uint16_t waitTimeOutPostEarlyHarqUs = 0;
    const uint8_t enableEarlyHarqProc = 0;
    const uint8_t enableFrontLoadedDmrsProc = 0;
    const uint8_t enableDeviceGraphLaunch = 0;
    bool enableCpuToGpuDescrAsyncCpy = false;
    uint8_t preEarlyHarqWaitKernelStatus_d = 0;
    uint8_t postEarlyHarqWaitKernelStatus_d = 0;
    uint8_t maxDmrsMaxLen = 1;
    cuphyStatus_t chEstSetupStatus = m_chest->setup(m_chestKernelBuilder.get(),
                                                            pPuschRxUeGrpPrmsCpu,
                                                            pPuschRxUeGrpPrmsGpu,
                                                            nUeGrps,
                                                            maxDmrsMaxLen,
                                                            &preEarlyHarqWaitKernelStatus_d,
                                                            &postEarlyHarqWaitKernelStatus_d,
                                                            waitTimeOutPreEarlyHarqUs,
                                                            waitTimeOutPostEarlyHarqUs,
                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                            gsl_lite::span(dynCpuDescrStartAddrs.data(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST),
                                                            gsl_lite::span(dynGpuDescrStartAddrs.data(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST),
                                                            enableEarlyHarqProc,
                                                            enableFrontLoadedDmrsProc,
                                                            enableDeviceGraphLaunch,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            m_cuStream);
    if(chEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(chEstSetupStatus, "pycuphy cuphySetupPuschRxChEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
        puschParams.copyPuschRxUeGrpPrms();
    }

    // Launch kernel using the CUDA driver API.
    m_chest->chestStream().launchKernels(m_cuStream);
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
        m_chest->chestStream().launchSecondaryKernels(m_cuStream);
    }
}


void ChannelEstimator::destroy() {
    // Destroy the PUSCH channel estimation handle.
    m_chest.reset();
    m_chestKernelBuilder.reset();
}



PyChannelEstimator::PyChannelEstimator(const PuschParams& puschParams, uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_chEstimator(puschParams, (cudaStream_t)cuStream) {}


const std::vector<cuda_array_t<std::complex<float>>>& PyChannelEstimator::estimate(PuschParams& puschParams) {

    m_chEst.clear();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();

    // Channel estimation algorithm. RKHS not yet supported by pyAerial.
    cuphyPuschChEstAlgoType_t chEstAlgo = puschParams.m_puschStatPrms.chEstAlgo;
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS) {
        throw std::invalid_argument("RKHS not supported by pyAerial yet.");
    }

    // Run channel estimation.
    m_chEstimator.estimate(puschParams);

    m_chEst.reserve(nUeGrps);

    // Create the return values to Python.
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE || chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
        const std::vector<cuphy::tensor_ref>& channelEst = m_chEstimator.getChEst();

        for(int i = 0; i < nUeGrps; ++i) {

            std::vector shape = {static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nRxAnt),
                                 static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nLayers),
                                 static_cast<size_t>(CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb),
                                 static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nTimeChEsts)};
            m_chEst.push_back(deviceToCudaArray<std::complex<float>>(const_cast<void*>(channelEst[i].addr()), shape));
        }
    }
    else if (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {

        const std::vector<cuphy::tensor_ref>& lsChEst = m_chEstimator.getLsChEst();

        for(int i = 0; i < nUeGrps; ++i) {

            std::vector shape = {static_cast<size_t>(CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2),
                                 static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nLayers),
                                 static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nRxAnt),
                                 static_cast<size_t>(pPuschRxUeGrpPrmsCpu[i].nTimeChEsts)};
            m_chEst.push_back(deviceToCudaArray<std::complex<float>>(const_cast<void*>(lsChEst[i].addr()), shape));
        }
    }
    return m_chEst;
}


} // namespace pycuphy
