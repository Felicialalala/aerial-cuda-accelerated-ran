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

#ifndef CUPHY_CH_EST_SETTINGS_HPP
#define CUPHY_CH_EST_SETTINGS_HPP

#include <cstdint>
#include <optional>

#include "cuphy.h"
#include "cuphy.hpp"

//--------------------------------------------------------------------------------
// cuphyChEstSettings contains receiver ChEst settings

struct cuphyChEstSettings final
{
    uint8_t              nTimeChEsts{};
    const uint32_t*      pSymbolRxStatus{};
    cuphyPuschRkhsPrms_t*  pPuschRkhsPrms{};

    uint8_t enableCfoCorrection{}, enableWeightedAverageCfo{}, enableToEstimation{}, enableDftSOfdm{}, enableTbSizeCheck{}, enableMassiveMIMO{}, enableRssiMeasurement{}, enableSinrMeasurement{}, enablePuschTdi{}, enablePerPrgChEst{};
    cuphyPuschEqCoefAlgoType_t eqCoeffAlgo{};
    cuphyPuschChEstAlgoType_t  chEstAlgo{};
    uint32_t                   nMaxChEstHetCfgs{};
    uint32_t                   nMaxLdpcHetConfigs{};
    std::optional<std::string> puschrxChestFactorySettingsFilename;

    template<std::size_t DIM>
    struct TensorWrapper final {
        TensorWrapper() = default;
        TensorWrapper(const cuphyTensorPrm_t *const prm,
                      cudaStream_t cuStream,
                      cuphyMemoryFootprint* pMemoryFootprint) {
            std::array<int, DIM> dim{};
            std::array<int, DIM> stride{};
            cuphyDataType_t type{};
            CUPHY_CHECK(cuphyGetTensorDescriptor(prm->desc, DIM, &type, nullptr, dim.data(), stride.data()));
            cuphy::tensor_layout layout(DIM, dim.data(), stride.data());
            cuphy::tensor_info info(type, layout);
            tDev = cuphy::tensor_device(info, cuphy::tensor_flags::align_tight, pMemoryFootprint);
            tPrm.desc  = tDev.desc().handle();
            tPrm.pAddr = tDev.addr();
            CUPHY_CHECK(cuphyConvertTensor(tPrm.desc, tPrm.pAddr, prm->desc, prm->pAddr, cuStream));
        }

        // cannot enable copy. Copy of tensor_device means a whole new alloc
        // and copy of payload. This is not use case supported in pusch_rx.
        // We pass settings around and expect it to be the exact same tensor device
        // But we allow move. Move is ok.
        TensorWrapper(const TensorWrapper& tensorWrapper) = delete;
        TensorWrapper& operator=(const TensorWrapper& tensorWrapper) = delete;
        TensorWrapper(TensorWrapper&& tensorWrapper) = default;
        TensorWrapper& operator=(TensorWrapper&& tensorWrapper) = default;

        cuphy::tensor_device tDev;
        cuphyTensorPrm_t tPrm{};
    };

    static constexpr int nDimWFreq = 3, nDimWFreq4 = 3, nDimWFreqSmall = 3, nDimShiftSeq = 2, nDimShiftSeq4 = 2,
            nDimUnShiftSeq = 2, nDimUnShiftSeq4 = 2;
    TensorWrapper<nDimWFreq> WFreq;
    TensorWrapper<nDimWFreq4> WFreq4;
    TensorWrapper<nDimWFreqSmall> WFreqSmall;
    TensorWrapper<nDimShiftSeq> ShiftSeq;
    TensorWrapper<nDimUnShiftSeq> UnShiftSeq;
    TensorWrapper<nDimShiftSeq4> ShiftSeq4;
    TensorWrapper<nDimUnShiftSeq4> UnShiftSeq4;

    cuphyChEstSettings(cuphyPuschStatPrms_t const *pStatPrms, cudaStream_t cuStream,
                       cuphyMemoryFootprint *pMemoryFootprint = nullptr) :
            puschrxChestFactorySettingsFilename(pStatPrms->puschrxChestFactorySettingsFilename ?
                                                std::optional(pStatPrms->puschrxChestFactorySettingsFilename)
                                                                                               : std::nullopt),
            WFreq(pStatPrms->pWFreq, cuStream, pMemoryFootprint),
            WFreq4(pStatPrms->pWFreq4, cuStream, pMemoryFootprint),
            WFreqSmall(pStatPrms->pWFreqSmall, cuStream, pMemoryFootprint),
            ShiftSeq(pStatPrms->pShiftSeq, cuStream, pMemoryFootprint),
            UnShiftSeq(pStatPrms->pUnShiftSeq, cuStream, pMemoryFootprint),
            ShiftSeq4(pStatPrms->pShiftSeq4, cuStream, pMemoryFootprint),
            UnShiftSeq4(pStatPrms->pUnShiftSeq4, cuStream, pMemoryFootprint)
    {
        nTimeChEsts = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
        pSymbolRxStatus = pStatPrms->pSymRxStatus;
        pPuschRkhsPrms  = pStatPrms->pPuschRkhsPrms;

        enableCfoCorrection   = pStatPrms->enableCfoCorrection;
        enableWeightedAverageCfo = pStatPrms->enableCfoCorrection ? pStatPrms->enableWeightedAverageCfo : 0;
        enableToEstimation    = pStatPrms->enableToEstimation;
        enableDftSOfdm        = pStatPrms->enableDftSOfdm;
        enableTbSizeCheck     = pStatPrms->enableTbSizeCheck;
        enableMassiveMIMO     = pStatPrms->enableMassiveMIMO;
        enableRssiMeasurement = pStatPrms->enableRssiMeasurement;
        enableSinrMeasurement = pStatPrms->enableSinrMeasurement;
        enablePuschTdi        = pStatPrms->enablePuschTdi;
        eqCoeffAlgo           = pStatPrms->eqCoeffAlgo;
        chEstAlgo             = pStatPrms->chEstAlgo;
        enablePerPrgChEst     = pStatPrms->enablePerPrgChEst;
        nMaxChEstHetCfgs      = (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
                                CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS : CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS;
        nMaxLdpcHetConfigs    = pStatPrms->nMaxLdpcHetConfigs;
    }
};


#endif //CUPHY_CH_EST_SETTINGS_HPP
