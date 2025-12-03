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

#include "cuphy.h"
#include "cuphy_api.h"
#include "tensor_desc.hpp"

#pragma once

static constexpr uint8_t  F3_SEG_LLR_UCI_PER_Block = 1; // floor(1024/(CUPHY_PUCCH_F3_MAX_PRBS*12))
static constexpr uint8_t  F3_SEG_LLR_THREAD_PER_UCI = 192; // 12*CUPHY_PUCCH_F3_MAX_PRBS
static constexpr uint16_t F3_SEG_LLR_THREAD_PER_BLOCK = 192; // F3_SEG_LLR_THREAD_PER_UCI * F3_SEG_LLR_UCI_PER_Block
static constexpr uint16_t TEMP_LLR_ARR_MAX_SIZE = 4608; // CUPHY_PUCCH_F3_MAX_PRBS*12*(14-12)[maximum number of data symbols]*2[QPSK]

struct cuphyPucchF3SegLLRs
{};

struct perUciPrms
{
    uint8_t nSym;
    uint8_t nSym_data;
    uint8_t nSym_dmrs;
    uint8_t Qm;

    uint16_t nSymUci;
    uint16_t E_seg1;
    uint16_t E_seg2;
};
typedef struct perUciPrms perUciPrms_t;

struct pucchF3SegLLRsDynDescr
{
    uint16_t numUcis;
    perUciPrms_t perUciPrmsArray[CUPHY_PUCCH_F3_MAX_UCI];
    
    __half* pInLLRaddrs[CUPHY_PUCCH_F3_MAX_UCI];
};
typedef struct pucchF3SegLLRsDynDescr pucchF3SegLLRsDynDescr_t;

struct pucchF3SegLLRsKernelArgs
{
    pucchF3SegLLRsDynDescr_t*  pDynDescr;
};
typedef struct pucchF3SegLLRsKernelArgs pucchF3SegLLRsKernelArgs_t;

class pucchF3SegLLRs : public cuphyPucchF3SegLLRs
{
public:
    pucchF3SegLLRs();
    ~pucchF3SegLLRs() = default;
    pucchF3SegLLRs(pucchF3SegLLRs const&)            = delete;
    pucchF3SegLLRs& operator=(pucchF3SegLLRs const&) = delete;

    void setup(uint16_t                             nF3Ucis,                 
               cuphyPucchUciPrm_t*                  pF3UciPrms,
               __half**                             pDescramLLRaddrs,
               pucchF3SegLLRsDynDescr_t*            pCpuDynDesc,
               void*                                pGpuDynDesc,
               bool                                 enableCpuToGpuDescrAsyncCpy,
               cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
               cudaStream_t                         strm);

    void kernelSelect(uint16_t                           nF3Ucis,
                      cuphyPucchF3SegLLRsLaunchCfg_t*    pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    pucchF3SegLLRsKernelArgs_t m_kernelArgs;
};