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
#include "cuphy_internal.h"
#include "tensor_desc.hpp"
#include "uciOnPusch_common.cuh"

using namespace uci_on_pusch_common;


// Implementation of polSegDeRmDeItl interface exposed as an opaque data type to abstract out implementation
// details (polSegDeRmDeItl  C++ class). polSegDeRmDeItl is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyUciOnPuschSegLLRs0
{};

struct uciToUserMap_t
{
    uint16_t ueIdx;
    uint16_t ueGrpIdx;
};

struct perUciPrms0_t
{
    uint8_t nSym;  // number of symbols carrying UCI and/or SCH.

    reGrid_t rvdHarqReGrids[MAX_ND_SUPPORTED];
    reGrid_t harqReGrids[MAX_ND_SUPPORTED];
    reGrid_t csi1ReGrids[MAX_ND_SUPPORTED];

    uint32_t descramOffsets[MAX_ND_SUPPORTED];
    bool     dmrsFlags[MAX_ND_SUPPORTED];
    uint32_t schRmBuffOffsets[MAX_ND_SUPPORTED];

    bool    harqPunctFlag;
    uint8_t harqSpx1Flag;
    uint8_t nBitsPerRe;
};


struct uciOnPuschSegLLRs0DynDescr_t{
    cuphyUciToSeg_t uciToSeg;

    perUciPrms0_t perUciPrmsArray[CUPHY_MAX_N_UCI_ON_PUSCH];

    // user indicies
    uciToUserMap_t uciToUserMap[CUPHY_MAX_N_UCI_ON_PUSCH];    //ToDo change to uint16_t*

    // pusch pipeline parameters
    PerTbParams*              pUePrmsGpu;
    cuphyPuschRxUeGrpPrms_t*  pUeGrpPrmsGpu;

    // input buffers
    tensor_ref_any<CUPHY_R_16F>   tEqOutLLRs[MAX_N_USER_GROUPS_SUPPORTED];
};


//  uciOnPuschSegLLRs0 kernel arguments (supplied via descriptors)
struct uciOnPuschSegLLRs0KernelArgs_t
{
    uciOnPuschSegLLRs0DynDescr_t*  pDynDescr;
};

class uciOnPuschSegLLRs0 : public cuphyUciOnPuschSegLLRs0
{
public:
    void setup(uint16_t                             nUciUes,
               uint16_t*                            pUciUeIdxs,
               PerTbParams*                         pTbPrmsCpu,
               PerTbParams*                         pTbPrmsGpu,
               uint16_t                             nUeGrps,
               cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu, 
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsGpu,
               cuphyUciToSeg_t                      uciToSeg,               
               uciOnPuschSegLLRs0DynDescr_t*        pCpuDynDesc,
               void*                                pGpuDynDesc,
               uint8_t                              enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs0LaunchCfg_t*  pLaunchCfg,
               cudaStream_t                         strm);
               
    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    uciOnPuschSegLLRs0KernelArgs_t m_kernelArgs;
};