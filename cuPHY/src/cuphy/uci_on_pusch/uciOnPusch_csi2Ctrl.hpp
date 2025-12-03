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
#include "tensor_desc.hpp"
#include "cuphy_api.h"


struct cuphyUciOnPuschCsi2Ctrl
{};


struct csi2ToBuffersMap_t 
{
    // Indicies used to look up parameters:
    uint16_t ueIdx;                    
    uint16_t statCellIdx; 

    // Offset for finding users decoded csi1Payload within pUciPayloads:
    uint32_t csi1PayloadByteOffset;    

    // Offset for where to store computed numCsi2Bits within pNumCsi2Bits:
    uint16_t numCsi2BitsOffset;
};



struct uciOnPuschCsi2CtrlDynDescr_t{

    uint16_t nCsi2Ues;

    // debug paramaters:
    uint16_t forcedNumCsi2Bits; // if > 0 kernel assumes all csi2 UCIs have forcedNumCsi2Bits bits

    // option to enable FAPIv3 method for computing CSI-P2 size:
    uint8_t enableCsiP2Fapiv3;

    // Per CSI-P2 parameters:
    csi2ToBuffersMap_t csi2ToBuffersMap[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];

    // UCI payload buffer:
    uint8_t* pUciPayloads;

    // CSI-P2 size buffer:
    uint16_t* pNumCsi2Bits;

    // Parameter buffers:
    cuphyPuschCellStatPrm_t* pPuschCellStatPrms;
    PerTbParams*             pPerTbPrms;
    cuphyPolarCwPrm_t*       pPolCwPrms;
    cuphyPolarUciSegPrm_t*   pPolSegPrms;
    cuphySimplexCwPrm_t*     pSpxCwPrms;
    cuphyRmCwPrm_t*          pRmCwPrms;
};


//  uciOnPuschCsi2Ctrl kernel arguments (supplied via descriptors)
struct uciOnPuschCsi2CtrlKernelArgs_t
{
    uciOnPuschCsi2CtrlDynDescr_t*  pDynDescr;
};

class uciOnPuschCsi2Ctrl : public cuphyUciOnPuschCsi2Ctrl
{
public:
    uciOnPuschCsi2Ctrl();
    ~uciOnPuschCsi2Ctrl()                           = default;
    uciOnPuschCsi2Ctrl(uciOnPuschCsi2Ctrl const&)            = delete;
    uciOnPuschCsi2Ctrl& operator=(uciOnPuschCsi2Ctrl const&) = delete;

    void setup(uint16_t                             nCsi2Ues,                 
               uint16_t*                            pCsi2UeIdxsCpu,
               PerTbParams*                         pTbPrmsCpu,                   
               PerTbParams*                         pTbPrmsGpu,
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu,
               cuphyPuschCellStatPrm_t*             pCellStatPrmsGpu,
               cuphyUciOnPuschOutOffsets_t*         pUciOnPuschOutOffsetsCpu,    
               uint8_t*                             pUciPayloadsGpu,              
               uint16_t*                            pNumCsi2BitsGpu,               
               cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,          
               cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,          
               cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,            
               cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,   
               uint16_t                             forcedNumCsi2Bits,
               uint8_t                              enableCsiP2Fapiv3,
               uciOnPuschCsi2CtrlDynDescr_t*        pCpuDynDesc,
               void*                                pGpuDynDesc,
               bool                                 enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschCsi2CtrlLaunchCfg_t*  pLaunchCfg,
               cudaStream_t                         strm);

    void kernelSelect(uint16_t                            nCsi2Ues,
                      cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    uciOnPuschCsi2CtrlKernelArgs_t m_kernelArgs;

};