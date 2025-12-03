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
#include "pucch_F0_receiver/pucch_F0_receiver.hpp"
#include "pucch_F1_receiver/pucch_F1_receiver.hpp"
#include "tensor_desc.hpp"


// Implementation of polSegDeRmDeItl interface exposed as an opaque data type to abstract out implementation
// details (polSegDeRmDeItl  C++ class). polSegDeRmDeItl is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyUciOnPuschSegLLRs1
{};

struct uciRvdStride {
    uint16_t rvdStride;
    uint16_t rvdCount;
};

struct uciRvdStrideArray {
    uciRvdStride strideMap[14];
};

struct uciRvdLcmArray {
    uint16_t lcmMap[14];
};

struct uciOnPuschSegLLRs1DynDescr_t
{
    // input buffers
    tensor_ref_any<CUPHY_R_16F>   tEqOutLLRs[MAX_N_USER_GROUPS_SUPPORTED];
    uint16_t                      nPrbs[MAX_N_USER_GROUPS_SUPPORTED];      // each eqOutLLR buffer can have different number of prbs

    // Pre-calculated strides for each TB
    uciRvdStrideArray harqRvdStride[MAX_N_USER_GROUPS_SUPPORTED];
    uciRvdStrideArray harqUciStride[MAX_N_USER_GROUPS_SUPPORTED];
    uciRvdStrideArray csi1RvdStride[MAX_N_USER_GROUPS_SUPPORTED];
    uciRvdStrideArray harqAckStride[MAX_N_USER_GROUPS_SUPPORTED];

    // Pre-calculated stride intersections for each TB
    uciRvdLcmArray    harqRvd_CsiLcms[MAX_N_USER_GROUPS_SUPPORTED];

    // each eqOutLLR buffer has the same time allocation
    uint8_t startSym;
    uint8_t nPuschSym;
    uint8_t nDataSym;
    uint8_t nDmrsSym;
    uint8_t dataSymIdxs[14];
    uint8_t dmrsSymIdxs[14];

    // user parameters
    uint16_t               uciUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];    //ToDo change to uint16_t*
    PerTbParams*           pTbPrms;
};


//  uciOnPuschSegLLRs1 kernel arguments (supplied via descriptors)
struct uciOnPuschSegLLRs1KernelArgs_t
{
    uciOnPuschSegLLRs1DynDescr_t*  pDynDescr;
};

class uciOnPuschSegLLRs1 : public cuphyUciOnPuschSegLLRs1
{
public:
    void setup(cuphyUciOnPuschSegLLRs1Hndl_t        uciOnPuschSegLLRs1Hndl,
               uint16_t                             nUciUes,
               uint16_t*                            pUciUserIdxs,
               PerTbParams*                         pTbPrmsCpu,
               PerTbParams*                         pTbPrmsGpu,
               uint16_t                             nUeGrps,
               cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
               uint16_t*                            pNumPrbs,
               uint8_t                              startSym,
               uint8_t                              nPuschSym,
               uint8_t                              nPuschDataSym,
               uint8_t*                             pDataSymIdxs,
               uint8_t                              nPuschDmrsSym,
               uint8_t*                             pDmrsSymIdxs,
               uciOnPuschSegLLRs1DynDescr_t*        pCpuDynDesc,
               void*                                pGpuDynDesc,
               uint8_t                              enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs1LaunchCfg_t*  pLaunchCfg,
               cudaStream_t                         strm);

    void kernelSelect(uint16_t                            nUciUes,
                      uint16_t*                           pUciUserIdxs,
                      PerTbParams*                        pTbPrmsCpu,
                      uint16_t*                           pNumPrbs,
                      uint8_t                             nPuschDataSym,
                      cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    uciOnPuschSegLLRs1KernelArgs_t m_kernelArgs;
};
