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
#include "cuphy.hpp"
#include "cuphy_internal.h"

#include "tensor_desc.hpp"
#include "csirs_rx.hpp"
#include "csirs_rx.cuh"
#include "descrambling.hpp"
#include "descrambling.cuh"

using namespace descrambling;

namespace csirs_rx
{

__global__ void csirsRxChEstKernel(float2** inputArray, float2** chEstArray, CsirsRxParams* csirsRxParams, CsirsRxUeParams* csirsRxUeParams)
{
//    // for debug purpose
//    if((threadIdx.x==0)&&(blockIdx.x==0)&&(blockIdx.y==0))
//    {
//        printf("[%d %d %d][%d %d %d]\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
//    }
//    
//    if((threadIdx.x==0)&&(blockIdx.y==0))
//    {
//        float2* data=inputArray[blockIdx.x];
//        float2* chest = chEstArray[blockIdx.x];
//        printf("UE[%d]Input[%f %f][%f %f]output[%f %f]\n", blockIdx.x, data[0].x, data[0].y, data[1].x, data[1].y, chest[0].x, chest[0].y);
//    }
    //////////////////////////////////////////////////////////
    
    int ueIdx = blockIdx.x;
    int csirsRxIdxinUe = blockIdx.y;
    
    CsirsRxUeParams ueParam = csirsRxUeParams[ueIdx];
    if(csirsRxIdxinUe >= ueParam.nCsirs)
    {
        return;
    }
    
    CsirsRxParams csirsRxParam = csirsRxParams[ueParam.csirsRxParamsStartIdx + csirsRxIdxinUe];
    uint16_t      nRb          =  csirsRxParam.nRb;
    if(threadIdx.x >= nRb)
    {
        return;
    }
    
    float2* dataRx = inputArray[ueIdx];
    float2* chEst  = chEstArray[ueParam.csirsRxChEstBufStartIdx+csirsRxIdxinUe];
    
    uint8_t row              =  csirsRxParam.row;
    CsirsSymbLocRow& rowData =  constRowDataCsirs[row - 1];
    uint8_t numPorts         =  rowData.numPorts;  
    uint8_t nRxAnt           =  ueParam.nRxAnt;    
    uint16_t nRxRe           =  ueParam.nRxRe;                      
    uint8_t lenKBarLBar      =  rowData.lenKBarLBar;                                  
    uint8_t lenKPrime        =  rowData.lenKPrime;                  
    uint8_t lenLPrime        =  rowData.lenLPrime;  
    
    uint16_t startRb         =  csirsRxParam.startRb;                            
    uint8_t  freqDensity     =  csirsRxParam.freqDensity;
    uint8_t  seqIndexCount   =  csirsRxParam.seqIndexCount;
    uint8_t  genEvenRB       =  csirsRxParam.genEvenRB;
    uint8_t  idxSlotInFrame  =  csirsRxParam.idxSlotInFrame;
    uint16_t scrambId        =  csirsRxParam.scrambId;
    float    alpha           =  csirsRxParam.alpha;
    float    rho             =  csirsRxParam.rho;
    float    beta            =  csirsRxParam.beta; 
    
    // get CDM table for Wf/wt value
    uint    cdmTableLoc = (uint8_t)(csirsRxParam.cdmType);
    int8_t* seqTable    = &constSeqTableCsirs[cdmTableLoc][0][0][0];
    
    uint16_t rbIdx = startRb + threadIdx.x;
    bool isEvenRB = (rbIdx & 1) == 0;
    if(rho == 0.5f) // need to write alternative RB
    {
        if((genEvenRB && (!isEvenRB)) || ((!genEvenRB) && isEvenRB))
            return;
    }
    
    float2 xcor[2][4][32][4] = {0.0f, 0.0f};
    
    for (int idxKBarLBar = 0; idxKBarLBar < lenKBarLBar; idxKBarLBar++)
    {
        uint kBar = csirsRxParam.ki[rowData.kIndices[idxKBarLBar]] + rowData.kOffsets[idxKBarLBar];
        uint lBar = csirsRxParam.li[rowData.lIndices[idxKBarLBar]] + rowData.lOffsets[idxKBarLBar];
        
//        if(threadIdx.x==0)
//        {
//            printf("[%d][%d]kBar[%d]lBar[%d][%d][%d]\n", ueIdx, idxKBarLBar, kBar, lBar, lenLPrime, lenKPrime);
//        }
        
        for (int idxLPrime = 0; idxLPrime < lenLPrime; idxLPrime++)
        {
            for (int idxKPrime = 0; idxKPrime < lenKPrime; idxKPrime++)
            {                       
                uint k = kBar + idxKPrime + rbIdx * CUPHY_N_TONES_PER_PRB;
                uint l = lBar + idxLPrime;
                uint mPrime = floorf(rbIdx * alpha) + idxKPrime + floorf(kBar * rho/12.0f);
                
//                if(threadIdx.x==0)
//                {
//                    printf("k[%d]l[%d]mPrime[%d]\n", k,l,mPrime);
//                }
                
                if(beta > 0.0)
                {
                    for (int s = 0; s < seqIndexCount; s++)
                    {
                        int wf = seqTable[s*2*4 + idxKPrime];
                        int wt = seqTable[s*2*4 + 1*4 + idxLPrime];
                        
                        uint32_t c_init_scrambling = ((1 << 10) * (OFDM_SYMBOLS_PER_SLOT * idxSlotInFrame + l + 1) * (2 * scrambId + 1) + scrambId) & 0x7FFFFFFF;
                        uint32_t val = gold32n(c_init_scrambling, 2 * mPrime);
                        float2 a;
                        a.x = (1.0f/beta) * wf * wt  *sqrt(0.5f)*(1.0f-2.0f*(val&0x1));
                        a.y = (1.0f/beta) * wf * wt * sqrt(0.5f)*(1.0f-2.0f*((val>>1)&0x1));
                        
                        uint jj = rowData.cdmGroupIndex[idxKBarLBar];
                        uint p = jj * seqIndexCount + s;
                        
//                        if(threadIdx.x==0)
//                        {
//                            printf("wf[%d]wt[%d]a[%f %f]val[%d]p[%d]\n", wf, wt, a.x, a.y, val, p);
//                        }
                        
                        for(int antIdx = 0; antIdx < nRxAnt; antIdx++)
                        {
                            float2 input = dataRx[k + nRxRe*l + nRxRe*OFDM_SYMBOLS_PER_SLOT*antIdx];
                            float2 a_conj = cuConjf(a);
                            float2 temp;
                            temp.x = input.x*a_conj.x-input.y*a_conj.y;
                            temp.y = input.x*a_conj.y+input.y*a_conj.x;
                            
                            if(row==1)
                            {
                                xcor[idxKPrime][idxLPrime][idxKBarLBar][antIdx] = temp;
                            }
                            else
                            {
                                xcor[idxKPrime][idxLPrime][p][antIdx] = temp;
                            }
                            
//                            if(threadIdx.x==0)
//                            {
//                                printf("input[%f %f]a_conj[%f %f]temp[%f %f][%d][%d][%d][%d]\n", input.x, input.y, a_conj.x, a_conj.y, xcor[idxKPrime][idxLPrime][idxKBarLBar][antIdx].x, xcor[idxKPrime][idxLPrime][p][antIdx].y, idxKPrime, idxLPrime, p, antIdx);
//                            }
                        }
                        
                        
                    }
                } 
            }
        }
    }
    
    uint16_t chEstRbIdx = threadIdx.x;
    if((row==1)&&(freqDensity==3))
    {
        chEstRbIdx = chEstRbIdx*3;
        nRb = nRb*3;
    }
    else if(rho == 0.5f)
    {
        chEstRbIdx= (chEstRbIdx>>1);
        nRb = (nRb>>1);
    }
    
    if(beta > 0.0)
    {
        if(row==1)
        {
            for(int idxKBarLBar=0; idxKBarLBar < 3; idxKBarLBar++)
            {
                for(int antIdx=0; antIdx < nRxAnt; antIdx++)
                {
                    chEst[chEstRbIdx+idxKBarLBar+antIdx*nRb] = xcor[0][0][idxKBarLBar][antIdx];
                }
            }
        }
        else
        {
            for(int idxPort = 0; idxPort < numPorts; idxPort++)
            {
                for(int idxAnt = 0; idxAnt < nRxAnt; idxAnt++)
                {
                    float2 sum;
                    sum.x = 0.0f;
                    sum.y = 0.0f;
                    int count = 0;
                    for (int idxLPrime = 0; idxLPrime < lenLPrime; idxLPrime++)
                    {
                        for (int idxKPrime = 0; idxKPrime < lenKPrime; idxKPrime++)
                        {
                            sum.x = sum.x + xcor[idxKPrime][idxLPrime][idxPort][idxAnt].x;
                            sum.y = sum.y + xcor[idxKPrime][idxLPrime][idxPort][idxAnt].y;
                            count = count + 1;
                            
//                            if((threadIdx.x==0)&&(idxPort==0)&&(idxAnt==0))
//                            {
//                                printf("xorr[%f][%f]\n", xcor[idxKPrime][idxLPrime][idxPort][idxAnt].x, xcor[idxKPrime][idxLPrime][idxPort][idxAnt].y);
//                            }
                        }
                    }
                    sum.x = sum.x/(count*1.0f);
                    sum.y = sum.y/(count*1.0f);
                    chEst[chEstRbIdx+idxPort*nRb+idxAnt*nRb*numPorts] = sum;
                }
            }
        }
    }
}

} // namespace csirs_rx

using namespace csirs_rx;

cuphyStatus_t cuphyCsirsRxKernelSelect(cuphyCsirsRxChEstLaunchCfg_t* pCsirsRxChEstLaunchCfg, int numUes, int maxNumRrcParamPerUe, int maxNumPrbPerUe)
{
    if(!pCsirsRxChEstLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    kernelSelectCsirsRxChEst(pCsirsRxChEstLaunchCfg, numUes, maxNumRrcParamPerUe, maxNumPrbPerUe);
    return CUPHY_STATUS_SUCCESS;
}


void kernelSelectCsirsRxChEst(cuphyCsirsRxChEstLaunchCfg_t* pLaunchCfg, int numUes, int maxNumRrcParamPerUe, int maxNumPrbPerUe)
{
    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(csirsRxChEstKernel);
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}

    // launch geometry
    dim3 blockDim(maxNumPrbPerUe);
    dim3 gridDim(numUes, maxNumRrcParamPerUe);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}
