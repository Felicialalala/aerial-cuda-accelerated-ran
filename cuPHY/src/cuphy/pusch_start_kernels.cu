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
#include "common_utils.hpp"
#include "ch_est/ch_est.hpp"

#include "pusch_start_kernels.hpp"

namespace
{
__device__ __forceinline__ unsigned long long get_ptimer_ns()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}
}

namespace pusch {

// kernel to wait until required number of symbol bits are ready
template <bool SUB_SLOT>
static __global__ void
symbolWaitKernel(ch_est::puschRxChEstStatDescr_t* pStatDescr, ch_est::puschRxChEstDynDescr_t* pDynDescr)
{
    if(threadIdx.x==0)
    {
        bool     bitsReady  = false;
        bool     timedOut   = false;
        uint64_t startTime, maxWaitTime;
        uint32_t nSymbols;
        if (SUB_SLOT)
        {
            startTime   = get_ptimer_ns();
            maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPreEarlyHarqUs) * 1000;
            nSymbols    = (uint32_t)(pDynDescr->nSymPreSubSlotWaitKernel);
            pDynDescr->mPuschStartTimeNs = startTime; // record the start time to be used in full-slot (i.e. SUB_SLOT == false)
        }
        else
        {
            startTime   = pDynDescr->mPuschStartTimeNs; // note that this is being initialized to 0, but updated if sub-slot is enabled
            // do not wait for full-slot symbols if sub-slot mode is disabled
            if (startTime == 0)
            {
                bitsReady = true;
            }
            else
            {
                maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPostEarlyHarqUs) * 1000;
                nSymbols    = (uint32_t)(pDynDescr->nSymPostSubSlotWaitKernel);
            }
        }

        while (!bitsReady && !timedOut)
        {
            bitsReady = true;
            for(int i = 0; i < nSymbols; i++)
            {
                volatile uint32_t* pstat = (volatile uint32_t*)&(pStatDescr->pSymbolRxStatus[i]);
                if (*pstat != SYM_RX_DONE)
                {
                    bitsReady = false;
                    break;
                }
            }
            timedOut = get_ptimer_ns() > maxWaitTime;   // to ensure the kernel won't get stuck in while loop
        }

        uint8_t* pTimeOutFlag = SUB_SLOT? pDynDescr->pPreSubSlotWaitKernelStatusGpu  : pDynDescr->pPostSubSlotWaitKernelStatusGpu;
        if (pTimeOutFlag)
        {
            *pTimeOutFlag = (timedOut && !bitsReady) ? PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT : PUSCH_RX_WAIT_KERNEL_STATUS_DONE;
        }
    }
}

template <bool ENABLE_DEVICE_GRAPH_LAUNCH>
static __global__ void
deviceGraphLaunchKernel(ch_est::puschRxChEstDynDescr_t* pDynDescr, CUgraphExec deviceGraphExec)
{
    if(ENABLE_DEVICE_GRAPH_LAUNCH && threadIdx.x==0)
    {
        bool timedOut = false;
        if(pDynDescr->pPostSubSlotWaitKernelStatusGpu)
        {
            timedOut = *(pDynDescr->pPostSubSlotWaitKernelStatusGpu) == PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT;
        }
        if(!timedOut)
        {
            cudaGraphLaunch(deviceGraphExec, cudaStreamGraphFireAndForget);
        }
    }
}

template <typename CFG, typename FUNC>
void setKernelParamsImpl(CFG& launchCfg, const bool mode, void* kernelArgs0, void* kernelArgs1, FUNC&& kernelFunc)
{
    // static_assert(std::is_invocable_r_v<CUfunction, FUNC>,
    //     "FUNC must be invocable (Function, lambda etc...), returning CUfunction");
    dim3 gridDims(1);
    dim3 blockDims(32);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc(mode)));}

    // populate kernel parameters
    kernelNodeParamsDriver.blockDimX = blockDims.x;
    kernelNodeParamsDriver.blockDimY = blockDims.y;
    kernelNodeParamsDriver.blockDimZ = blockDims.z;

    kernelNodeParamsDriver.gridDimX = gridDims.x;
    kernelNodeParamsDriver.gridDimY = gridDims.y;
    kernelNodeParamsDriver.gridDimZ = gridDims.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;

    // input placeholder for kernelargs
    void* placeholderPtr      = nullptr;
    launchCfg.kernelArgs[0] = kernelArgs0 == nullptr ? &placeholderPtr : kernelArgs0;
    launchCfg.kernelArgs[1] = kernelArgs1 == nullptr ? &placeholderPtr : kernelArgs1;
    launchCfg.kernelNodeParamsDriver.kernelParams = launchCfg.kernelArgs;
}


// setup for symbolWaitKernel, where it waits for certain number of symbols to arrive before proceeding to start the PUSCH pipeline
void StartKernels::setWaitKernelParams(cuphyPuschRxWaitLaunchCfg_t* pLaunchCfg,
                                       const uint8_t                puschRxProcMode,
                                       void*                        ppStatDescr,
                                       void*                        ppDynDescr)
{
    setKernelParamsImpl(*pLaunchCfg,
                        puschRxProcMode,
                        ppStatDescr,
                        ppDynDescr,
                        [](const auto mode) -> void* {
                            const auto ret = (mode == CUPHY_PUSCH_FULL_SLOT_PATH ?
                                                  symbolWaitKernel<false> :
                                                  symbolWaitKernel<true>);
                            return reinterpret_cast<void*>(ret);
                        });
}

// used in kernel node that launches full-slot or sub-slot PUSCH graph from device
void StartKernels::setDeviceGraphLaunchKernelParams(cuphyPuschRxDglLaunchCfg_t* pLaunchCfg,
                                                    uint8_t                     enableDeviceGraphLaunch,
                                                    void*                       ppDynDescr,
                                                    void*                       ppDeviceGraph)
{
    setKernelParamsImpl(*pLaunchCfg,
                        enableDeviceGraphLaunch,
                        ppDynDescr,
                        ppDeviceGraph,
                        [](const auto mode) -> void* {
                            const auto ret = mode ?
                                                 deviceGraphLaunchKernel<true> :
                                                 deviceGraphLaunchKernel<false>;
                            return reinterpret_cast<void*>(ret);
                        });
}
} // namespace pusch
