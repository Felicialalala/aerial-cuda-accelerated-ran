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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 27) // "DRV.MPS"

#include "mps.hpp"
#include "context.hpp"
#include "nvlog.hpp"

MpsCtx::MpsCtx(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    int _devSmCount
    ) :
    pdh(_pdh),
    gDev(_gDev),
    devSmCount(_devSmCount),
    isGreenContext(false)
{
    id         = Time::nowNs().count();
    cuCtx = 0;
    gDev->setDevice();

    CU_CHECK_PHYDRIVER(cuDeviceGet(&cuDev, gDev->getId()));

#if CUDART_VERSION >= 11040 // min CUDA version for MPS programmatic API

        if(devSmCount == 0)
            NVLOGC_FMT(TAG, "SM count is 0, will not actually create a new MPS context!");
        else
        {

            int actualDevSmCount = 0;
            CU_CHECK_PHYDRIVER(cuDeviceGetAttribute(&actualDevSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev));
            //Check number of SMs requested is less than the number of SMs on the device and exit before calling cuCtxCreate_v3.
            if (actualDevSmCount < devSmCount)
            {
                std::string err = "Requested " + std::to_string(devSmCount) + " SMs in cuCtxCreate_v3() but GPU has max " + std::to_string(actualDevSmCount) + " SMs.";
                throw std::runtime_error(err);
            }

            // Create CUDA context with SM affinity
            CUexecAffinityParam affinityPrm;
            affinityPrm.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
            affinityPrm.param.smCount.val = devSmCount; //confirmed number of SMs is valid for the device; a 224 error can only mean that MPS service is not running
#if CUDA_VERSION >= 13000
            CUctxCreateParams ctxParams{};
            ctxParams.execAffinityParams = &affinityPrm;
            ctxParams.numExecAffinityParams = 1;
            ctxParams.cigParams = nullptr;
            CU_CHECK_PHYDRIVER(cuCtxCreate(&cuCtx, &ctxParams, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
#else
            CU_CHECK_PHYDRIVER(cuCtxCreate_v3(&cuCtx, &affinityPrm, 1, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
#endif

            // Sanity check, not required!
            CUexecAffinityParam appliedAffinityPrm;
            CUresult result = cuCtxGetExecAffinity(&appliedAffinityPrm, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
            if(CUDA_SUCCESS != result) throw cuphy::cuda_driver_exception(result, "cuCtxGetExecAffinity()");
        }

#else
    if(devSmCount != 0)
    {
        devSmCount = 0;
        CU_CHECK_PHYDRIVER(cuDeviceGetAttribute(&devSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev));
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "GPU ordinal {} gpuId {} SM usage {}", static_cast<int>(cuDev), gDev->getId(), devSmCount);
        CU_CHECK_PHYDRIVER(cuCtxCreate(&cuCtx, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, cuDev));
    }
#endif
}


#if CUDA_VERSION >= 12040
MpsCtx::MpsCtx(
    phydriver_handle _pdh,
    GpuDevice*       _gDev,
    CUdevResource* _resources
    ) :
    pdh(_pdh),
    gDev(_gDev),
    isGreenContext(true)
{
    id         = Time::nowNs().count();
    cuCtx = 0;
    cuGreenCtx = 0;
    gDev->setDevice();

    CU_CHECK_PHYDRIVER(cuDeviceGet(&cuDev, gDev->getId()));

    if(_resources == nullptr)
    {
        NVLOGC_FMT(TAG, "Cannot create a green context with invalid CUdevResource!");
    }
    else
    {
        devSmCount = _resources->sm.smCount;
        const unsigned int num_resources = 1;
        CUresult result = cuDevResourceGenerateDesc(&cuResourceDesc, _resources, num_resources);
        if (result != CUDA_SUCCESS)
        {
            const char* pErrStr;
            cuGetErrorString(result,&pErrStr);
            NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "cuDevResourceGenerateDesc() requesting {} SMs failed with '{}'", _resources->sm.smCount, pErrStr);
        }

        unsigned int default_ctx_creation_flags = CU_GREEN_CTX_DEFAULT_STREAM;

        // Create a green context for this resource descriptor
        CU_CHECK_EXCEPTION(cuGreenCtxCreate(&cuGreenCtx, cuResourceDesc, cuDev, default_ctx_creation_flags));

        // Get primary context from green context
        CU_CHECK_EXCEPTION(cuCtxFromGreenCtx(&cuCtx, cuGreenCtx));
        ctxCreated = true;
    }
}
#endif

MpsCtx::~MpsCtx()
{
    gDev->setDevice();
    if(!isGreenContext && (devSmCount > 0))
    {
        CU_CHECK_PHYDRIVER(cuCtxSynchronize());
        CU_CHECK_PHYDRIVER(cuCtxDestroy(cuCtx));
        ctxDestroyed = true;
    }
#if CUDA_VERSION >= 12040
    if(isGreenContext && ctxCreated && !ctxDestroyed)
    {
        CU_CHECK_PHYDRIVER(cuCtxSynchronize()); //FIXME
        CU_CHECK_PHYDRIVER(cuGreenCtxDestroy(cuGreenCtx));
        ctxDestroyed = true;
    }
#endif
}

uint64_t MpsCtx::getId() const
{
    return id;
}

void MpsCtx::setGpuDevice()
{
    gDev->setDevice();
}

GpuDevice* MpsCtx::getGpuDevice()
{
    return gDev;
}

void MpsCtx::setCtx()
{
    setGpuDevice();
    if(devSmCount > 0)
    {
        CU_CHECK_PHYDRIVER(cuCtxSetCurrent(cuCtx));
        // NVLOGC_FMT(TAG, "Setting MPS TX {} SM", devSmCount);
    }
}

#if CUDA_VERSION >= 12040
void MpsCtx::getResources(CUdevResource* resource) const
{
   if(!isGreenContext)
   {
       NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "Cannot call getResources() on a context that is not a green context.");
       return;
   }
   if(!ctxCreated)
   {
       NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "Cannot call getResources() before a context has been created.");
       return;
   }
   CU_CHECK_PHYDRIVER(cuGreenCtxGetDevResource(cuGreenCtx, resource,  CU_DEV_RESOURCE_TYPE_SM));
}
#endif
