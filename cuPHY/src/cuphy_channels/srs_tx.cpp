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

#include "srs_tx.hpp"
#include "cuphy_internal.h"
#include "utils.cuh"

#define CHECK_CONFIG 1 // runtime check enabled

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyCreateSrsTx(cuphySrsTxHndl_t* pSrsTxHndl, cuphySrsTxStatPrms_t const* pStatPrms)
{
    if((pSrsTxHndl == nullptr) || (pStatPrms == nullptr))
    {
        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "cuphyCreateSrsTx called with cuphySrsTxHndl_t ptr ({:p}) or cuphySrsTxStatPrms_t ptr ({:p}) nullptr.",
                  static_cast<void*>(pSrsTxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        // Use std::nothrow to force return nullptr instead of throwing
        // in case of failed allocation.
        auto* new_pipeline = new (std::nothrow) SrsTx(pStatPrms);
        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pSrsTxHndl = new_pipeline;

        return CUPHY_STATUS_SUCCESS;
    });
}

cuphyStatus_t CUPHYWINAPI cuphySetupSrsTx(cuphySrsTxHndl_t srsTxHndl, cuphySrsTxDynPrms_t* pDynPrms)
{
    if((pDynPrms == nullptr) ||  (srsTxHndl == nullptr))
    {
        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "cuphySetupSrsTx called with nullptr for cuphySrsTxHndl_t ({:p}) or for cuphySrsTxDynPrms_t ptr ({:p}).",
                   static_cast<void*>(srsTxHndl), static_cast<void*>(pDynPrms));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphySetupSrsTx", 1);
        auto* pipeline_ptr  = static_cast<SrsTx*>(srsTxHndl);
        pDynPrms->chan_graph = pipeline_ptr->GetGraph();
        pipeline_ptr->dynamic_params = pDynPrms;
        cuphyStatus_t status = pipeline_ptr->expandParameters(pDynPrms, pDynPrms->cuStream);
        pipeline_ptr->setKernelParams();
        POP_RANGE
        return status;
    }, CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunSrsTx(cuphySrsTxHndl_t srsTxHndl)
{
    if(srsTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "cuphyRunSrsTx error: cuphySrsTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        auto* pipeline_ptr  = static_cast<SrsTx*>(srsTxHndl);
        cuphyStatus_t status = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroySrsTx(cuphySrsTxHndl_t srsTxHndl)
{
    if(srsTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "cuphyDestroySrsTx error: cuphySrsTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    auto* pipeline_ptr  = static_cast<SrsTx*>(srsTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

SrsTx::SrsTx(cuphySrsTxStatPrms_t const* pStatPrms) :
    static_params(pStatPrms),
    m_cudaGraphModeEnabled(true)
{

    memset(&m_genSrsTxLaunchCfg, 0, sizeof(cuphyGenSrsTxLaunchCfg_t));

    pStatPrms->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    m_genSrsTxLaunchCfg.kernelNodeParamsDriver.func = nullptr;

    // Allocate workspace buffers and update relevant host/device pointers
    workspace_offsets[0] = 0; // unchanged
    updateWorkspaceOffsets(pStatPrms->nMaxSrsUes);
    h_workspace = make_unique_pinned<uint8_t>(workspace_offsets[N_SRSTX_COMPONENTS]);
    d_workspace = make_unique_device<uint8_t>(workspace_offsets[N_SRSTX_COMPONENTS], &memory_footprint);
    updateWorkspacePtrs();

    // params pointers unchanged as they point to the beginning of the workspace buffer
    h_params = (SrsTxParams*)(h_workspace.get() + workspace_offsets[0]);
    d_params = (SrsTxParams*)(d_workspace.get() + workspace_offsets[0]);

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        memory_footprint.printMemoryFootprint(this, "SRS TX");
    }
}

void SrsTx::updateWorkspaceOffsets(int nSrsTxParams)
{
    int max_alignment = alignof(__half2*); // for the output tensor addr.
    //workspace_offsets[0] = 0;
    workspace_offsets[1] = workspace_offsets[0] + round_up_to_next<int>(nSrsTxParams * sizeof(SrsTxParams), max_alignment); 
    workspace_offsets[2] = workspace_offsets[1] + round_up_to_next<int>(nSrsTxParams * sizeof(__half2*), max_alignment); 
}

void SrsTx::updateWorkspacePtrs()
{

    h_ue_tensor_addr = (__half2**)(h_workspace.get() + workspace_offsets[1]);
    d_ue_tensor_addr = (__half2**)(d_workspace.get() + workspace_offsets[1]);
}

SrsTx::~SrsTx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
}

cuphyStatus_t SrsTx::checkConfig(SrsTxParams* params, int numSrsTxParams)
{
    return CUPHY_STATUS_SUCCESS;
}

template <fmtlog::LogLevel log_level>
void SrsTx::printSrsTxConfig(const cuphySrsTxStatPrms_t& static_params, const cuphySrsTxDynPrms_t& dyn_params)
{
    NVLOG_FMT(log_level, NVLOG_SRSTX, "SRS TX pipeline parameters: ");
}

cuphyStatus_t SrsTx::expandParameters(cuphySrsTxDynPrms_t* dyn_params, cudaStream_t cuda_strm)
{
    if(dyn_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "expandParameters() got cuphySrsTxDynPrms_t nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    printSrsTxConfig<fmtlog::DBG>(*static_params, *dyn_params);

    updateWorkspaceOffsets(dyn_params->nSrsUes);
    updateWorkspacePtrs();

    for (int j = 0; j < dyn_params->nSrsUes; j++)
    {
        h_ue_tensor_addr[j] = (__half2*)dyn_params->pTDataSrsTx[j].pAddr;

        SrsTxParams* h_param = &(h_params[j]);
        h_param->nAntPorts               = dyn_params->pUeSrsTxPrms[j].nAntPorts;
        h_param->nSyms                   = dyn_params->pUeSrsTxPrms[j].nSyms;
        h_param->nRepetitions            = dyn_params->pUeSrsTxPrms[j].nRepetitions;
        h_param->combSize                = dyn_params->pUeSrsTxPrms[j].combSize;
        h_param->startSym                = dyn_params->pUeSrsTxPrms[j].startSym;
        h_param->sequenceId              = dyn_params->pUeSrsTxPrms[j].sequenceId;
        h_param->configIdx               = dyn_params->pUeSrsTxPrms[j].configIdx;
        h_param->bandwidthIdx            = dyn_params->pUeSrsTxPrms[j].bandwidthIdx;
        h_param->combOffset              = dyn_params->pUeSrsTxPrms[j].combOffset;
        h_param->cyclicShift             = dyn_params->pUeSrsTxPrms[j].cyclicShift;
        h_param->frequencyPosition       = dyn_params->pUeSrsTxPrms[j].frequencyPosition;
        h_param->frequencyShift          = dyn_params->pUeSrsTxPrms[j].frequencyShift;
        h_param->frequencyHopping        = dyn_params->pUeSrsTxPrms[j].frequencyHopping;
        h_param->resourceType            = dyn_params->pUeSrsTxPrms[j].resourceType;
        h_param->Tsrs                    = dyn_params->pUeSrsTxPrms[j].Tsrs;
        h_param->Toffset                 = dyn_params->pUeSrsTxPrms[j].Toffset;
        h_param->groupOrSequenceHopping  = dyn_params->pUeSrsTxPrms[j].groupOrSequenceHopping;
        h_param->idxSlotInFrame          = dyn_params->pUeSrsTxPrms[j].idxSlotInFrame;
        h_param->idxFrame                = dyn_params->pUeSrsTxPrms[j].idxFrame;
        h_param->nSlotsPerFrame          = static_params->nSlotsPerFrame;
        h_param->nSymbsPerSlot           = static_params->nSymbsPerSlot;
    }

//#if CHECK_CONFIG
//    cuphyStatus_t check_config_status = checkConfig(h_params, numParams);
//    if (check_config_status != CUPHY_STATUS_SUCCESS) {
//        return check_config_status;
//    }
//#endif

    CUDA_CHECK(cudaMemcpyAsync(d_workspace.get(), h_workspace.get(), workspace_offsets[N_SRSTX_COMPONENTS], cudaMemcpyHostToDevice, cuda_strm));
    return CUPHY_STATUS_SUCCESS;
}

void SrsTx::createGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    void* arg;
    void* kernelParams[2] = {&arg, &arg}; 
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode2ParamsDriver, 2, &kernelParams[0]));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_genSrsTxNode, m_graph, nullptr, 0, &m_emptyNode2ParamsDriver));
}

void SrsTx::updateGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_SRSTX, cuGraphExecKernelNodeSetParams(m_graphExec, m_genSrsTxNode, &(m_genSrsTxLaunchCfg.kernelNodeParamsDriver)));
}

void SrsTx::setKernelParams()
{
    cuphySrsTxKernelSelect(&m_genSrsTxLaunchCfg, dynamic_params->nSrsUes);
    //---------------------------------------------
    // set kernel args for genScramblingKernel()
    m_genSrsTxArgs[0] = d_params;
    m_genSrsTxArgs[1] = d_ue_tensor_addr;

    m_genSrsTxLaunchCfg.kernelArgs[0] = &m_genSrsTxArgs[0];
    m_genSrsTxLaunchCfg.kernelArgs[1] = &m_genSrsTxArgs[1];

    m_genSrsTxLaunchCfg.kernelNodeParamsDriver.kernelParams = m_genSrsTxLaunchCfg.kernelArgs;
    
    updateGraph();
}

cuphyStatus_t SrsTx::run(const cudaStream_t& cuda_strm) const
{
    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult e = cuGraphLaunch(m_graphExec, cuda_strm);
        if (e != CUDA_SUCCESS) {
            const char* pErrStr;
            CUresult str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "Invalid graph launch for CSI-RS ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

