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

#include "csirs_rx.hpp"
#include "cuphy_internal.h"
#include "utils.cuh"

#define CHECK_CONFIG 1 // runtime check enabled

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyCreateCsirsRx(cuphyCsirsRxHndl_t* pCsirsRxHndl, cuphyCsirsStatPrms_t const* pStatPrms)
{
    if((pCsirsRxHndl == nullptr) || (pStatPrms == nullptr))
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyCreateCsirsRx called with cuphyCsirsRxHndl_t ptr ({:p}) or cuphyCsirsStatPrms_t ptr ({:p}) nullptr.",
                  static_cast<void*>(pCsirsRxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        CsirsRx* new_pipeline = new CsirsRx(pStatPrms);
        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pCsirsRxHndl = new_pipeline;

        return CUPHY_STATUS_SUCCESS;
    });
}

const void* CsirsRx::getMemoryTracker()
{
    return &memory_footprint;
}

cuphyStatus_t CUPHYWINAPI cuphySetupCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl, cuphyCsirsRxDynPrms_t* pDynPrms)
{
    if((pDynPrms == nullptr) ||  (csirsRxHndl == nullptr))
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphySetupCsirsRx called with nullptr for cuphyCsirsRxHndl_t ({:p}) or for cuphyCsirsRxDynPrms_t ptr ({:p}).",
                   static_cast<void*>(csirsRxHndl), static_cast<void*>(pDynPrms));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PUSH_RANGE("cuphySetupCsirsRx", 1);
        CsirsRx* pipeline_ptr  = static_cast<CsirsRx*>(csirsRxHndl);
        pDynPrms->chan_graph = pipeline_ptr->GetGraph();
        pipeline_ptr->dynamic_params = pDynPrms;
        cuphyStatus_t status = pipeline_ptr->expandParameters(pDynPrms, pDynPrms->cuStream);
        pipeline_ptr->setKernelParams();

        POP_RANGE
        return status;
    }, CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl)
{
    if(csirsRxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyRunCsirsRx error: cuphyCsirsRxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        CsirsRx* pipeline_ptr  = static_cast<CsirsRx*>(csirsRxHndl);
        cuphyStatus_t status = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        return status;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyCsirsRx(cuphyCsirsRxHndl_t csirsRxHndl)
{
    if(csirsRxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "cuphyDestroyCsirsRx error: cuphyCsirsRxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    CsirsRx* pipeline_ptr  = static_cast<CsirsRx*>(csirsRxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}

CsirsRx::CsirsRx(cuphyCsirsStatPrms_t const* pStatPrms) :
    dynamic_params(nullptr),
    static_params(pStatPrms),
    m_chEstArgs{0},
    m_cudaGraphModeEnabled(true),
    numParams(1),
    numCells(1),
    numUes(1)
{

    //Memset all launch configs structs to 0; they will be properly updated on every setup() as needed
    memset(&m_csirsRxChEstLaunchCfg, 0, sizeof(cuphyCsirsRxChEstLaunchCfg_t));

    maxCellsPerSlot = pStatPrms->nMaxCellsPerSlot;
    maxUesPerSlot = CUPHY_CSIRS_MAX_NUM_UES;
    pStatPrms->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    // Allocate workspace buffers and update relevant host/device pointers
    workspace_offsets[0] = 0; // unchanged
    updateWorkspaceOffsets(CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot, maxUesPerSlot, CUPHY_CSIRS_MAX_NUM_PARAMS*maxUesPerSlot);
    h_workspace = make_unique_pinned<uint8_t>(workspace_offsets[N_CSIRSRX_COMPONENTS]);
    d_workspace = make_unique_device<uint8_t>(workspace_offsets[N_CSIRSRX_COMPONENTS], &memory_footprint);
    updateWorkspacePtrs();

    // params pointers unchanged as they point to the beginning of the workspace buffer
    h_params = (CsirsRxParams*)(h_workspace.get() + workspace_offsets[0]);
    d_params = (CsirsRxParams*)(d_workspace.get() + workspace_offsets[0]);

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif

    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        memory_footprint.printMemoryFootprint(this, "CSIRSRX");
    }
}

void CsirsRx::updateWorkspaceOffsets(int nRrcParams, int nUes, int nUesRrcParams)
{
    int max_alignment = alignof(float2*); // for the output tensor addr.
    //workspace_offsets[0] = 0;
    workspace_offsets[1] = workspace_offsets[0] + round_up_to_next<int>(nRrcParams * sizeof(CsirsRxParams), max_alignment);//Alignment of UE params following CSI-RS params
    workspace_offsets[2] = workspace_offsets[1] + round_up_to_next<int>((nUes) * sizeof(CsirsRxUeParams), max_alignment);  // Alignment of input tensor addr. after UE params
    workspace_offsets[3] = workspace_offsets[2] + round_up_to_next<int>(nUes * sizeof(float2*), max_alignment);            // Alignment of channel estimation tensor addr. after input tensor addr.
    workspace_offsets[4] = workspace_offsets[3] + round_up_to_next<int>(nUesRrcParams * sizeof(float2*), max_alignment);
}

void CsirsRx::updateWorkspacePtrs()
{
    // The *_params pointers are updated once during CSIRS-TX creation and not here

    h_ue_params = (CsirsRxUeParams*)(h_workspace.get() + workspace_offsets[1]);
    d_ue_params = (CsirsRxUeParams*)(d_workspace.get() + workspace_offsets[1]);

    h_input_tensor_addr = (float2**)(h_workspace.get() + workspace_offsets[2]);
    d_input_tensor_addr = (float2**)(d_workspace.get() + workspace_offsets[2]);

    h_chest_tensor_addr = (float2**)(h_workspace.get() + workspace_offsets[3]);
    d_chest_tensor_addr = (float2**)(d_workspace.get() + workspace_offsets[3]);


}

CsirsRx::~CsirsRx()
{
    CUDA_CHECK_NO_THROW(cudaGraphDestroy(m_graph));
    CUDA_CHECK_NO_THROW(cudaGraphExecDestroy(m_graphExec));
}

cuphyStatus_t CsirsRx::checkConfig(CsirsRxParams* params, int numCsirsRxParams)
{
    for(int i = 0; i < numCsirsRxParams; ++i)
    {
        if (params[i].startRb + params[i].nRb > MAX_N_PRBS_SUPPORTED)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported startRb/nRb");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if (params[i].csiType == cuphyCsiType_t::ZP_CSI_RS)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Should not call CSI-RS channel with zero power CSI Type.");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if (params[i].li[0] > OFDM_SYMBOLS_PER_SLOT || params[i].li[1] > OFDM_SYMBOLS_PER_SLOT)
        {
            NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unsupported time domain location.");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

//template <fmtlog::LogLevel log_level>
//void CsirsRx::printCsirsConfig(const cuphyCsirsStatPrms_t& cell_static_params, const cuphyCsirsDynPrms_t& dyn_params)
//{
//    NVLOG_FMT(log_level, NVLOG_CSIRS, "CSIRS pipeline parameters: ");
//
//    int total_rrc_params = 0;
//    for(int i = 0; i < dyn_params.nCells; i++)
//    {
//        const cuphyCsirsCellDynPrm_t& params = dyn_params.pCellParam[i];
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "------------------------------------------------------");
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "pCellParam Index:  {:5d}", i);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "rrcParamsOffset:   {:5d}", params.rrcParamsOffset);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "nRrcParams:        {:5d}", params.nRrcParams);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "slotBufferIdx:     {:5d}", params.slotBufferIdx);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "cellPrmStatIdx:    {:5d}", params.cellPrmStatIdx);
//        // only the nPrbDlBwp field from the static parameters is currently used, so print only that
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "cell's {} nPrbDlBwp:{:5d} (from static parameters)", params.cellPrmStatIdx, cell_static_params.pCellStatPrms[params.cellPrmStatIdx].nPrbDlBwp);
//        total_rrc_params += params.nRrcParams;
//    }
//
//    NVLOG_FMT(log_level, NVLOG_CSIRS, "CSIRS TX pipeline with {} precoded CSIRS RRC", dyn_params.nPrecodingMatrices);
//    for(int i = 0; i < total_rrc_params; ++i)
//    {
//        const cuphyCsirsRrcDynPrm_t& params = dyn_params.pRrcDynPrm[i];
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "------------------------------------------------------");
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "pRrcDynPrm Index: {:5d}", i);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "startRb:          {:5d}", params.startRb);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "nRb:              {:5d}", params.nRb);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "csiType:          {:5d}", params.csiType);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "row:              {:5d}", params.row);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "freqDomain:       {:5d}", params.freqDomain);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "symbL0:           {:5d}", params.symbL0);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "symbL1:           {:5d}", params.symbL1);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "cdmType:          {:5d}", params.cdmType);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "freqDensity:      {:5d}", params.freqDensity);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "scrambId:         {:5d}", params.scrambId);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "beta:             {:.2f}", params.beta);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "idxSlotInFrame:   {:5d}", params.idxSlotInFrame);
//        NVLOG_FMT(log_level, NVLOG_CSIRS, "enablePrcdBf:     {:5d}", params.enablePrcdBf);
//        if(params.enablePrcdBf)
//        {
//            NVLOG_FMT(log_level, NVLOG_CSIRS, "pmwPrmIdx:        {:5d}", params.pmwPrmIdx);
//            NVLOG_FMT(log_level, NVLOG_CSIRS, "nPorts :          {:5d}", dyn_params.pPmwParams[params.pmwPrmIdx].nPorts);
//            std::stringstream matrix_row;
//            matrix_row.precision(5);
//            matrix_row << "Precoding Matrix:      ";
//            for(int idx = 0; idx < dyn_params.pPmwParams[params.pmwPrmIdx].nPorts; idx++)
//            {
//                if (idx != 0) matrix_row << ", ";
//                matrix_row << std::fixed << "{" << (float)dyn_params.pPmwParams[params.pmwPrmIdx].matrix[idx].x << ", " <<  (float)dyn_params.pPmwParams[params.pmwPrmIdx].matrix[idx].y << "}";
//            }
//            NVLOG_FMT(log_level, NVLOG_CSIRS, "{}", matrix_row.str());
//        }
//    }
//}

cuphyStatus_t CsirsRx::expandParameters(cuphyCsirsRxDynPrms_t* dyn_params, cudaStream_t cuda_strm)
{
    if(dyn_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "expandParameters() got cuphyCsirsRxDynPrms_t nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    //printCsirsConfig<fmtlog::DBG>(*static_params, *dyn_params);
    numCells = dyn_params->nCells;

    if (numCells > maxCellsPerSlot) {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Number of cells passed in CSI-RS RX setup {} > nMaxCellsPerSlot from static parameters {}.", numCells, maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    numUes = dyn_params->nUes;

    if (numUes > maxUesPerSlot) {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Number of ues passed in CSI-RS RX setup {} > nMaxUesPerSlot from static parameters {}.", numUes, maxUesPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    numParams = 0;
    maxNumRrcParamPerUe = 0;
    for (int j = 0; j < numCells; j++)
    {
        numParams += dyn_params->pCellParam[j].nRrcParams;
        if(maxNumRrcParamPerUe < dyn_params->pCellParam[j].nRrcParams)
        {
            maxNumRrcParamPerUe = dyn_params->pCellParam[j].nRrcParams;
        }
    }
    
    if(numParams > (CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot))
    {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Number of parameters in cell passed in CSI-RS RX setup ({}) is more than {}.", numParams, CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    numUesRrcParams = 0;
    for (int j =0; j < numUes; j++)
    {
        uint16_t cellIdx = dyn_params->pUeParam[j].cellPrmStatIdx;
        numUesRrcParams += dyn_params->pCellParam[cellIdx].nRrcParams;
    }
    
    if(numUesRrcParams > (CUPHY_CSIRS_MAX_NUM_PARAMS * maxUesPerSlot))
    {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Number of parameters in ue passed in CSI-RS RX setup ({}) is more than {}.", numUesRrcParams, CUPHY_CSIRS_MAX_NUM_PARAMS * maxCellsPerSlot);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    updateWorkspaceOffsets(numParams, numUes, numUesRrcParams);
    updateWorkspacePtrs();


    for (int j = 0; j < numCells; j++)
    {
        uint16_t cell_index = dyn_params->pCellParam[j].cellPrmStatIdx; 

        for(int i = dyn_params->pCellParam[j].rrcParamsOffset; i < dyn_params->pCellParam[j].rrcParamsOffset + dyn_params->pCellParam[j].nRrcParams; ++i)
        {
            if(dyn_params->pRrcDynPrm[i].row > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH)
            {
                NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Row {} in CSI-RS RX parameter {}  of cell {} is more than {}.", dyn_params->pRrcDynPrm[i].row, i, j, CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            CsirsRxParams* h_param = &(h_params[i]);
            // copy and expand config params to CSIRS params
            h_param->startRb = dyn_params->pRrcDynPrm[i].startRb;
            h_param->nRb = dyn_params->pRrcDynPrm[i].nRb;
            h_param->freqDensity = dyn_params->pRrcDynPrm[i].freqDensity;
            h_param->csiType = dyn_params->pRrcDynPrm[i].csiType;
            h_param->row = dyn_params->pRrcDynPrm[i].row;
            h_param->li[0] = dyn_params->pRrcDynPrm[i].symbL0;
            h_param->li[1] = dyn_params->pRrcDynPrm[i].symbL1;
            h_param->cdmType = dyn_params->pRrcDynPrm[i].cdmType;
            h_param->scrambId = dyn_params->pRrcDynPrm[i].scrambId;
            h_param->idxSlotInFrame = dyn_params->pRrcDynPrm[i].idxSlotInFrame;
            h_param->beta = dyn_params->pRrcDynPrm[i].beta;
            h_param->enablePrcdBf = 0;
            h_param->pmwPrmIdx = 0;

            //update cell_index; needed to access the addr. of this cell's output tensor and its number of PRBs (freq. dimension)
            h_param->cell_index = cell_index;

            if ((h_param->csiType == cuphyCsiType_t::TRS) && ((h_param->row != 1) || (dyn_params->pRrcDynPrm[i].freqDensity != 3)))
            {
                NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Invalid TRS config. Should have row = 1 and frequency density = 3.");
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            switch (h_param->cdmType)
            {
                case cuphyCdmType_t::NO_CDM:
                    h_param->seqIndexCount = 1;
                    break;
                case cuphyCdmType_t::CDM2_FD:
                    h_param->seqIndexCount = 2;
                    break;
                case cuphyCdmType_t::CDM4_FD2_TD2:
                    h_param->seqIndexCount = 4;
                    break;
                case cuphyCdmType_t::CDM8_FD2_TD4:
                    h_param->seqIndexCount = 8;
                    break;
                default:
                {
                    NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Unknown cdmType");
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
            }

            h_param->rho = 0.0f;
            h_param->genEvenRB = 0;
            switch (dyn_params->pRrcDynPrm[i].freqDensity)
            {
                case 0:
                    h_param->rho = 0.5f;
                    h_param->genEvenRB = 1;
                    break;
                case 1:
                    h_param->rho = 0.5f;
                    h_param->genEvenRB = 0;
                    break;
                case 2:
                    h_param->rho = 1;
                    break;
                case 3:
                    h_param->rho = 3;
                    break;
                default:
                {
                    NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Unknown freqDensity value {} in CSI-RS RX parameter {} of cell {}", dyn_params->pRrcDynPrm[i].freqDensity, i, j);
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
            }

            uint8_t numPorts = csirsRowDataNumPorts[h_param->row - 1];
            if(numPorts == 1)
            {
                h_param->alpha = h_param->rho;
            }
            else
            {
                h_param->alpha = 2 * h_param->rho;
            }

            if(h_param->csiType == cuphyCsiType_t::ZP_CSI_RS)
            {
                h_param->beta = 0;
            }

            // fill ki
            uint16_t freqDomain = dyn_params->pRrcDynPrm[i].freqDomain;
            for(int i = 0; i < CUPHY_CSIRS_MAX_KI_INDEX_LENGTH && freqDomain > 0; ++i)
            {
                // rightmost(least significant) bit
                uint8_t ki = log2((freqDomain & (freqDomain - 1)) ^ freqDomain);
                switch (h_param->row)
                {
                    case 1:
                    case 2:
                        h_param->ki[i] = ki;
                        break;
                    case 4:
                        h_param->ki[i] = 4 * ki;
                        break;
                    default:
                        h_param->ki[i] = 2 * ki;
                }

                freqDomain = freqDomain & (freqDomain - 1);
            }
        }
    }
    
    int countUesRrcParams = 0;
    maxNumPrbPerUe = 0; 
    for (int j = 0; j < numUes; j++)
    {
        h_input_tensor_addr[j] = (float2*)dyn_params->pDataIn->pTDataRx[j].pAddr;
        
        CsirsRxUeParams* h_ue_param = &(h_ue_params[j]);
        h_ue_param->cellIdx = dyn_params->pUeParam[j].cellPrmStatIdx;
        h_ue_param->csirsRxParamsStartIdx = dyn_params->pCellParam[h_ue_param->cellIdx].rrcParamsOffset;
        h_ue_param->nCsirs = dyn_params->pCellParam[h_ue_param->cellIdx].nRrcParams;
        h_ue_param->nRxAnt = dyn_params->pUeParam[j].nRxAnt;
        h_ue_param->nRxRe  = static_params->pCellStatPrms[h_ue_param->cellIdx].nPrbDlBwp * CUPHY_N_TONES_PER_PRB;
        h_ue_param->csirsRxChEstBufStartIdx = countUesRrcParams;
        
        dyn_params->pDataOut->pChEstBuffInfo[j].nCsirs = h_ue_param->nCsirs;
        
        for (int i = 0; i < h_ue_param->nCsirs; i++)
        {
            h_chest_tensor_addr[countUesRrcParams] = (float2*)dyn_params->pDataOut->pChEstBuffInfo[j].tChEstBuffer[i].pAddr;
            
            dyn_params->pDataOut->pChEstBuffInfo[j].startPrb[i] = dyn_params->pRrcDynPrm[h_ue_param->csirsRxParamsStartIdx + i].startRb;
            
            uint16_t nRb         = dyn_params->pRrcDynPrm[h_ue_param->csirsRxParamsStartIdx + i].nRb;
            
            if(maxNumPrbPerUe < nRb)
            {
                maxNumPrbPerUe = nRb;
            }
            
            uint8_t  row         = dyn_params->pRrcDynPrm[h_ue_param->csirsRxParamsStartIdx + i].row; 
            uint8_t  freqDensity = dyn_params->pRrcDynPrm[h_ue_param->csirsRxParamsStartIdx + i].freqDensity; 
            if((row==1)&&(freqDensity==3))
            {
                nRb = nRb*3;
            }
            else if((freqDensity==0)||(freqDensity==1))
            {
                nRb = (nRb>>1);
            }
            dyn_params->pDataOut->pChEstBuffInfo[j].sizePrb[i] = nRb; 
            
            countUesRrcParams++;
            
            //printf("UE[%d]csirs[%d]nRb[%d]nRxAnt[%d]countUesRrcParams[%d]nRxRe[%d]\n", j, i, nRb, h_ue_param->nRxAnt, countUesRrcParams, h_ue_param->nRxRe);
        }   
    }
    
    //printf("numUes[%d]maxNumRrcParamPerUe[%d]maxNumPrbPerUe[%d]\n", numUes, maxNumRrcParamPerUe, maxNumPrbPerUe);
    
    if(countUesRrcParams!=numUesRrcParams)
    {
        NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "CSI-RS RX setup wrong with countUesRrcParams {} != numUesRrcParams {}", countUesRrcParams, numUesRrcParams);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

#if CHECK_CONFIG
    cuphyStatus_t check_config_status = checkConfig(h_params, numParams);
    if (check_config_status != CUPHY_STATUS_SUCCESS) {
        return check_config_status;
    }
#endif


    CUDA_CHECK(cudaMemcpyAsync(d_workspace.get(), h_workspace.get(), workspace_offsets[N_CSIRSRX_COMPONENTS], cudaMemcpyHostToDevice, cuda_strm));
    return CUPHY_STATUS_SUCCESS;
}

void CsirsRx::createGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating.
    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. If the number of kernel parameters changes, the calls below should be updated.
    void* arg;
    void* kernelParams[9] = {&arg, &arg, &arg, &arg, &arg, &arg, &arg, &arg, &arg}; //use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode4ParamsDriver, 4, &kernelParams[0]));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_csirsRxChEstNode, m_graph, nullptr, 0, &m_emptyNode4ParamsDriver));
}

void CsirsRx::updateGraph()
{
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_CSIRS, cuGraphExecKernelNodeSetParams(m_graphExec, m_csirsRxChEstNode, &(m_csirsRxChEstLaunchCfg.kernelNodeParamsDriver)));
}

void CsirsRx::setKernelParams()
{
    cuphyCsirsRxKernelSelect(&m_csirsRxChEstLaunchCfg, numUes, maxNumRrcParamPerUe, maxNumPrbPerUe);

    // set kernel args for genCsirsTfSignalKernel()
    m_chEstArgs[0] = d_input_tensor_addr;
    m_chEstArgs[1] = d_chest_tensor_addr;
    m_chEstArgs[2] = d_params;
    m_chEstArgs[3] = d_ue_params;

    m_csirsRxChEstLaunchCfg.kernelArgs[0] = &m_chEstArgs[0];
    m_csirsRxChEstLaunchCfg.kernelArgs[1] = &m_chEstArgs[1];
    m_csirsRxChEstLaunchCfg.kernelArgs[2] = &m_chEstArgs[2];  
    m_csirsRxChEstLaunchCfg.kernelArgs[3] = &m_chEstArgs[3];  

    m_csirsRxChEstLaunchCfg.kernelNodeParamsDriver.kernelParams = m_csirsRxChEstLaunchCfg.kernelArgs;
    //---------------------------------------------

    //executable graph setup
    m_cudaGraphModeEnabled = (dynamic_params->procModeBmsk & CSIRS_PROC_MODE_GRAPHS) ? true : false;
    if(m_cudaGraphModeEnabled)
    {
        updateGraph();
    }

}

// Generate CSIRS symbols and map them to subcarriers
cuphyStatus_t CsirsRx::run(const cudaStream_t& cuda_strm)
{
    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult e = cuGraphLaunch(m_graphExec, cuda_strm);
        if (e != CUDA_SUCCESS) {
            const char* pErrStr;
            CUresult str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Invalid graph launch for CSI-RS RX ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }
    else
    {
        CUresult e = launch_kernel(m_csirsRxChEstLaunchCfg.kernelNodeParamsDriver, cuda_strm);
        if(e != CUDA_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_CSIRSRX, AERIAL_CUPHY_EVENT, "Invalid argument for csirsRxChEstKernel launch.");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

