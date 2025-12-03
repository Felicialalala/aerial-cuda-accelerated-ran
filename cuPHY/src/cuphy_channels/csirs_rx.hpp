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

#if !defined(CSIRS_RX_HPP_INCLUDED_)
#define CSIRS_RX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "util.hpp"
#include "csirs_rx/csirs_rx.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>

struct cuphyCsirsRx
{};

class CsirsRx : public cuphyCsirsRx {

public:

    enum Component
    {
        CSIRSRX_PARAMS             = 0,
        CSIRSRX_UE_PARAMS          = 1,
        CSIRSRX_INPUT_TENSOR_ADDR  = 2,
        CSIRSRX_CHEST_TENSOR_ADDR  = 3,
        N_CSIRSRX_COMPONENTS       = 4
    };

    /**
     * @brief: Construct CsirsRx class.
     * @param[in] pStatPrms: static parameters for CsirsRx
     */
     CsirsRx(cuphyCsirsStatPrms_t const* pStatPrms);

    /**
     * @brief: CsirsRx cleanup.
     */
    ~CsirsRx();

    /**
     * @brief: Convert input dyn_params to CsirsRxParams struct
     * @param[in] dyn_params: input parameters to CSI-RS RX.
     * @param[in] cuda_strm: CUDA stream for memcpys.
     * @return CUPHY_STATUS_SUCCESS or relevant error status
     */
    cuphyStatus_t expandParameters(cuphyCsirsRxDynPrms_t* dyn_params, cudaStream_t cuda_strm);

    /**
     * @brief: set kernel launch parameters
     */
    void setKernelParams();

    /**
     * @brief: Run CSI-RS
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INTERNAL_ERROR
     */
    cuphyStatus_t run(const cudaStream_t& cuda_strm);

    const cuphyCsirsRxDynPrms_t*  dynamic_params;

//    /**
//     * @brief Print input parameters
//     * @param[in] cell_static_params: CSI-RS static input parameters
//     * @param[in] dyn_params: CSI-RS dynamic input parameters
//     */
//    template <fmtlog::LogLevel log_level=fmtlog::DBG>
//    void printCsirsConfig(const cuphyCsirsStatPrms_t& cell_static_params, const cuphyCsirsRxDynPrms_t& dyn_params);

    const void* getMemoryTracker();

    CUgraph *GetGraph() { return &m_graph; }

    const cuphyCsirsStatPrms_t* static_params;

private:

    cuphyMemoryFootprint memory_footprint;

    /**
     * @brief Validate input parameters
     * @param[in] params: Parameter list array
     * @param[in] numCsirsRxParams: number of elements in params
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
     */
    cuphyStatus_t checkConfig(CsirsRxParams* params, int numCsirsRxParams);

    // Workspace buffers that enables single, not overprovisioned H2D copy, per setup instead of the
    // overprovisioned one with descriptors.
    cuphy::unique_device_ptr<uint8_t> d_workspace;
    cuphy::unique_pinned_ptr<uint8_t> h_workspace;

    // Pointers to host or device buffers within h_workspace  or d_workspace respectively

    // CSI-RS input parameter list
    CsirsRxParams* h_params;
    CsirsRxParams* d_params;
    
    CsirsRxUeParams* h_ue_params;
    CsirsRxUeParams* d_ue_params;

    // kernel launch configurations
    cuphyCsirsRxChEstLaunchCfg_t m_csirsRxChEstLaunchCfg;

    // kernel args
    void* m_chEstArgs[4];

    // graph functions
    void createGraph();
    void updateGraph();

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_csirsRxChEstNode;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode4ParamsDriver;

    float2** h_input_tensor_addr;
    float2** d_input_tensor_addr;
    
    float2** h_chest_tensor_addr;
    float2** d_chest_tensor_addr;


    // Offset of the various host/device buffers, in bytes, from the start of the workspace
    std::array<int, N_CSIRSRX_COMPONENTS+1> workspace_offsets;

    // Update workspace_offsets for all buffers for each new config of cells and CSI-RS parameters
    void updateWorkspaceOffsets(int nRrcParams, int nUes, int nUesRrcParams);
    // Update the workspace pointers
    void updateWorkspacePtrs();

    // number of CSI-RS parameters in the batch across all cells
    size_t numParams;

    // number of cells to be processed in this batch
    int    numCells;
    
    // number of ues to be processed in this batch
    int    numUes;
    
    int    numUesRrcParams;

    // maximum number of cells to be processed per slot
    int    maxCellsPerSlot;
    
    // maximum number of ues to be processed per slot
    int    maxUesPerSlot;
    
    int    maxNumRrcParamPerUe;
    
    int    maxNumPrbPerUe;
};

#endif // !defined(CSIRS_RX_HPP_INCLUDED_)
