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

#if !defined(SRS_TX_HPP_INCLUDED_)
#define SRS_TX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "srstx/srstx.hpp"

struct cuphySrsTx
{};

class SrsTx : public cuphySrsTx {

public:
    static constexpr int Ng = 273*2*3;

    enum Component
    {
        SRSTX_PARAMS       = 0,
        SRSTX_TENSOR_ADDR  = 1,
        N_SRSTX_COMPONENTS = 2
    };

    explicit SrsTx(cuphySrsTxStatPrms_t const* pStatPrms);
    ~SrsTx();
    
    const cuphySrsTxStatPrms_t* static_params{};
    const cuphySrsTxDynPrms_t*  dynamic_params{};

    cuphyStatus_t expandParameters(cuphySrsTxDynPrms_t* dyn_params, cudaStream_t cuda_strm);

    /**
     * @brief: set kernel launch parameters
     */
    void setKernelParams();

    /**
     * @brief: Run SRS TX
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INTERNAL_ERROR
     */
    cuphyStatus_t run(const cudaStream_t& cuda_strm) const;

    /**
     * @brief Print input parameters
     * @param[in] static_params: CSI-RS static input parameters
     * @param[in] dyn_params: CSI-RS dynamic input parameters
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    void printSrsTxConfig(const cuphySrsTxStatPrms_t& static_params, const cuphySrsTxDynPrms_t& dyn_params);

    const void* getMemoryTracker();

    CUgraph *GetGraph() { return &m_graph; }

private:

    cuphyMemoryFootprint memory_footprint;

    /**
     * @brief Validate input parameters
     * @param[in] params: Parameter list array
     * @param[in] numSrsTxParams: number of elements in params
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
     */
    cuphyStatus_t checkConfig(SrsTxParams* params, int numSrsTxParams);

    // Workspace buffers that enables single, not overprovisioned H2D copy, per setup instead of the
    // overprovisioned one with descriptors.
    cuphy::unique_device_ptr<uint8_t> d_workspace;
    cuphy::unique_pinned_ptr<uint8_t> h_workspace;

    // Pointers to host or device buffers within h_workspace  or d_workspace respectively

    // CSI-RS input parameter list
    SrsTxParams* h_params{};
    SrsTxParams* d_params{};

    // kernel launch configurations
    cuphyGenSrsTxLaunchCfg_t    m_genSrsTxLaunchCfg{};

    // kernel args
    void* m_genSrsTxArgs[2]{};

    // graph functions
    void createGraph();
    void updateGraph();

    // graph parameters
    bool        m_cudaGraphModeEnabled{};
    CUgraph     m_graph{};
    CUgraphExec m_graphExec{};
    CUgraphNode m_genSrsTxNode{};
    CUDA_KERNEL_NODE_PARAMS m_emptyNode2ParamsDriver{};


    // Offset array used to map starting thread index for each parameter in the batch
    uint32_t* h_offsets{};
    uint32_t* d_offsets{};

    // Array with output tensor address for each cell in the batch
    __half2** h_ue_tensor_addr{};
    __half2** d_ue_tensor_addr{};


    // Offset of the various host/device buffers, in bytes, from the start of the workspace
    std::array<int, N_SRSTX_COMPONENTS+1> workspace_offsets{};

    // Update workspace_offsets for all buffers for each new config of cells and CSI-RS parameters
    void updateWorkspaceOffsets(int nSrsTxParams);
    // Update the workspace pointers
    void updateWorkspacePtrs();
};

#endif // !defined(SRS_TX_HPP_INCLUDED_)
