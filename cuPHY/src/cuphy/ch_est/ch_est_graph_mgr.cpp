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

#include "ch_est_graph_mgr.hpp"
#include "cuphy.hpp"
#include "cuphy_utils.hpp"

namespace ch_est {

void addKernelImpl(CUgraphNode                     nodes[],
                   const std::size_t               nMaxChEstHetCfgs,
                   const CUgraph                   graph,
                   const std::vector<CUgraphNode>& currNodeDeps,
                   std::vector<CUgraphNode>&       nextNodeDeps,
                   const CUDA_KERNEL_NODE_PARAMS&  nodeParams)
{
    for(int hetCfgIdx = 0; hetCfgIdx < nMaxChEstHetCfgs; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&nodes[hetCfgIdx], graph,
            currNodeDeps.data(), currNodeDeps.size(),
            &nodeParams));
        nextNodeDeps.push_back(nodes[hetCfgIdx]);
    }
}

void addChestKernelImpl(CUgraphNode                     nodes[][CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS],
                        const std::size_t               nMaxChEstHetCfgs,
                        const CUgraph                   graph,
                        const std::vector<CUgraphNode>& currNodeDeps,
                        std::vector<CUgraphNode>&       nextNodeDeps,
                        const CUDA_KERNEL_NODE_PARAMS&  nodeParams)
{
    for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
    {
        addKernelImpl(nodes[chEstTimeInstIdx],
                      nMaxChEstHetCfgs,
                      graph,
                      currNodeDeps,
                      nextNodeDeps,
                      nodeParams);
    }
}

    void ChestSubSlotNodes::addKernelNodeToGraph(CUgraph graph,
                                                 std::vector<CUgraphNode> &currNodeDeps,
                                                 std::vector<CUgraphNode> &nextNodeDeps,
                                                 CUDA_KERNEL_NODE_PARAMS &nodeParams) {
        addKernelImpl(m_nodes,
                      m_nMaxChEstHetCfgs,
                      graph,
                      currNodeDeps,
                      nextNodeDeps,
                      nodeParams);

        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            currNodeDeps = nextNodeDeps;
            nextNodeDeps.clear();

            addSecondaryKernelNodeToGraph(graph,
                                          currNodeDeps,
                                          nextNodeDeps,
                                          nodeParams);
        }
    }

    void ChestSubSlotNodes::addSecondaryKernelNodeToGraph(CUgraph graph,
                                                          std::vector<CUgraphNode> &currNodeDeps,
                                                          std::vector<CUgraphNode> &nextNodeDeps,
                                                          CUDA_KERNEL_NODE_PARAMS &nodeParams) {
        addKernelImpl(m_secondNodes,
                      m_nMaxChEstHetCfgs,
                      graph,
                      currNodeDeps,
                      nextNodeDeps,
                      nodeParams);
    }

    void ChestSubSlotNodes::setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                          CUgraphExec graphExec) {
        const int nCfgs = disableAllNodes == ChestCudaUtils::DisableAllNodes::TRUE ? 0 : m_cfgs0Slot.nCfgs;
        cuphy_utils::setHetCfgNodeStatus(nCfgs, m_nMaxChEstHetCfgs, m_cfgs0Slot.cfgs,
                                         m_nodesEnabled, m_nodes, graphExec);

        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            setSecondaryNodeStatus(
                disableAllNodes,
                graphExec);
        }
    }

    void ChestSubSlotNodes::setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                                   CUgraphExec graphExec) {
        const int nCfgs = disableAllNodes == ChestCudaUtils::DisableAllNodes::TRUE ? 0 : m_cfgs0Slot.nCfgs;
        cuphy_utils::setHetCfgNodeStatusSecond(nCfgs, m_nMaxChEstHetCfgs, m_cfgs0Slot.cfgs,
                                               m_secondNodesEnabled, m_secondNodes, graphExec);
    }

    void ChestNodes::addKernelNodeToGraph(CUgraph graph,
                                          std::vector<CUgraphNode> &currNodeDeps,
                                          std::vector<CUgraphNode> &nextNodeDeps,
                                          CUDA_KERNEL_NODE_PARAMS &nodeParams) {
        addChestKernelImpl(m_chEstNodes,
                           m_nMaxChEstHetCfgs,
                           graph,
                           currNodeDeps,
                           nextNodeDeps,
                           nodeParams);

        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            currNodeDeps = nextNodeDeps;
            nextNodeDeps.clear();
            addSecondaryKernelNodeToGraph(graph,
                                          currNodeDeps,
                                          nextNodeDeps,
                                          nodeParams);
        }
    }

    void ChestNodes::addSecondaryKernelNodeToGraph(CUgraph graph,
                                                   std::vector<CUgraphNode> &currNodeDeps,
                                                   std::vector<CUgraphNode> &nextNodeDeps,
                                                   CUDA_KERNEL_NODE_PARAMS &nodeParams) {
        addChestKernelImpl(m_chEstSecondNodes,
                           m_nMaxChEstHetCfgs,
                           graph,
                           currNodeDeps,
                           nextNodeDeps,
                           nodeParams);
    }

    void ChestNodes::setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                   CUgraphExec graphExec) {
        const uint32_t startChEstInstIdx =
                (m_earlyHarqModeEnabled && m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
                1 : 0;
        for (int32_t chEstTimeInstIdx = startChEstInstIdx;
             chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx) {
            const int nCfgs = disableAllNodes == ChestCudaUtils::DisableAllNodes::TRUE ? 0 : m_chEstLaunchCfgs[chEstTimeInstIdx].nCfgs;
            cuphy_utils::setHetCfgNodeStatus(nCfgs, m_nMaxChEstHetCfgs, m_chEstLaunchCfgs[chEstTimeInstIdx].cfgs,
                                             m_chEstNodesEnabled[chEstTimeInstIdx],
                                             m_chEstNodes[chEstTimeInstIdx], graphExec);
        }

        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            setSecondaryNodeStatus(
                disableAllNodes,
                graphExec);
        }
    }

    void ChestNodes::setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                            CUgraphExec graphExec) {
        for (int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx) {
            const int nCfgs = disableAllNodes == ChestCudaUtils::DisableAllNodes::TRUE ? 0 : m_chEstLaunchCfgs[chEstTimeInstIdx].nCfgs;
            cuphy_utils::setHetCfgNodeStatusSecond(nCfgs, m_nMaxChEstHetCfgs, m_chEstLaunchCfgs[chEstTimeInstIdx].cfgs,
                                                   m_chEstSecondNodesEnabled[chEstTimeInstIdx],
                                                   m_chEstSecondNodes[chEstTimeInstIdx], graphExec);
        }
    }

    void ChestNodes::disableNodes0Slot(CUgraphExec graph) {
        const uint32_t startChEstInstIdx =
            (m_earlyHarqModeEnabled && m_chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
            1 : 0;

        if (startChEstInstIdx != 1)
        {
            return;
        }

        // disable channel estimation kernel nodes for instance index 0
        for(int hetCfgIdx = 0; hetCfgIdx < m_nMaxChEstHetCfgs; ++hetCfgIdx)
        {
            if (m_chEstNodesEnabled[0][hetCfgIdx] != 0)
            {
                m_chEstNodesEnabled[0][hetCfgIdx] = 0;
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graph, m_chEstNodes[0][hetCfgIdx], 0));
            }
        }
    }

}
