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

#ifndef CUPHY_CHEST_IGRAPHMGR_HPP
#define CUPHY_CHEST_IGRAPHMGR_HPP

#include <vector>

#include "cuda.h"
#include "driver_types.h"

#include "cuphy.h"
#include "ch_est/ch_est_utils.hpp"

namespace ch_est {

// Helper functions for Graph manipulation operations

/**
 * @brief calls cuGraphAddKernelNode, iterating over nMaxChEstHetCfgs
 * @param nodes incoming params to pass to cuGraphAddKernelNode. It will populate it with a new node handle
 * @param nMaxChEstHetCfgs Maximum Heterogeneous configs
 * @param graph graph to add the node to
 * @param currNodeDeps dependencies to pass to cuGraphAddKernelNode
 * @param nextNodeDeps output dependencies list/vector. Every node handle added, will be on that list
 * @param nodeParams Node params, empty at the beginning
 */
void addKernelImpl(CUgraphNode                     nodes[],
                   std::size_t                     nMaxChEstHetCfgs,
                   CUgraph                         graph,
                   const std::vector<CUgraphNode>& currNodeDeps,
                   std::vector<CUgraphNode>&       nextNodeDeps,
                   const CUDA_KERNEL_NODE_PARAMS&  nodeParams);

/**
 * @brief calls cuGraphAddKernelNode, iterating over CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST and for each "nodes[idx]",
 * call addKernelImpl. This is basically a top wrapper to addKernelImpl
 * @param nodes incoming 2D params to pass to cuGraphAddKernelNode. It will populate it with a new node handle
 * @param nMaxChEstHetCfgs Maximum Heterogeneous configs
 * @param graph graph to add the node to
 * @param currNodeDeps dependencies to pass to cuGraphAddKernelNode
 * @param nextNodeDeps output dependencies list/vector. Every node handle added, will be on that list
 * @param nodeParams Node params, empty at the beginning
 */
void addChestKernelImpl(CUgraphNode                     nodes[][CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS],
                        std::size_t                     nMaxChEstHetCfgs,
                        CUgraph                         graph,
                        const std::vector<CUgraphNode>& currNodeDeps,
                        std::vector<CUgraphNode>&       nextNodeDeps,
                        const CUDA_KERNEL_NODE_PARAMS&  nodeParams);

/**
 * @class IChestSubSlotNodes is designed to be a base class of both
 *        Early-HARQ and DMRS kernels/graph nodes management.
 *
 * The base class is designed to provide a uniform interface to clients
 * using Early HARQ or DMRS kernels.
 * The same interface serves the purpose of both
 * use cases since there is a complete symmetry.
 */
class IChestSubSlotNodes {
public:
    IChestSubSlotNodes() = default;
    virtual ~IChestSubSlotNodes() = default;
    IChestSubSlotNodes(const IChestSubSlotNodes& earlyHarqGraph) = default;
    IChestSubSlotNodes& operator=(const IChestSubSlotNodes& earlyHarqGraph) = default;
    IChestSubSlotNodes(IChestSubSlotNodes&& earlyHarqGraph) = default;
    IChestSubSlotNodes& operator=(IChestSubSlotNodes&& earlyHarqGraph) = default;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Kernel parameter to be used for the KERNEL that is added.
     */
    virtual void addKernelNodeToGraph(CUgraph graph,
                                      std::vector<CUgraphNode> &currNodeDeps,
                                      std::vector<CUgraphNode> &nextNodeDeps,
                                      CUDA_KERNEL_NODE_PARAMS &nodeParams) = 0;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * This is the secondary kernel of Channel Estimate
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams kernel parameter to be used for the KERNEL that is added.
     */
    virtual void addSecondaryKernelNodeToGraph(CUgraph graph,
                                               std::vector<CUgraphNode> &currNodeDeps,
                                               std::vector<CUgraphNode> &nextNodeDeps,
                                               CUDA_KERNEL_NODE_PARAMS &nodeParams) = 0;


    /**
     * @brief set node status on the primary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    virtual void setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) = 0;

    /**
     * @brief set node status on the secondary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    virtual void
    setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) = 0;
};

/**
 * @brief IChestGraph is designed to be a base class of Channel estimate
 *        specific kernels/graph nodes management.
 *
 * The base class is designed to provide a uniform interface to clients
 * using Channel estimate kernels.
 */
class IChestGraphNodes{
public:
    IChestGraphNodes()  = default;
    virtual ~IChestGraphNodes() = default;
    IChestGraphNodes(const IChestGraphNodes& chestGraph) = default;
    IChestGraphNodes& operator=(const IChestGraphNodes& chestGraph) = default;
    IChestGraphNodes(IChestGraphNodes&& chestGraph) = default;
    IChestGraphNodes& operator=(IChestGraphNodes&& chestGraph) = default;

    /**
     * @brief implement any initialization logic here.
     */
    virtual void init() = 0;

    /**
     * @brief Set boolean of early HARQ if needed. no-op otherwise.
     * @param earlyHarqModeEnabled true/false
     */
    virtual void setEarlyHarqModeEnabled(bool earlyHarqModeEnabled) = 0;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Paramters to be used for the KERNEL that is added.
     */
    virtual void addKernelNodeToGraph(CUgraph graph,
                                      std::vector<CUgraphNode> &currNodeDeps,
                                      std::vector<CUgraphNode> &nextNodeDeps,
                                      CUDA_KERNEL_NODE_PARAMS &nodeParams) = 0;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * This is the secondary kernel of Channel Estimate
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams to be used for the KERNEL that is added.
     */
    virtual void addSecondaryKernelNodeToGraph(CUgraph graph,
                                               std::vector<CUgraphNode> &currNodeDeps,
                                               std::vector<CUgraphNode> &nextNodeDeps,
                                               CUDA_KERNEL_NODE_PARAMS &nodeParams) = 0;

    /**
     * @brief set node status on the primary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    virtual void setNodeStatus(ch_est::ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) = 0;

    /**
     * @brief set node status on the secondary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    virtual void
    setSecondaryNodeStatus(ch_est::ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) = 0;

    /**
     * @brief disable nodes in slot0
     * @param graphExec CUDA graph exec
     */
    virtual void disableNodes0Slot(CUgraphExec graphExec) = 0;
};

} // namespace ch_est

#endif //CUPHY_CHEST_IGRAPHMGR_HPP
