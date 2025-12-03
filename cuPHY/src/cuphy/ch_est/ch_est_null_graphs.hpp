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

#ifndef CUPHY_CHEST_NULL_GRAPHS_HPP
#define CUPHY_CHEST_NULL_GRAPHS_HPP

#include <vector>

#include "IGraph_mgr.hpp"

namespace ch_est {
    /**
     * @brief NullChestSubSlotNodes is implementing the NullObject pattern
     *        of IChestSubSlotNodes, to have no-op operations in case
     *        Early HARQ or DMRS kernels are not needed at the client side.
     *
     * @see PuschRx as an example of using the interface.
     */
    class NullChestSubSlotNodes final : public IChestSubSlotNodes {
    public:
        void addKernelNodeToGraph([[maybe_unused]] CUgraph graph,
                                  [[maybe_unused]] std::vector<CUgraphNode> &currNodeDeps,
                                  [[maybe_unused]] std::vector<CUgraphNode> &nextNodeDeps,
                                  [[maybe_unused]] CUDA_KERNEL_NODE_PARAMS &nodeParams) final {}
        void addSecondaryKernelNodeToGraph([[maybe_unused]] CUgraph graph,
                                           [[maybe_unused]] std::vector<CUgraphNode> &currNodeDeps,
                                           [[maybe_unused]] std::vector<CUgraphNode> &nextNodeDeps,
                                           [[maybe_unused]] CUDA_KERNEL_NODE_PARAMS &nodeParams) final {}

        void setNodeStatus([[maybe_unused]] ChestCudaUtils::DisableAllNodes disableAllNodes, [[maybe_unused]] CUgraphExec graphExec) final {}

        void
        setSecondaryNodeStatus([[maybe_unused]] ChestCudaUtils::DisableAllNodes disableAllNodes, [[maybe_unused]] CUgraphExec graphExec) final {}
    };

    /**
     * @brief NullChestGraph is implementing the NullObject pattern
     *        of IChestGraph, to have no-op operations in case
     *        Channel Estimate kernels are not needed at the client side.
     *
     * @see PuschRx as an example of using the interface.
     */
    class NullChestGraph final : public IChestGraphNodes {
    public:
        void init() final {}
        void setEarlyHarqModeEnabled([[maybe_unused]] bool earlyHarqModeEnabled) final {}
        void addKernelNodeToGraph([[maybe_unused]] CUgraph graph,
                                  [[maybe_unused]] std::vector<CUgraphNode> &currNodeDeps,
                                  [[maybe_unused]] std::vector<CUgraphNode> &nextNodeDeps,
                                  [[maybe_unused]] CUDA_KERNEL_NODE_PARAMS &nodeParams) final {}
        void addSecondaryKernelNodeToGraph([[maybe_unused]] CUgraph graph,
                                           [[maybe_unused]] std::vector<CUgraphNode> &currNodeDeps,
                                           [[maybe_unused]] std::vector<CUgraphNode> &nextNodeDeps,
                                           [[maybe_unused]] CUDA_KERNEL_NODE_PARAMS &nodeParams) final {}
        void setNodeStatus([[maybe_unused]] ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final {}
        void setSecondaryNodeStatus([[maybe_unused]] ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final {}
        void disableNodes0Slot([[maybe_unused]] CUgraphExec graphExec) final {}
    };

    /**
     * @brief NullChestStream is implementing the NullObject pattern
     *        of IChestStream, to have no-op operations in case
     *        Channel Estimate stream operations are not needed at the client side.
     *
     * @see PuschRx as an example of using the interface.
     */
    class NullChestStream final : public IChestStream {
    public:
        void launchKernels([[maybe_unused]] cudaStream_t phase1Stream) final {}
        void launchKernels0Slot([[maybe_unused]] cudaStream_t phase1Stream) final {}
        void launchSecondaryKernels([[maybe_unused]] cudaStream_t phase1Stream) final {}
        void launchSecondaryKernels0Slot([[maybe_unused]] cudaStream_t phase1Stream) final {}
    };
}

#endif //CUPHY_CHEST_NULL_GRAPHS_HPP
