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

#ifndef CUPHY_TRTENGINE_CHEST_HPP
#define CUPHY_TRTENGINE_CHEST_HPP

#include <stdexcept>

#include "fmt/format.h"

#include <gsl-lite/gsl-lite.hpp>

#include "tensor_desc.hpp"
#include "trt_engine/trt_engine.hpp"
#include "ch_est_config_loader.hpp"

#include "IModule.hpp"
#include "ch_est_graph_mgr.hpp"
#include "ch_est_settings.hpp"
#include "pusch_start_kernels.hpp"
#include "ch_est_trtengine_pre_post_conversion.hpp"

namespace ch_est {

// Pre, TrtEngine Node, Post nodes. total 3.
inline constexpr auto NUM_NODES_IN_SECONDARY_PHASE = 3ULL;

/**
 * @brief TrtEngineChestSubSlotNodes is implementing sub-slot related logic
 * for Channel Estimate and TrtEngine.
 */
class TrtEngineChestSubSlotNodes final : public IChestSubSlotNodes {
public:
    /**
     * @brief Constructor
     * @param nMaxChEstHetCfgs Max Channel estimate Cfgs
     * @param preConvertCfg Kernel Launch config for pre-kernel
     * @param postConvertCfg Kernel Launch config for post-kernel
     * @param cfgs0Slot single config, from 0 slot
     * @param pCapture Stream capture, to obtain the Graph Node as a result of the capture
     */
    explicit TrtEngineChestSubSlotNodes(const std::uint32_t                                                                 nMaxChEstHetCfgs,
                                        const cuphyPuschChEstAlgoType_t                                                     chEstAlgo,
                                        const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& preConvertCfg,
                                        const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& postConvertCfg,
                                        const cuphyPuschRxChEstLaunchCfgs_t&                                                cfgs0Slot,
                                        trt_engine::CaptureStreamPrePostTrtEngEnqueue*                                      pCapture) :
        m_chEstAlgo(chEstAlgo),
        m_nMaxChEstHetCfgs{nMaxChEstHetCfgs},
        m_preConvertCfg(preConvertCfg),
        m_postConvertCfg(postConvertCfg),
        m_cfgs0Slot{cfgs0Slot},
        m_nodesEnabled(m_nMaxChEstHetCfgs, std::numeric_limits<uint8_t>::max()),
        m_secondNodesEnabled(NUM_NODES_IN_SECONDARY_PHASE, std::numeric_limits<uint8_t>::max()),
        m_pCapture(pCapture) {}

    TrtEngineChestSubSlotNodes(const TrtEngineChestSubSlotNodes &chestEarlyHarqNodes) = delete;
    TrtEngineChestSubSlotNodes &operator=(const TrtEngineChestSubSlotNodes &chestEarlyHarqNodes) = delete;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Parameters to be used for the KERNEL that is added.
     */
    void addKernelNodeToGraph(CUgraph graph,
                              std::vector<CUgraphNode> &currNodeDeps,
                              std::vector<CUgraphNode> &nextNodeDeps,
                              CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * This is the secondary kernel of Channel Estimate
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Parameters to be used for the KERNEL that is added.
     */
    void addSecondaryKernelNodeToGraph(CUgraph graph,
                                       std::vector<CUgraphNode> &currNodeDeps,
                                       std::vector<CUgraphNode> &nextNodeDeps,
                                       CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief set node status on the primary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

    /**
     * @brief set node status on the secondary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

private:
    cuphyPuschChEstAlgoType_t                                                           m_chEstAlgo;
    std::uint32_t                                                                       m_nMaxChEstHetCfgs{};
    const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& m_preConvertCfg;
    const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& m_postConvertCfg;
    const cuphyPuschRxChEstLaunchCfgs_t&                                                m_cfgs0Slot; // single cfg
    CUgraphNode                                                                         m_nodes[CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{},
                                                                                        m_secondNodes[NUM_NODES_IN_SECONDARY_PHASE]{};
    std::vector<uint8_t>                           m_nodesEnabled, m_secondNodesEnabled;
    trt_engine::CaptureStreamPrePostTrtEngEnqueue* m_pCapture{};
};

    /**
     * @brief TrtEngineChestNodeGraph implements Channel Estimate Graph nodes handling for TrtEngine
     * use case. This is for Full Slot use-case.
     */
    class TrtEngineChestNodeGraph final :  public IChestGraphNodes {
    public:
        /**
         * @brief Constructor
         * @param nMaxChEstHetCfgs Max Channel estimate Cfgs
         * @param preConvertCfg Kernel Launch config for pre-kernel
         * @param postConvertCfg Kernel Launch config for post-kernel
         * @param chEstAlgo The current configured channel estimate
         * @param chEstLaunchCfgs non-owning view to the Kernel Launch configs of Channel Estimate
         * @param pCapture Stream capture, to obtain the Graph Node as a result of the capture
         */
        explicit TrtEngineChestNodeGraph(const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& preConvertCfg,
                                         const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& postConvertCfg,
                                         const cuphyPuschChEstAlgoType_t                                                     chEstAlgo,
                                         gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t>                                            chEstLaunchCfgs,
                                         const std::size_t                                                                   nMaxChEstHetCfgs,
                                         trt_engine::CaptureStreamPrePostTrtEngEnqueue*                                      pCapture) :
            m_preConvertCfg(preConvertCfg),
            m_postConvertCfg(postConvertCfg),
            m_chEstAlgo(chEstAlgo),
            m_chEstLaunchCfgs(chEstLaunchCfgs),
            m_nMaxChEstHetCfgs(nMaxChEstHetCfgs),
            m_chEstNodesEnabled(CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST,
                                std::vector<uint8_t>(m_nMaxChEstHetCfgs,
                                                     std::numeric_limits<uint8_t>::max())),
            m_chEstSecondNodesEnabled(NUM_NODES_IN_SECONDARY_PHASE, std::numeric_limits<uint8_t>::max()),
            m_pCapture(pCapture) {}
        // API
        void init() final {
            for (auto &m_chEstLaunchCfg: m_chEstLaunchCfgs) {
                m_chEstLaunchCfg.nCfgs = 0;
            }
        }
        void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) final {
            m_earlyHarqModeEnabled = earlyHarqModeEnabled;
        }
        void addKernelNodeToGraph(CUgraph                   graph,
                                  std::vector<CUgraphNode>& currNodeDeps,
                                  std::vector<CUgraphNode>& nextNodeDeps,
                                  CUDA_KERNEL_NODE_PARAMS&  nodeParams) final;
        void addSecondaryKernelNodeToGraph(CUgraph                   graph,
                                           std::vector<CUgraphNode>& currNodeDeps,
                                           std::vector<CUgraphNode>& nextNodeDeps,
                                           CUDA_KERNEL_NODE_PARAMS&  nodeParams) final;
        void setNodeStatus(ch_est::ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;
        void setSecondaryNodeStatus(ch_est::ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;
        [[nodiscard]] const auto &chEstFirstLaunchCfgs() const noexcept { return m_chEstLaunchCfgs[0]; }
        void disableNodes0Slot(CUgraphExec graphExec) final;
    private:
        const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& m_preConvertCfg;
        const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& m_postConvertCfg;
        const cuphyPuschChEstAlgoType_t                                                     m_chEstAlgo{};
        gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t>                                            m_chEstLaunchCfgs;
        bool                                                                                m_earlyHarqModeEnabled{};
        std::uint32_t                                                                       m_nMaxChEstHetCfgs{};
        CUgraphNode                                                                         m_chEstNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{};
        std::array<CUgraphNode, 3>                                                          m_chEstSecondNodes{};
        std::vector<std::vector<uint8_t>>                                                   m_chEstNodesEnabled;
        std::vector<uint8_t>                                                                m_chEstSecondNodesEnabled;
        trt_engine::CaptureStreamPrePostTrtEngEnqueue*                                      m_pCapture{};
    };

    /**
     * @brief Graph manager for TrtEngine use-case, composing Full and Sub-slot implementation.
     */
    class TrtEngineChestGraphMgr final {
    public:
        /**
         * @brief Constructor
         * @param preConvertCfg Kernel Launch config for pre-kernel
         * @param postConvertCfg Kernel Launch config for post-kernel
         * @param chEstAlgo The current configured channel estimate
         * @param nMaxChEstHetCfgs Max Channel estimate Cfgs
         * @param pCapture Stream capture, to obtain the Graph Node as a result of the capture
        */
        explicit TrtEngineChestGraphMgr(const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& preConvertCfg,
                                        const ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& postConvertCfg,
                                        const cuphyPuschChEstAlgoType_t                                                     chEstAlgo,
                                        const std::size_t                                                                   nMaxChEstHetCfgs,
                                        trt_engine::CaptureStreamPrePostTrtEngEnqueue*                                      pCapture) :
            m_chestNodes(preConvertCfg,
                         postConvertCfg,
                         chEstAlgo,
                         m_chEstLaunchCfgs,
                         nMaxChEstHetCfgs,
                         pCapture),
            m_chestEarlyHarqNodes(nMaxChEstHetCfgs,
                                  chEstAlgo,
                                  preConvertCfg,
                                  postConvertCfg,
                                  m_chestNodes.chEstFirstLaunchCfgs(),
                                  pCapture),
            m_chestFrontDmrsNodes(nMaxChEstHetCfgs,
                                  chEstAlgo,
                                  preConvertCfg,
                                  postConvertCfg,
                                  m_chestNodes.chEstFirstLaunchCfgs(),
                                  pCapture) {}
        [[nodiscard]] auto& asChest() { return m_chestNodes; }
        [[nodiscard]] auto& asEarlyHarq() { return m_chestEarlyHarqNodes; }
        [[nodiscard]] auto& asFrontDmrs() { return m_chestFrontDmrsNodes; }

        void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled)
        {
            m_chestNodes.setEarlyHarqModeEnabled(earlyHarqModeEnabled);
        }
        [[nodiscard]] auto getLaunchCfgs() {
            return gsl_lite::span(m_chEstLaunchCfgs);
        }

    private:
        cuphyPuschRxChEstLaunchCfgs_t m_chEstLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST]{}; // For LS
        TrtEngineChestNodeGraph       m_chestNodes;
        TrtEngineChestSubSlotNodes    m_chestEarlyHarqNodes;
        TrtEngineChestSubSlotNodes    m_chestFrontDmrsNodes;
    };

    /**
     * @class StreamChest implement Stream CUDA operations
     *        while the Nodes Chest class types are in charge
     *        of CUDA graph related  operations.
     */
    class TrtEngineChestStream final : public IChestStream {
    public:
        explicit TrtEngineChestStream(const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t> chEstLaunchCfgs,
                                      // TODO FIXME do we need this (early Harq, chestAlgo) in trt use-case?
                                      const bool earlyHarqModeEnabled,
                                      const cuphyPuschChEstAlgoType_t chEstAlgo,
                                      trt_engine::trtEngine *const pEng) :
                m_chEstLaunchCfgs(chEstLaunchCfgs),
                m_earlyHarqModeEnabled(earlyHarqModeEnabled),
                m_chEstAlgo(chEstAlgo),
                m_trtEngine(pEng) {}
        void launchKernels(cudaStream_t stream) final;
        void launchKernels0Slot(cudaStream_t stream) final;

        void launchSecondaryKernels(cudaStream_t stream) final {
            CUPHY_CHECK(m_trtEngine->run(stream));
        }
        void launchSecondaryKernels0Slot([[maybe_unused]] cudaStream_t stream) final {
            CUPHY_CHECK(m_trtEngine->run(stream));
        }
        void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) { m_earlyHarqModeEnabled = earlyHarqModeEnabled; }
    private:
        const gsl_lite::span<const cuphyPuschRxChEstLaunchCfgs_t> m_chEstLaunchCfgs{};
        bool m_earlyHarqModeEnabled{};
        cuphyPuschChEstAlgoType_t m_chEstAlgo{};
        trt_engine::trtEngine *m_trtEngine{};
    };

    // Class implementation of the channel estimation component
    class TrtEnginePuschRxChEst final : public IModule
    {
    public:
        TrtEnginePuschRxChEst(const cuphyChEstSettings& chEstSettings,
                              bool earlyHarqModeEnabled);
        TrtEnginePuschRxChEst(TrtEnginePuschRxChEst const&) = delete;
        TrtEnginePuschRxChEst& operator=(TrtEnginePuschRxChEst const&) = delete;

        // initialize channel estimator object and static component descriptor
        void init(IKernelBuilder*             pKernelBuilder,
                  bool                        enableCpuToGpuDescrAsyncCpy,
                  gsl_lite::span<uint8_t*>         ppStatDescrsCpu,
                  gsl_lite::span<uint8_t*>         ppStatDescrsGpu,
                  cudaStream_t                strm) final;

        // setup object state and dynamic component descriptor in preparation towards execution
        [[nodiscard]]
        cuphyStatus_t setup(IKernelBuilder*                       pKernelBuilder,
                            gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsCpu,
                            gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsGpu,
                            uint16_t                              nUeGrps,
                            uint8_t                               maxDmrsMaxLen,
                            uint8_t*                              pPreEarlyHarqWaitKernelStatusGpu,
                            uint8_t*                              pPostEarlyHarqWaitKernelStatusGpu,
                            uint16_t                              waitTimeOutPreEarlyHarqUs,
                            uint16_t                              waitTimeOutPostEarlyHarqUs,
                            bool                                  enableCpuToGpuDescrAsyncCpy,
                            gsl_lite::span<uint8_t*>                   ppDynDescrsCpu,
                            gsl_lite::span<uint8_t*>                   ppDynDescrsGpu,
                            uint8_t                               enableEarlyHarqProc,
                            uint8_t                               enableFrontLoadedDmrsProc,
                            uint8_t                               enableDeviceGraphLaunch,
                            CUgraphExec*                          pSubSlotDeviceGraphExec,
                            CUgraphExec*                          pFullSlotDeviceGraphExec,
                            cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
                            cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
                            cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
                            cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
                            cudaStream_t                          strm) final;

        static void getDescrInfo(size_t &statDescrSizeBytes, size_t &statDescrAlignBytes, size_t &dynDescrSizeBytes,
                                 size_t &dynDescrAlignBytes);

        void setEarlyHarqModeEnabled(bool earlyHarqModeEnabled) final;

        // Set of functions related to the graph management API/abstract interface
        // We need to have a config that has FrontLoadDMRS, EarlyHARQ marked as false.
        IChestGraphNodes& chestGraph() final { return m_chestGraphMgr->asChest(); }

        // These 2 are no-op. so they will do nothing when PuschRx will invoke it.
        IChestSubSlotNodes& earlyHarqGraph() final { return m_chestGraphMgr->asEarlyHarq(); }
        IChestSubSlotNodes& frontDmrsGraph() final { return m_chestGraphMgr->asFrontDmrs(); }
        IChestStream& chestStream() final { return *m_chestStream; }
        pusch::IStartKernels& startKernels() final { return m_startKernels; }
    private:
        // Has ctor that needs into that we dont have right now
        // maxBatchSize, input/output TensorPrms
        YAMLLoader                               m_yamlLoader;
        const cuphyChEstSettings&                m_chEstSettings;
        // m_earlyHarqModeEnabled is only used once, when creating m_chestStream
        // and once used, it will be set to nullopt so it is not used again.
        // Reason for this is the fact that we do not build m_chestStream in the ctor,
        // but instead, we create the instance in init()
        std::optional<bool>                      m_earlyHarqModeEnabled;
        std::unique_ptr<trt_engine::trtEngine>   m_trtEngine;
        std::unique_ptr<TrtEngineChestGraphMgr>  m_chestGraphMgr;
        std::unique_ptr<TrtEngineChestStream>    m_chestStream;
        pusch::StartKernels            m_startKernels;
        // Non owning, used for the sole of getGraph(), owned by trtEngine class type
        trt_engine::CaptureStreamPrePostTrtEngEnqueue* m_capturePrePost{};
        ChestPrePostEnqueueTensorConversion* m_convertPrePost{};
    };

} // namespace ch_est


#endif //CUPHY_TRTENGINE_CHEST_HPP
