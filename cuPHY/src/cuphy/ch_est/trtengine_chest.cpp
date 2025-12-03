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

#include "trtengine_chest.hpp"
#include "ch_est_types.hpp"
#include "cuphy_utils.hpp"

namespace {
void dump(const gsl_lite::span<const trt_engine::TrtParams> prms) {
    for (const auto& e : prms) {
        NVLOGD_FMT(NVLOG_PUSCH, "name: {}, datatype: {}, ndims: {}, actual dims:",
                   e.name, static_cast<std::underlying_type_t<decltype(e.params.dataType)>>(e.params.dataType),
                   +e.params.nDims);
        for(const auto& d : e.params.dims) {
            NVLOGD_FMT(NVLOG_PUSCH, "{}, ", d);
        }
    }
}

void addSecondaryKernelNodeToGraphImpl(CUgraph                                        graph,
                                       std::vector<CUgraphNode>&                      currNodeDeps,
                                       std::vector<CUgraphNode>&                      nextNodeDeps,
                                       CUDA_KERNEL_NODE_PARAMS&                       nodeParams,
                                       const gsl_lite::span<CUgraphNode>                   secondNodes,
                                       trt_engine::CaptureStreamPrePostTrtEngEnqueue* pCapture)
{
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&secondNodes[0], graph,
        currNodeDeps.data(), currNodeDeps.size(), &nodeParams));
    nextNodeDeps.push_back(secondNodes[0]);
    currNodeDeps.clear();
    currNodeDeps        = nextNodeDeps;
    auto* capturedGraph = pCapture->getGraph();
    CUDA_CHECK_EXCEPTION(cudaGraphAddChildGraphNode(&secondNodes[1],
        graph, currNodeDeps.data(),
        currNodeDeps.size(), capturedGraph));
    nextNodeDeps.clear();
    nextNodeDeps.push_back(secondNodes[1]);
    currNodeDeps.clear();
    currNodeDeps = nextNodeDeps;
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&secondNodes[2], graph,
        currNodeDeps.data(), currNodeDeps.size(), &nodeParams));
    nextNodeDeps.clear();
    nextNodeDeps.push_back(secondNodes[2]);
}

void setSecondaryNodeStatusImpl(ch_est::ChestCudaUtils::DisableAllNodes                                                     disableAllNodes,
                                CUgraphExec                                                                                 graphExec,
                                const gsl_lite::span<CUgraphNode>                                                                secondNodes,
                                std::vector<uint8_t>&                                                                       secondNodesEnabled,
                                const ch_est::ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& preConvertCfg,
                                const ch_est::ChestPrePostEnqueueTensorConversion::puschPrePostTensorConversionLaunchCfg_t& postConvertCfg)
{
    if (disableAllNodes == ch_est::ChestCudaUtils::DisableAllNodes::FALSE)
    {
        // for pre
        if(secondNodesEnabled[0] != 1)
        {
            secondNodesEnabled[0] = 1;
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, secondNodes[0], 1));
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(graphExec, secondNodes[0], &preConvertCfg.kernelNodeParamsDriver));

        // no need for the middle one - captured graph from trt
        if(secondNodesEnabled[1] != 1)
        {
            secondNodesEnabled[1] = 1;
        }

        // for post
        if(secondNodesEnabled[2] != 1)
        {
            secondNodesEnabled[2] = 1;
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, secondNodes[2], 1));
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(graphExec, secondNodes[2], &postConvertCfg.kernelNodeParamsDriver));
    } else
    {

        if(secondNodesEnabled[0] != 0)
        {
            secondNodesEnabled[0] = 0;
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, secondNodes[0], 0));
        }

        if(secondNodesEnabled[1] != 0)
        {
            secondNodesEnabled[1] = 0;
        }

        if(secondNodesEnabled[2] != 0)
        {
            secondNodesEnabled[2] = 0;
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, secondNodes[2], 0));
        }
    }
}

} // anonymous

namespace ch_est {

    // FIXME this is copied from ch_est.cu, it's static and needs to be in a more accessible, non-member one.
    void TrtEnginePuschRxChEst::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes,
                                             size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes) {
        statDescrSizeBytes  = sizeof(puschRxChEstStatDescr_t);
        statDescrAlignBytes = alignof(puschRxChEstStatDescr_t);

        dynDescrSizeBytes  = sizeof(puschRxChEstDynDescrVec_t);
        dynDescrAlignBytes = alignof(puschRxChEstDynDescrVec_t);
    }

    void TrtEngineChestStream::launchKernels(cudaStream_t stream) {
        const uint32_t startChEstInstIdx = (m_earlyHarqModeEnabled && m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ? 1 : 0;
        launchKernelsImpl(stream, m_chEstLaunchCfgs, startChEstInstIdx);
        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            launchSecondaryKernels(stream);
        }
    }

    void TrtEngineChestStream::launchKernels0Slot(cudaStream_t stream) {
        launchKernels0SlotImpl(stream, m_chEstLaunchCfgs);
        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            launchSecondaryKernels0Slot(stream);
        }
    }

    TrtEnginePuschRxChEst::TrtEnginePuschRxChEst(const cuphyChEstSettings &chEstSettings,
                                                 const bool earlyHarqModeEnabled) : m_chEstSettings(
            chEstSettings), m_earlyHarqModeEnabled(earlyHarqModeEnabled) {}

    void TrtEnginePuschRxChEst::setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) {
        m_chestGraphMgr->setEarlyHarqModeEnabled(earlyHarqModeEnabled);
        m_chestStream->setEarlyHarqModeEnabled(earlyHarqModeEnabled);
    }

    // Sub-slot

    void TrtEngineChestSubSlotNodes::addKernelNodeToGraph(CUgraph graph,
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

    void TrtEngineChestSubSlotNodes::addSecondaryKernelNodeToGraph(CUgraph graph,
                                                          std::vector<CUgraphNode> &currNodeDeps,
                                                          std::vector<CUgraphNode> &nextNodeDeps,
                                                          CUDA_KERNEL_NODE_PARAMS &nodeParams) {
        addSecondaryKernelNodeToGraphImpl(graph, currNodeDeps, nextNodeDeps, nodeParams, m_secondNodes, m_pCapture);
    }

    void TrtEngineChestSubSlotNodes::setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                          CUgraphExec graphExec) {
        const int nCfgs = disableAllNodes == ChestCudaUtils::DisableAllNodes::TRUE ? 0 : m_cfgs0Slot.nCfgs;
        cuphy_utils::setHetCfgNodeStatus(nCfgs, m_nMaxChEstHetCfgs, m_cfgs0Slot.cfgs,
                                         m_nodesEnabled, m_nodes, graphExec);
        if(m_chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        {
            setSecondaryNodeStatus(disableAllNodes,
                                   graphExec);
        }
    }

    void TrtEngineChestSubSlotNodes::setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes,
                                                            CUgraphExec                     graphExec) {
        setSecondaryNodeStatusImpl(disableAllNodes, graphExec, m_secondNodes, m_secondNodesEnabled, m_preConvertCfg, m_postConvertCfg);
    }

    // Full slot

    void TrtEngineChestNodeGraph::addKernelNodeToGraph(CUgraph graph,
                                  std::vector<CUgraphNode> &currNodeDeps,
                                  std::vector<CUgraphNode> &nextNodeDeps,
                                  CUDA_KERNEL_NODE_PARAMS &nodeParams)
    {
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

    void TrtEngineChestNodeGraph::addSecondaryKernelNodeToGraph(CUgraph graph,
                                   std::vector<CUgraphNode> &currNodeDeps,
                                   std::vector<CUgraphNode> &nextNodeDeps,
                                   CUDA_KERNEL_NODE_PARAMS &nodeParams)
    {
        addSecondaryKernelNodeToGraphImpl(graph, currNodeDeps, nextNodeDeps, nodeParams, m_chEstSecondNodes, m_pCapture);
    }

    void TrtEngineChestNodeGraph::setNodeStatus(ch_est::ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec)
    {
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

    void TrtEngineChestNodeGraph::setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec)
    {
        setSecondaryNodeStatusImpl(disableAllNodes, graphExec, m_chEstSecondNodes, m_chEstSecondNodesEnabled, m_preConvertCfg, m_postConvertCfg);
    }

    void TrtEngineChestNodeGraph::disableNodes0Slot(CUgraphExec graph) {
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

    void TrtEnginePuschRxChEst::init(IKernelBuilder*     pKernelBuilder,
                                     bool                enableCpuToGpuDescrAsyncCpy,
                                     gsl_lite::span<uint8_t*> ppStatDescrsCpu,
                                     gsl_lite::span<uint8_t*> ppStatDescrsGpu,
                                     cudaStream_t        strm)
    {
        CUPHY_CHECK(m_yamlLoader.load(m_chEstSettings.puschrxChestFactorySettingsFilename.value_or("")));
        NVLOGD_FMT(NVLOG_PUSCH, "inputs: ");
        const auto& inputs = m_yamlLoader.getInputs();
        dump(inputs);
        const auto& outputs = m_yamlLoader.getOutputs();
        NVLOGD_FMT(NVLOG_PUSCH, "outputs: ");
        dump(outputs);
        NVLOGC_FMT(NVLOG_PUSCH, "Model filename: {}", m_yamlLoader.getModelFilename());
        NVLOGC_FMT(NVLOG_PUSCH, "Model max batch size: {}", m_yamlLoader.getMaxBatchSize());
        auto capturePrePost = std::make_unique<trt_engine::CaptureStreamPrePostTrtEngEnqueue>();
        m_capturePrePost    = capturePrePost.get();
        auto convertPrePost = std::make_unique<ChestPrePostEnqueueTensorConversion>();
        m_convertPrePost    = convertPrePost.get();
        m_trtEngine         = std::make_unique<trt_engine::trtEngine>(m_yamlLoader.getMaxBatchSize(),
                                                                      std::vector(inputs.cbegin(), inputs.cend()),
                                                                      std::vector(outputs.cbegin(), outputs.cend()),
                                                                      std::move(capturePrePost),
                                                                      std::move(convertPrePost));
        for(auto* cpuDesc : ppStatDescrsCpu)
        {
            auto& statDescrCpu           = *reinterpret_cast<puschRxChEstStatDescr_t*>(cpuDesc);
            statDescrCpu.pSymbolRxStatus = m_chEstSettings.pSymbolRxStatus;
        }
        pKernelBuilder->init(ppStatDescrsCpu, ppStatDescrsGpu, enableCpuToGpuDescrAsyncCpy, strm);
        CUPHY_CHECK(m_trtEngine->init(m_yamlLoader.getModelFilename().data()));
        CUPHY_CHECK(m_trtEngine->warmup(strm));
        m_chestGraphMgr = std::make_unique<TrtEngineChestGraphMgr>(m_convertPrePost->getPreCfg(),
                                                                   m_convertPrePost->getPostCfg(),
                                                                   m_chEstSettings.chEstAlgo,
                                                                   m_chEstSettings.nMaxChEstHetCfgs,
                                                                   m_capturePrePost);
        m_chestStream = std::make_unique<TrtEngineChestStream>(m_chestGraphMgr->getLaunchCfgs(), *m_earlyHarqModeEnabled, m_chEstSettings.chEstAlgo, m_trtEngine.get());
        m_earlyHarqModeEnabled.reset();
    }

    cuphyStatus_t TrtEnginePuschRxChEst::setup(IKernelBuilder*                       pKernelBuilder,
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
                                               cudaStream_t                          strm) {
        if (!pKernelBuilder) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Invalid nullptr pKernelBuilder");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        const auto ret = pKernelBuilder->build(pDrvdUeGrpPrmsCpu,
                                               pDrvdUeGrpPrmsGpu,
                                               nUeGrps,
                                               maxDmrsMaxLen,
                                               m_chEstSettings.enableDftSOfdm,
                                               m_chEstSettings.chEstAlgo,
                                               m_chEstSettings.enableMassiveMIMO,
                                               m_chEstSettings.enablePerPrgChEst,
                                               pPreEarlyHarqWaitKernelStatusGpu,
                                               pPostEarlyHarqWaitKernelStatusGpu,
                                               waitTimeOutPreEarlyHarqUs,
                                               waitTimeOutPostEarlyHarqUs,
                                               enableCpuToGpuDescrAsyncCpy,
                                               ppDynDescrsCpu,
                                               ppDynDescrsGpu,
                                               &m_startKernels,
                                               m_chestGraphMgr->getLaunchCfgs(),
                                               enableEarlyHarqProc,
                                               enableFrontLoadedDmrsProc,
                                               enableDeviceGraphLaunch,
                                               pSubSlotDeviceGraphExec,
                                               pFullSlotDeviceGraphExec,
                                               pWaitKernelLaunchCfgsPreSubSlot,
                                               pWaitKernelLaunchCfgsPostSubSlot,
                                               pDglKernelLaunchCfgsPreSubSlot,
                                               pDglKernelLaunchCfgsPostSubSlot,
                                               strm);
        if (ret != CUPHY_STATUS_SUCCESS) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "TrtEnginePuschRxChEst failed to run KernelBuilder::build");
            return ret;
        }

        // Assuming only supporting single UE for now.
        if (nUeGrps > 1) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,
                       "TrtEnginePuschRxChEst do not support multiple UEs. Only single UE. Inputs to Setup was provided with {} UEs",
                       pDrvdUeGrpPrmsCpu.size());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        gsl_Expects(not pDrvdUeGrpPrmsCpu.empty());
        gsl_Expects(pDrvdUeGrpPrmsCpu.size() == pDrvdUeGrpPrmsGpu.size());
        // New setup overload now pass this.
        return m_trtEngine->setup(pDrvdUeGrpPrmsCpu, pDrvdUeGrpPrmsGpu, strm);
    }

} // namespace ch_est
