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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "api.h"
#include "GnnRlPolicyRuntime.h"

namespace cumac {
class CustomUePrgScheduler {
public:
    enum class PolicyMode : uint8_t {
        Legacy = 0,
        GnnRlHeuristic = 1,
        GnnRlModel = 2,
    };

    struct Config {
        PolicyMode policyMode = PolicyMode::GnnRlHeuristic;
        float sinrWeight = 1.0f;
        float pfWeight = 0.25f;
        float bufferWeight = 0.05f;
        float reliabilityWeight = 0.15f;
        float staleWeight = 0.05f;
        float cellContextWeight = 0.20f;
        float prgSinrWeight = 0.85f;
        float prgPfWeight = 0.25f;
        float prgBufferWeight = 0.10f;
        float noTxThreshold = -0.05f;
        uint16_t maxActiveCellsPerPrg = 0; // 0 means no limit
        std::string modelPath;
        int policyTimeoutMs = 0;
        // gnnrl_model decode stabilization knobs.
        float modelNoUeBias = 0.0f;
        float modelMinSchedRatio = 1.0f;
        float modelNoPrgBias = 0.0f;
        float modelMinPrgRatio = 1.0f;
        float modelMaxPrgSharePerUe = 1.0f;
    };

    CustomUePrgScheduler();
    explicit CustomUePrgScheduler(const Config& cfg);
    ~CustomUePrgScheduler();

    void run(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
             cumacSchdSol* schdSolCpu,
             cumacCellGrpPrms* cellGrpPrmsCpu,
             cumacSchdSol* schdSolGpu,
             cudaStream_t stream) const;

private:
    struct SelectedUe {
        uint16_t schedSlot;
        uint16_t activeUeId;
        float score;
    };

    Config m_cfg;
    mutable bool m_modelInitTried = false;
    mutable bool m_modelReady = false;
    mutable std::unique_ptr<GnnRlPolicyRuntime> m_modelRuntime = nullptr;

    static Config loadConfigFromEnv();

    void runType0(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                  cumacSchdSol* schdSolCpu,
                  cumacCellGrpPrms* cellGrpPrmsCpu,
                  cumacSchdSol* schdSolGpu,
                  cudaStream_t stream) const;

    void runType1(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                  cumacSchdSol* schdSolCpu,
                  cumacCellGrpPrms* cellGrpPrmsCpu,
                  cumacSchdSol* schdSolGpu,
                  cudaStream_t stream) const;

    float ueScore(uint16_t activeUeId,
                  uint16_t cellId,
                  float cellContext,
                  const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                  const cumacCellGrpPrms* cellGrpPrmsCpu) const;

    float prgScore(uint16_t activeUeId,
                   uint16_t prgIdx,
                   uint16_t cellId,
                   float cellContext,
                   float prgUtilRatio,
                   const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                   const cumacCellGrpPrms* cellGrpPrmsCpu) const;

    std::vector<float> buildCellContext(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                        const cumacCellGrpPrms* cellGrpPrmsCpu) const;

    std::vector<std::vector<SelectedUe>> buildCellCandidates(
        const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
        const cumacCellGrpPrms* cellGrpPrmsCpu,
        const std::vector<float>& cellContext) const;

    uint32_t activeTrafficBytes(uint16_t activeUeId,
                                const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const;
};

} // namespace cumac
