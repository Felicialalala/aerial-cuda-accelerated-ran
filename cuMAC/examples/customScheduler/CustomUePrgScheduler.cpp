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

#include "CustomUePrgScheduler.h"
#include "GnnRlPolicyRuntime.h"
#include "cumac.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cumac {
namespace {

float envFloat(const char* name, float defaultValue)
{
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return defaultValue;
    }
    char* endPtr = nullptr;
    const float parsed = std::strtof(v, &endPtr);
    if (endPtr == v) {
        return defaultValue;
    }
    return parsed;
}

int envInt(const char* name, int defaultValue)
{
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return defaultValue;
    }
    char* endPtr = nullptr;
    const long parsed = std::strtol(v, &endPtr, 10);
    if (endPtr == v) {
        return defaultValue;
    }
    return static_cast<int>(parsed);
}

bool assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint16_t cellId, uint16_t activeUeId)
{
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        return cellGrpPrmsCpu->cellAssocActUe[cellId * cellGrpPrmsCpu->nActiveUe + activeUeId] != 0;
    }

    if (cellGrpPrmsCpu->nCell > 0 && (cellGrpPrmsCpu->nActiveUe % cellGrpPrmsCpu->nCell) == 0) {
        const uint16_t uePerCell = static_cast<uint16_t>(cellGrpPrmsCpu->nActiveUe / cellGrpPrmsCpu->nCell);
        return (activeUeId / uePerCell) == cellId;
    }
    return false;
}

} // namespace

CustomUePrgScheduler::CustomUePrgScheduler() : m_cfg(loadConfigFromEnv())
{
}

CustomUePrgScheduler::CustomUePrgScheduler(const Config& cfg) : m_cfg(cfg)
{
}

CustomUePrgScheduler::~CustomUePrgScheduler() = default;

CustomUePrgScheduler::Config CustomUePrgScheduler::loadConfigFromEnv()
{
    Config cfg;
    const char* policyEnv = std::getenv("CUMAC_CUSTOM_POLICY");
    if (policyEnv != nullptr && policyEnv[0] != '\0') {
        std::string p(policyEnv);
        std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (p == "legacy" || p == "pf") {
            cfg.policyMode = PolicyMode::Legacy;
        } else if (p == "gnnrl_model" || p == "model") {
            cfg.policyMode = PolicyMode::GnnRlModel;
        } else {
            cfg.policyMode = PolicyMode::GnnRlHeuristic;
        }
    }

    cfg.sinrWeight = envFloat("CUMAC_CUSTOM_SINR_WEIGHT", cfg.sinrWeight);
    cfg.pfWeight = envFloat("CUMAC_CUSTOM_PF_WEIGHT", cfg.pfWeight);
    cfg.bufferWeight = envFloat("CUMAC_CUSTOM_BUFFER_WEIGHT", cfg.bufferWeight);
    cfg.reliabilityWeight = envFloat("CUMAC_CUSTOM_RELIABILITY_WEIGHT", cfg.reliabilityWeight);
    cfg.staleWeight = envFloat("CUMAC_CUSTOM_STALE_WEIGHT", cfg.staleWeight);
    cfg.cellContextWeight = envFloat("CUMAC_CUSTOM_CELL_CONTEXT_WEIGHT", cfg.cellContextWeight);
    cfg.prgSinrWeight = envFloat("CUMAC_CUSTOM_PRG_SINR_WEIGHT", cfg.prgSinrWeight);
    cfg.prgPfWeight = envFloat("CUMAC_CUSTOM_PRG_PF_WEIGHT", cfg.prgPfWeight);
    cfg.prgBufferWeight = envFloat("CUMAC_CUSTOM_PRG_BUFFER_WEIGHT", cfg.prgBufferWeight);
    cfg.noTxThreshold = envFloat("CUMAC_CUSTOM_NO_TX_THRESHOLD", cfg.noTxThreshold);
    cfg.maxActiveCellsPerPrg =
        static_cast<uint16_t>(std::max(0, envInt("CUMAC_CUSTOM_MAX_ACTIVE_CELLS_PER_PRG", cfg.maxActiveCellsPerPrg)));
    const char* modelPath = std::getenv("CUMAC_GNNRL_MODEL_PATH");
    if (modelPath != nullptr && modelPath[0] != '\0') {
        cfg.modelPath = std::string(modelPath);
    }
    cfg.policyTimeoutMs = std::max(0, envInt("CUMAC_POLICY_TIMEOUT_MS", cfg.policyTimeoutMs));
    return cfg;
}

float CustomUePrgScheduler::ueScore(uint16_t activeUeId,
                                    uint16_t cellId,
                                    float cellContext,
                                    const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu) const
{
    const uint8_t nUeAnt = cellGrpPrmsCpu->nUeAnt;
    float sinrLin = 1.0e-6f;
    if (cellGrpPrmsCpu->wbSinr != nullptr && nUeAnt > 0) {
        float sinrSum = 0.0f;
        for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
            sinrSum += std::max(cellGrpPrmsCpu->wbSinr[activeUeId * nUeAnt + ant], 1.0e-6f);
        }
        sinrLin = std::max(sinrSum / static_cast<float>(nUeAnt), 1.0e-6f);
    }

    const float avgRate = std::max(cellGrpUeStatusCpu->avgRatesActUe[activeUeId], 1.0e-6f);
    const float seTerm = std::log2(1.0f + sinrLin);
    const float pfTerm = seTerm / avgRate;
    const float queueTerm = std::log1p(static_cast<float>(activeTrafficBytes(activeUeId, cellGrpUeStatusCpu)));

    if (m_cfg.policyMode == PolicyMode::Legacy) {
        return (m_cfg.sinrWeight * seTerm) + (m_cfg.pfWeight * pfTerm) + (m_cfg.bufferWeight * queueTerm);
    }

    float reliability = 0.0f;
    if (cellGrpUeStatusCpu->tbErrLastActUe != nullptr) {
        const int8_t tbErr = cellGrpUeStatusCpu->tbErrLastActUe[activeUeId];
        if (tbErr == 0) {
            reliability = 1.0f;
        } else if (tbErr == 1) {
            reliability = -1.0f;
        }
    }

    float staleBoost = 0.0f;
    if (cellGrpUeStatusCpu->lastSchdSlotActUe != nullptr && cellGrpPrmsCpu->currSlotIdxPerCell != nullptr &&
        cellId < cellGrpPrmsCpu->nCell) {
        const uint32_t currSlot = cellGrpPrmsCpu->currSlotIdxPerCell[cellId];
        const uint32_t lastSlot = cellGrpUeStatusCpu->lastSchdSlotActUe[activeUeId];
        if (lastSlot == 0xFFFFFFFF) {
            staleBoost = 1.0f;
        } else if (currSlot >= lastSlot) {
            staleBoost = std::min(1.0f, static_cast<float>(currSlot - lastSlot) / 200.0f);
        }
    }

    return (m_cfg.sinrWeight * seTerm) + (m_cfg.pfWeight * pfTerm) + (m_cfg.bufferWeight * queueTerm) +
           (m_cfg.reliabilityWeight * reliability) + (m_cfg.staleWeight * staleBoost) +
           (m_cfg.cellContextWeight * cellContext);
}

float CustomUePrgScheduler::prgScore(uint16_t activeUeId,
                                     uint16_t prgIdx,
                                     uint16_t cellId,
                                     float cellContext,
                                     float prgUtilRatio,
                                     const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                     const cumacCellGrpPrms* cellGrpPrmsCpu) const
{
    const uint8_t nUeAnt = cellGrpPrmsCpu->nUeAnt;
    float prgSinrLin = 1.0e-6f;

    if (cellGrpPrmsCpu->postEqSinr != nullptr && nUeAnt > 0) {
        float prgSinrSum = 0.0f;
        const size_t base = static_cast<size_t>(activeUeId) * cellGrpPrmsCpu->nPrbGrp * nUeAnt +
                            static_cast<size_t>(prgIdx) * nUeAnt;
        for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
            prgSinrSum += std::max(cellGrpPrmsCpu->postEqSinr[base + ant], 1.0e-6f);
        }
        prgSinrLin = std::max(prgSinrSum / static_cast<float>(nUeAnt), 1.0e-6f);
    } else if (cellGrpPrmsCpu->wbSinr != nullptr && nUeAnt > 0) {
        float wbSinrSum = 0.0f;
        for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
            wbSinrSum += std::max(cellGrpPrmsCpu->wbSinr[activeUeId * nUeAnt + ant], 1.0e-6f);
        }
        prgSinrLin = std::max(wbSinrSum / static_cast<float>(nUeAnt), 1.0e-6f);
    }

    const float avgRate = std::max(cellGrpUeStatusCpu->avgRatesActUe[activeUeId], 1.0e-6f);
    const float seTerm = std::log2(1.0f + prgSinrLin);
    const float pfTerm = seTerm / avgRate;
    const float queueTerm = std::log1p(static_cast<float>(activeTrafficBytes(activeUeId, cellGrpUeStatusCpu)));

    float score = (m_cfg.prgSinrWeight * seTerm) + (m_cfg.prgPfWeight * pfTerm) + (m_cfg.prgBufferWeight * queueTerm);
    if (m_cfg.policyMode == PolicyMode::Legacy) {
        return score;
    }

    float reliability = 0.0f;
    if (cellGrpUeStatusCpu->tbErrLastActUe != nullptr) {
        const int8_t tbErr = cellGrpUeStatusCpu->tbErrLastActUe[activeUeId];
        if (tbErr == 0) {
            reliability = 1.0f;
        } else if (tbErr == 1) {
            reliability = -1.0f;
        }
    }

    const float sparseBonus = 1.0f - std::min(1.0f, std::max(0.0f, prgUtilRatio));
    score += (m_cfg.cellContextWeight * cellContext) + (m_cfg.reliabilityWeight * reliability) + (0.10f * sparseBonus);
    if (cellId >= cellGrpPrmsCpu->nCell) {
        return score;
    }
    return score;
}

std::vector<float> CustomUePrgScheduler::buildCellContext(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                                          const cumacCellGrpPrms* cellGrpPrmsCpu) const
{
    const uint16_t nCell = cellGrpPrmsCpu->nCell;
    const uint16_t nActiveUe = cellGrpPrmsCpu->nActiveUe;
    std::vector<float> cellLoad(nCell, 0.0f);
    std::vector<float> cellErrRate(nCell, 0.0f);

    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        float errCnt = 0.0f;
        float tbCnt = 0.0f;
        for (uint16_t activeUeId = 0; activeUeId < nActiveUe; ++activeUeId) {
            if (!assocToCell(cellGrpPrmsCpu, cIdx, activeUeId)) {
                continue;
            }
            if (cellGrpUeStatusCpu->bufferSize != nullptr) {
                cellLoad[cIdx] += static_cast<float>(cellGrpUeStatusCpu->bufferSize[activeUeId]);
            }
            if (cellGrpUeStatusCpu->tbErrLastActUe != nullptr) {
                const int8_t tbErr = cellGrpUeStatusCpu->tbErrLastActUe[activeUeId];
                if (tbErr == 0 || tbErr == 1) {
                    tbCnt += 1.0f;
                    errCnt += (tbErr == 1) ? 1.0f : 0.0f;
                }
            }
        }
        cellErrRate[cIdx] = tbCnt > 0.0f ? (errCnt / tbCnt) : 0.0f;
    }

    const float sumLoad = std::accumulate(cellLoad.begin(), cellLoad.end(), 0.0f);
    const float meanLoad = nCell > 0 ? (sumLoad / static_cast<float>(nCell)) : 0.0f;

    std::vector<float> ctx(nCell, 0.0f);
    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        const float loadNorm = sumLoad > 0.0f ? (cellLoad[cIdx] / sumLoad) : (1.0f / std::max<uint16_t>(1, nCell));
        const float neighborLoad = nCell > 1 ? (sumLoad - cellLoad[cIdx]) / static_cast<float>(nCell - 1) : cellLoad[cIdx];
        const float neighborPressure = meanLoad > 1.0e-6f ? (neighborLoad / meanLoad) : 1.0f;
        ctx[cIdx] = (0.60f * loadNorm) + (0.25f * neighborPressure) + (0.15f * cellErrRate[cIdx]);
    }
    return ctx;
}

std::vector<std::vector<CustomUePrgScheduler::SelectedUe>> CustomUePrgScheduler::buildCellCandidates(
    const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
    const cumacCellGrpPrms* cellGrpPrmsCpu,
    const std::vector<float>& cellContext) const
{
    const uint16_t nCell = cellGrpPrmsCpu->nCell;
    const uint16_t nActiveUe = cellGrpPrmsCpu->nActiveUe;
    std::vector<std::vector<SelectedUe>> allCandidates(nCell);

    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        std::vector<SelectedUe>& candidates = allCandidates[cIdx];
        candidates.reserve(nActiveUe);

        for (uint16_t activeUeId = 0; activeUeId < nActiveUe; ++activeUeId) {
            if (!assocToCell(cellGrpPrmsCpu, cIdx, activeUeId)) {
                continue;
            }
            candidates.push_back(
                {0xFFFF, activeUeId, ueScore(activeUeId, cIdx, cellContext[cIdx], cellGrpUeStatusCpu, cellGrpPrmsCpu)});
        }

        std::sort(candidates.begin(), candidates.end(), [](const SelectedUe& a, const SelectedUe& b) {
            return a.score > b.score;
        });
    }
    return allCandidates;
}

void CustomUePrgScheduler::runType0(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    cumacSchdSol* schdSolCpu,
                                    cumacCellGrpPrms* cellGrpPrmsCpu,
                                    cumacSchdSol* schdSolGpu,
                                    cudaStream_t stream) const
{
    const uint16_t nCell = cellGrpPrmsCpu->nCell;
    const uint16_t totNumCell = cellGrpPrmsCpu->totNumCell;
    const uint16_t nUe = cellGrpPrmsCpu->nUe;
    const uint16_t nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    const uint8_t numUeSchdPerCellTTI = cellGrpPrmsCpu->numUeSchdPerCellTTI;

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + nUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (totNumCell * nPrbGrp), static_cast<int16_t>(-1));

    const std::vector<float> cellContext = buildCellContext(cellGrpUeStatusCpu, cellGrpPrmsCpu);
    std::vector<std::vector<SelectedUe>> candidates = buildCellCandidates(cellGrpUeStatusCpu, cellGrpPrmsCpu, cellContext);
    std::vector<std::vector<SelectedUe>> scheduledByCell(nCell);

    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        const uint8_t numCellSchedRaw = (cellGrpPrmsCpu->numUeSchdPerCellTTIArr != nullptr)
                                            ? cellGrpPrmsCpu->numUeSchdPerCellTTIArr[cIdx]
                                            : numUeSchdPerCellTTI;
        const uint8_t numCellSched = std::min<uint8_t>(numCellSchedRaw, numUeSchdPerCellTTI);
        const uint16_t chosenUe = std::min<uint16_t>(numCellSched, static_cast<uint16_t>(candidates[cIdx].size()));
        scheduledByCell[cIdx].reserve(chosenUe);

        for (uint16_t localIdx = 0; localIdx < numUeSchdPerCellTTI; ++localIdx) {
            const uint16_t schedSlot = static_cast<uint16_t>(cIdx * numUeSchdPerCellTTI + localIdx);
            if (localIdx < chosenUe) {
                SelectedUe selected = candidates[cIdx][localIdx];
                selected.schedSlot = schedSlot;
                schdSolCpu->setSchdUePerCellTTI[schedSlot] = selected.activeUeId;
                scheduledByCell[cIdx].push_back(selected);
            }
        }
    }

    std::vector<uint16_t> orderedCells(nCell, 0);
    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        orderedCells[cIdx] = cIdx;
    }
    std::sort(orderedCells.begin(), orderedCells.end(), [&](uint16_t a, uint16_t b) {
        if (cellContext[a] == cellContext[b]) {
            return a < b;
        }
        return cellContext[a] > cellContext[b];
    });

    for (uint16_t prgIdx = 0; prgIdx < nPrbGrp; ++prgIdx) {
        uint16_t activeCells = 0;
        const uint16_t rot = nCell > 0 ? static_cast<uint16_t>(prgIdx % nCell) : 0;

        for (uint16_t rank = 0; rank < nCell; ++rank) {
            const uint16_t ordIdx = nCell > 0 ? static_cast<uint16_t>((rank + rot) % nCell) : 0;
            const uint16_t cIdx = orderedCells[ordIdx];
            const std::vector<SelectedUe>& scheduled = scheduledByCell[cIdx];
            if (scheduled.empty()) {
                continue;
            }

            float bestScore = std::numeric_limits<float>::lowest();
            int16_t bestSlot = -1;
            const float prgUtilRatio = nCell > 0 ? static_cast<float>(activeCells) / static_cast<float>(nCell) : 0.0f;
            for (const SelectedUe& ue : scheduled) {
                const float score = prgScore(
                    ue.activeUeId, prgIdx, cIdx, cellContext[cIdx], prgUtilRatio, cellGrpUeStatusCpu, cellGrpPrmsCpu);
                if (score > bestScore) {
                    bestScore = score;
                    bestSlot = static_cast<int16_t>(ue.schedSlot);
                }
            }

            const bool hitMaxActive = (m_cfg.maxActiveCellsPerPrg > 0) && (activeCells >= m_cfg.maxActiveCellsPerPrg);
            if (bestSlot < 0 || bestScore < m_cfg.noTxThreshold || hitMaxActive) {
                continue;
            }
            schdSolCpu->allocSol[prgIdx * totNumCell + cIdx] = bestSlot;
            activeCells = static_cast<uint16_t>(activeCells + 1);
        }
    }

    const size_t ueSelBytes = static_cast<size_t>(nUe) * sizeof(uint16_t);
    const size_t allocBytes = static_cast<size_t>(totNumCell) * nPrbGrp * sizeof(int16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->allocSol, schdSolCpu->allocSol, allocBytes, cudaMemcpyHostToDevice, stream));
}

void CustomUePrgScheduler::runType1(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    cumacSchdSol* schdSolCpu,
                                    cumacCellGrpPrms* cellGrpPrmsCpu,
                                    cumacSchdSol* schdSolGpu,
                                    cudaStream_t stream) const
{
    const uint16_t nCell = cellGrpPrmsCpu->nCell;
    const uint16_t nUe = cellGrpPrmsCpu->nUe;
    const uint16_t nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    const uint8_t numUeSchdPerCellTTI = cellGrpPrmsCpu->numUeSchdPerCellTTI;

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + nUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (2 * nUe), static_cast<int16_t>(-1));

    const std::vector<float> cellContext = buildCellContext(cellGrpUeStatusCpu, cellGrpPrmsCpu);
    std::vector<std::vector<SelectedUe>> candidates = buildCellCandidates(cellGrpUeStatusCpu, cellGrpPrmsCpu, cellContext);

    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        const uint8_t numCellSchedRaw = (cellGrpPrmsCpu->numUeSchdPerCellTTIArr != nullptr)
                                            ? cellGrpPrmsCpu->numUeSchdPerCellTTIArr[cIdx]
                                            : numUeSchdPerCellTTI;
        const uint8_t numCellSched = std::min<uint8_t>(numCellSchedRaw, numUeSchdPerCellTTI);
        const uint16_t chosenUe = std::min<uint16_t>(numCellSched, static_cast<uint16_t>(candidates[cIdx].size()));

        std::vector<SelectedUe> scheduled;
        scheduled.reserve(chosenUe);
        for (uint16_t localIdx = 0; localIdx < numUeSchdPerCellTTI; ++localIdx) {
            const uint16_t schedSlot = static_cast<uint16_t>(cIdx * numUeSchdPerCellTTI + localIdx);
            if (localIdx < chosenUe) {
                SelectedUe selected = candidates[cIdx][localIdx];
                selected.schedSlot = schedSlot;
                schdSolCpu->setSchdUePerCellTTI[schedSlot] = selected.activeUeId;
                scheduled.push_back(selected);
            }
        }
        if (scheduled.empty()) {
            continue;
        }

        std::sort(scheduled.begin(), scheduled.end(), [](const SelectedUe& a, const SelectedUe& b) { return a.score > b.score; });
        const uint16_t numSel = static_cast<uint16_t>(scheduled.size());
        const uint16_t baseAlloc = static_cast<uint16_t>(nPrbGrp / numSel);
        const uint16_t extra = static_cast<uint16_t>(nPrbGrp % numSel);

        uint16_t cursor = 0;
        for (uint16_t rank = 0; rank < numSel; ++rank) {
            const uint16_t allocLen = static_cast<uint16_t>(baseAlloc + ((rank < extra) ? 1 : 0));
            if (allocLen == 0) {
                continue;
            }
            const uint16_t schedSlot = scheduled[rank].schedSlot;
            schdSolCpu->allocSol[2 * schedSlot] = static_cast<int16_t>(cursor);
            schdSolCpu->allocSol[2 * schedSlot + 1] = static_cast<int16_t>(cursor + allocLen);
            cursor = static_cast<uint16_t>(cursor + allocLen);
        }
    }

    const size_t ueSelBytes = static_cast<size_t>(nUe) * sizeof(uint16_t);
    const size_t allocBytes = static_cast<size_t>(2 * nUe) * sizeof(int16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->allocSol, schdSolCpu->allocSol, allocBytes, cudaMemcpyHostToDevice, stream));
}

uint32_t CustomUePrgScheduler::activeTrafficBytes(uint16_t activeUeId,
                                                  const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const
{
    if (cellGrpUeStatusCpu->bufferSize == nullptr) {
        return 0;
    }
    return cellGrpUeStatusCpu->bufferSize[activeUeId];
}

void CustomUePrgScheduler::run(cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                               cumacSchdSol* schdSolCpu,
                               cumacCellGrpPrms* cellGrpPrmsCpu,
                               cumacSchdSol* schdSolGpu,
                               cudaStream_t stream) const
{
    if (cellGrpPrmsCpu == nullptr || cellGrpUeStatusCpu == nullptr || schdSolCpu == nullptr || schdSolGpu == nullptr) {
        throw std::runtime_error("CustomUePrgScheduler got null input pointers");
    }
    if (cellGrpPrmsCpu->nCell == 0 || cellGrpPrmsCpu->numUeSchdPerCellTTI == 0 || cellGrpPrmsCpu->nPrbGrp == 0) {
        throw std::runtime_error("Invalid scheduler dimensions for CustomUePrgScheduler");
    }

    static bool logOnce = false;
    if (!logOnce) {
        const char* policyStr = "gnnrl";
        if (m_cfg.policyMode == PolicyMode::Legacy) {
            policyStr = "legacy";
        } else if (m_cfg.policyMode == PolicyMode::GnnRlModel) {
            policyStr = "gnnrl_model";
        }
        std::cout << "CustomUePrgScheduler config: policy=" << policyStr
                  << " noTxThreshold=" << m_cfg.noTxThreshold
                  << " maxActiveCellsPerPrg=" << m_cfg.maxActiveCellsPerPrg
                  << " modelPath=" << (m_cfg.modelPath.empty() ? "<empty>" : m_cfg.modelPath)
                  << " policyTimeoutMs=" << m_cfg.policyTimeoutMs
                  << std::endl;
        logOnce = true;
    }

    if (m_cfg.policyMode == PolicyMode::GnnRlModel) {
        if (cellGrpPrmsCpu->allocType == 0) {
            if (!m_modelInitTried) {
                m_modelInitTried = true;
                GnnRlPolicyRuntime::Config runtimeCfg;
                runtimeCfg.modelPath = m_cfg.modelPath;
                runtimeCfg.timeoutMs = m_cfg.policyTimeoutMs;
                m_modelRuntime = std::make_unique<GnnRlPolicyRuntime>(runtimeCfg);
                m_modelReady = (m_modelRuntime != nullptr) && m_modelRuntime->initialize(cellGrpPrmsCpu);
                if (!m_modelReady) {
                    std::cerr << "[GNNRL_MODEL] runtime init failed; fallback to legacy policy\n";
                }
            }

            if (m_modelReady && m_modelRuntime != nullptr) {
                const bool ok = m_modelRuntime->inferType0(
                    cellGrpUeStatusCpu, cellGrpPrmsCpu, schdSolCpu, schdSolGpu, stream);
                if (ok) {
                    return;
                }
                std::cerr << "[GNNRL_MODEL] inference failed; fallback to legacy policy\n";
            }
        } else {
            std::cerr << "[GNNRL_MODEL] allocType!=0 not supported; fallback to legacy policy\n";
        }

        Config fallbackCfg = m_cfg;
        fallbackCfg.policyMode = PolicyMode::Legacy;
        CustomUePrgScheduler fallbackSchd(fallbackCfg);
        if (cellGrpPrmsCpu->allocType == 0) {
            fallbackSchd.runType0(cellGrpUeStatusCpu, schdSolCpu, cellGrpPrmsCpu, schdSolGpu, stream);
        } else if (cellGrpPrmsCpu->allocType == 1) {
            fallbackSchd.runType1(cellGrpUeStatusCpu, schdSolCpu, cellGrpPrmsCpu, schdSolGpu, stream);
        } else {
            throw std::runtime_error("CustomUePrgScheduler supports allocType 0/1 only");
        }
        return;
    }

    if (cellGrpPrmsCpu->allocType == 0) {
        runType0(cellGrpUeStatusCpu, schdSolCpu, cellGrpPrmsCpu, schdSolGpu, stream);
    } else if (cellGrpPrmsCpu->allocType == 1) {
        runType1(cellGrpUeStatusCpu, schdSolCpu, cellGrpPrmsCpu, schdSolGpu, stream);
    } else {
        throw std::runtime_error("CustomUePrgScheduler supports allocType 0/1 only");
    }
}

} // namespace cumac
