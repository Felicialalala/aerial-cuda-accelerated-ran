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
#include "cumac.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cumac {

CustomUePrgScheduler::CustomUePrgScheduler() : m_cfg{}
{
}

CustomUePrgScheduler::CustomUePrgScheduler(const Config& cfg) : m_cfg(cfg)
{
}

float CustomUePrgScheduler::ueScore(uint16_t activeUeId,
                                    const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu) const
{
    const uint8_t nUeAnt = cellGrpPrmsCpu->nUeAnt;
    const float sinrLin = std::max(cellGrpPrmsCpu->wbSinr[activeUeId * nUeAnt], 1.0e-6f);
    const float avgRate = std::max(cellGrpUeStatusCpu->avgRatesActUe[activeUeId], 1.0e-6f);
    const float pfTerm = std::log2(1.0f + sinrLin) / avgRate;
    const uint32_t qBytes = activeTrafficBytes(activeUeId, cellGrpUeStatusCpu);
    const float queueTerm = std::log1p(static_cast<float>(qBytes));

    return (m_cfg.sinrWeight * std::log2(1.0f + sinrLin)) + (m_cfg.pfWeight * pfTerm) +
           (m_cfg.bufferWeight * queueTerm);
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
    if (cellGrpPrmsCpu->allocType != 1) {
        throw std::runtime_error("CustomUePrgScheduler currently supports type-1 PRG allocation only");
    }

    const uint16_t nCell = cellGrpPrmsCpu->nCell;
    const uint16_t nUe = cellGrpPrmsCpu->nUe;
    const uint16_t nActiveUe = cellGrpPrmsCpu->nActiveUe;
    const uint16_t nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    const uint8_t numUeSchdPerCellTTI = cellGrpPrmsCpu->numUeSchdPerCellTTI;

    if (nCell == 0 || numUeSchdPerCellTTI == 0) {
        throw std::runtime_error("Invalid scheduler dimensions for CustomUePrgScheduler");
    }
    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + nUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (2 * nUe), static_cast<int16_t>(-1));

    for (uint16_t cIdx = 0; cIdx < nCell; ++cIdx) {
        const uint8_t numCellSched = (cellGrpPrmsCpu->numUeSchdPerCellTTIArr != nullptr)
                                         ? cellGrpPrmsCpu->numUeSchdPerCellTTIArr[cIdx]
                                         : numUeSchdPerCellTTI;

        std::vector<SelectedUe> candidates;
        candidates.reserve(nActiveUe);

        for (uint16_t activeUeId = 0; activeUeId < nActiveUe; ++activeUeId) {
            if (cellGrpPrmsCpu->cellAssocActUe[cIdx * nActiveUe + activeUeId] == 0) {
                continue;
            }
            candidates.push_back({0xFFFF, activeUeId, ueScore(activeUeId, cellGrpUeStatusCpu, cellGrpPrmsCpu)});
        }

        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const SelectedUe& a, const SelectedUe& b) { return a.score > b.score; });

        const uint16_t chosenUe = std::min<uint16_t>(numCellSched, static_cast<uint16_t>(candidates.size()));
        std::vector<SelectedUe> scheduled;
        scheduled.reserve(chosenUe);

        for (uint16_t localIdx = 0; localIdx < numUeSchdPerCellTTI; ++localIdx) {
            const uint16_t schedSlot = static_cast<uint16_t>(cIdx * numUeSchdPerCellTTI + localIdx);

            if (localIdx < chosenUe) {
                schdSolCpu->setSchdUePerCellTTI[schedSlot] = candidates[localIdx].activeUeId;
                scheduled.push_back({schedSlot, candidates[localIdx].activeUeId, candidates[localIdx].score});
            }
        }

        if (scheduled.empty()) {
            continue;
        }

        std::vector<SelectedUe> prgTarget = scheduled;
        std::sort(prgTarget.begin(),
                  prgTarget.end(),
                  [](const SelectedUe& a, const SelectedUe& b) { return a.score > b.score; });

        const uint16_t numSel = static_cast<uint16_t>(prgTarget.size());
        const uint16_t baseAlloc = static_cast<uint16_t>(nPrbGrp / numSel);
        const uint16_t extra = static_cast<uint16_t>(nPrbGrp % numSel);

        uint16_t cursor = 0;
        for (uint16_t rank = 0; rank < numSel; ++rank) {
            const uint16_t allocLen = static_cast<uint16_t>(baseAlloc + ((rank < extra) ? 1 : 0));
            if (allocLen == 0) {
                continue;
            }

            const uint16_t schedSlot = prgTarget[rank].schedSlot;
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

} // namespace cumac
