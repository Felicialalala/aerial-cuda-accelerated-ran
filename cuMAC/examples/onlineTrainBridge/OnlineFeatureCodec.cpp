/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnlineFeatureCodec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace cumac::online {
namespace {

float clampNonNeg(float value)
{
    return std::max(0.0F, value);
}

} // namespace

bool OnlineFeatureCodec::initialize(const cumacCellGrpPrms* cellGrpPrmsCpu)
{
    if (cellGrpPrmsCpu == nullptr) {
        return false;
    }

    m_nCell = cellGrpPrmsCpu->nCell;
    m_nActiveUe = cellGrpPrmsCpu->nActiveUe;
    m_nSchedUe = cellGrpPrmsCpu->nUe;
    m_nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    m_totNumCell = cellGrpPrmsCpu->totNumCell;
    m_allocType = cellGrpPrmsCpu->allocType;
    m_nEdges = (m_nCell > 1U) ? (m_nCell * (m_nCell - 1U)) : 0U;
    m_actionAllocLen = (m_allocType == 0U) ? (m_totNumCell * m_nPrbGrp) : (2U * m_nSchedUe);

    if (m_nCell == 0U || m_nActiveUe == 0U || m_nSchedUe == 0U || m_nPrbGrp == 0U || m_totNumCell == 0U) {
        return false;
    }

    m_edgeIndex.clear();
    m_edgeIndex.reserve(static_cast<size_t>(m_nEdges) * 2U);
    for (uint32_t src = 0; src < m_nCell; ++src) {
        for (uint32_t dst = 0; dst < m_nCell; ++dst) {
            if (src == dst) {
                continue;
            }
            m_edgeIndex.push_back(static_cast<int16_t>(src));
            m_edgeIndex.push_back(static_cast<int16_t>(dst));
        }
    }

    return true;
}

bool OnlineFeatureCodec::assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx) const
{
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        return cellGrpPrmsCpu->cellAssocActUe[cIdx * m_nActiveUe + ueIdx] != 0;
    }

    if (m_nCell > 0U && (m_nActiveUe % m_nCell) == 0U) {
        const uint32_t uePerCell = m_nActiveUe / m_nCell;
        return (ueIdx / uePerCell) == cIdx;
    }
    return false;
}

void OnlineFeatureCodec::buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                          const cumacCellGrpPrms* cellGrpPrmsCpu,
                                          std::vector<float>& cellFeatures,
                                          std::vector<float>& ueFeatures,
                                          std::vector<float>& edgeAttr) const
{
    cellFeatures.assign(static_cast<size_t>(m_nCell) * kCellFeatDim, 0.0F);
    ueFeatures.assign(static_cast<size_t>(m_nActiveUe) * kUeFeatDim, 0.0F);
    edgeAttr.assign(static_cast<size_t>(m_nEdges) * kEdgeFeatDim, 0.0F);

    std::vector<int32_t> ueServingCell(m_nActiveUe, -1);
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                if (cellGrpPrmsCpu->cellAssocActUe[cIdx * m_nActiveUe + ueIdx] != 0 && ueServingCell[ueIdx] < 0) {
                    ueServingCell[ueIdx] = static_cast<int32_t>(cIdx);
                }
            }
        }
    } else if (m_nCell > 0U && (m_nActiveUe % m_nCell) == 0U) {
        const uint32_t uePerCell = m_nActiveUe / m_nCell;
        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            ueServingCell[ueIdx] = static_cast<int32_t>(ueIdx / uePerCell);
        }
    }

    std::vector<float> cellLoadBytes(m_nCell, 0.0F);
    std::vector<float> cellAvgRateSum(m_nCell, 0.0F);
    std::vector<float> cellWbSinrLinSum(m_nCell, 0.0F);
    std::vector<uint32_t> cellUeCount(m_nCell, 0U);
    std::vector<uint32_t> cellWbSinrCount(m_nCell, 0U);
    std::vector<uint32_t> cellTbValid(m_nCell, 0U);
    std::vector<uint32_t> cellTbErr(m_nCell, 0U);

    const uint8_t nUeAnt = cellGrpPrmsCpu->nUeAnt > 0 ? cellGrpPrmsCpu->nUeAnt : 1;

    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        const int32_t cellId = ueServingCell[ueIdx];
        const float bufferBytes = (cellGrpUeStatusCpu->bufferSize != nullptr)
                                      ? static_cast<float>(cellGrpUeStatusCpu->bufferSize[ueIdx])
                                      : 0.0F;
        const float avgRateBps = (cellGrpUeStatusCpu->avgRatesActUe != nullptr)
                                     ? cellGrpUeStatusCpu->avgRatesActUe[ueIdx]
                                     : 0.0F;

        float wbSinrLin = 0.0F;
        if (cellGrpPrmsCpu->wbSinr != nullptr) {
            for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
                wbSinrLin += std::max(cellGrpPrmsCpu->wbSinr[ueIdx * nUeAnt + ant], 1.0e-9F);
            }
            wbSinrLin /= static_cast<float>(nUeAnt);
        }

        float staleSlots = 0.0F;
        if (cellId >= 0 && cellGrpUeStatusCpu->lastSchdSlotActUe != nullptr && cellGrpPrmsCpu->currSlotIdxPerCell != nullptr) {
            const uint32_t currSlot = cellGrpPrmsCpu->currSlotIdxPerCell[static_cast<uint32_t>(cellId)];
            const uint32_t lastSlot = cellGrpUeStatusCpu->lastSchdSlotActUe[ueIdx];
            if (lastSlot == 0xFFFFFFFFU) {
                staleSlots = 10000.0F;
            } else if (currSlot >= lastSlot) {
                staleSlots = static_cast<float>(currSlot - lastSlot);
            }
        }

        const float cqi = (cellGrpUeStatusCpu->cqiActUe != nullptr) ? static_cast<float>(cellGrpUeStatusCpu->cqiActUe[ueIdx]) : -1.0F;
        const float ri = (cellGrpUeStatusCpu->riActUe != nullptr) ? static_cast<float>(cellGrpUeStatusCpu->riActUe[ueIdx]) : -1.0F;
        const float tbErrAct = (cellGrpUeStatusCpu->tbErrLastActUe != nullptr)
                                   ? static_cast<float>(cellGrpUeStatusCpu->tbErrLastActUe[ueIdx])
                                   : -1.0F;
        const float newData = (cellGrpUeStatusCpu->newDataActUe != nullptr)
                                  ? static_cast<float>(cellGrpUeStatusCpu->newDataActUe[ueIdx])
                                  : -1.0F;

        const size_t base = static_cast<size_t>(ueIdx) * kUeFeatDim;
        ueFeatures[base + 0] = bufferBytes;
        ueFeatures[base + 1] = avgRateBps / 1.0e6F;
        ueFeatures[base + 2] = wbSinrLin;
        ueFeatures[base + 3] = cqi;
        ueFeatures[base + 4] = ri;
        ueFeatures[base + 5] = tbErrAct;
        ueFeatures[base + 6] = newData;
        ueFeatures[base + 7] = staleSlots;

        if (cellId >= 0 && static_cast<uint32_t>(cellId) < m_nCell) {
            const uint32_t cIdx = static_cast<uint32_t>(cellId);
            cellLoadBytes[cIdx] += bufferBytes;
            cellAvgRateSum[cIdx] += avgRateBps / 1.0e6F;
            cellUeCount[cIdx] += 1U;
            cellWbSinrLinSum[cIdx] += wbSinrLin;
            cellWbSinrCount[cIdx] += 1U;
            if (tbErrAct == 0.0F || tbErrAct == 1.0F) {
                cellTbValid[cIdx] += 1U;
                if (tbErrAct > 0.5F) {
                    cellTbErr[cIdx] += 1U;
                }
            }
        }
    }

    float totalLoad = 0.0F;
    for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
        totalLoad += cellLoadBytes[cIdx];
        const float ueCount = static_cast<float>(std::max<uint32_t>(1U, cellUeCount[cIdx]));
        const float meanWbSinrLin = cellWbSinrCount[cIdx] > 0U
                                        ? cellWbSinrLinSum[cIdx] / static_cast<float>(cellWbSinrCount[cIdx])
                                        : 0.0F;
        const float tbErrRate = cellTbValid[cIdx] > 0U
                                    ? static_cast<float>(cellTbErr[cIdx]) / static_cast<float>(cellTbValid[cIdx])
                                    : 0.0F;
        const size_t base = static_cast<size_t>(cIdx) * kCellFeatDim;
        cellFeatures[base + 0] = cellLoadBytes[cIdx];
        cellFeatures[base + 1] = static_cast<float>(cellUeCount[cIdx]);
        cellFeatures[base + 2] = meanWbSinrLin;
        cellFeatures[base + 3] = cellAvgRateSum[cIdx] / ueCount;
        cellFeatures[base + 4] = tbErrRate;
    }

    const float normLoad = std::max(totalLoad, 1.0F);
    size_t edgePos = 0U;
    for (uint32_t src = 0; src < m_nCell; ++src) {
        for (uint32_t dst = 0; dst < m_nCell; ++dst) {
            if (src == dst) {
                continue;
            }
            edgeAttr[edgePos * kEdgeFeatDim + 0] = cellLoadBytes[src] / normLoad;
            edgeAttr[edgePos * kEdgeFeatDim + 1] = (cellLoadBytes[src] - cellLoadBytes[dst]) / normLoad;
            edgePos += 1U;
        }
    }
}

void OnlineFeatureCodec::buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                         const cumacCellGrpPrms* cellGrpPrmsCpu,
                                         std::vector<uint8_t>& ueMask,
                                         std::vector<uint8_t>& cellUeMask,
                                         std::vector<uint8_t>& prgMask) const
{
    ueMask.assign(m_nActiveUe, 0U);
    cellUeMask.assign(static_cast<size_t>(m_nCell) * m_nActiveUe, 0U);
    prgMask.assign(static_cast<size_t>(m_nCell) * m_nPrbGrp, 1U);

    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        const bool hasBuffer = (cellGrpUeStatusCpu->bufferSize != nullptr)
                                   ? (cellGrpUeStatusCpu->bufferSize[ueIdx] > 0U)
                                   : true;
        bool hasAssoc = false;
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            if (!assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                continue;
            }
            hasAssoc = true;
            if (hasBuffer) {
                cellUeMask[cIdx * m_nActiveUe + ueIdx] = 1U;
            }
        }
        ueMask[ueIdx] = (hasAssoc && hasBuffer) ? 1U : 0U;
    }

    if (cellGrpPrmsCpu->prgMsk != nullptr) {
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            if (cellGrpPrmsCpu->prgMsk[cIdx] == nullptr) {
                continue;
            }
            for (uint32_t prgIdx = 0; prgIdx < m_nPrbGrp; ++prgIdx) {
                prgMask[cIdx * m_nPrbGrp + prgIdx] = (cellGrpPrmsCpu->prgMsk[cIdx][prgIdx] != 0) ? 1U : 0U;
            }
        }
    }
}

OnlineFeatureCodec::RewardTerms OnlineFeatureCodec::buildReward(const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const
{
    RewardTerms terms;
    if (cellGrpUeStatusCpu == nullptr) {
        return terms;
    }

    double sumRate = 0.0;
    double sumRateSq = 0.0;
    double sumBufferBytes = 0.0;
    uint32_t tbValid = 0U;
    uint32_t tbErr = 0U;
    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        const float rate = (cellGrpUeStatusCpu->avgRatesActUe != nullptr)
                               ? clampNonNeg(cellGrpUeStatusCpu->avgRatesActUe[ueIdx])
                               : 0.0F;
        sumRate += static_cast<double>(rate);
        sumRateSq += static_cast<double>(rate) * static_cast<double>(rate);
        if (cellGrpUeStatusCpu->bufferSize != nullptr) {
            sumBufferBytes += static_cast<double>(cellGrpUeStatusCpu->bufferSize[ueIdx]);
        }
        if (cellGrpUeStatusCpu->tbErrLastActUe != nullptr) {
            const int8_t v = cellGrpUeStatusCpu->tbErrLastActUe[ueIdx];
            if (v == 0 || v == 1) {
                tbValid += 1U;
                if (v == 1) {
                    tbErr += 1U;
                }
            }
        }
    }

    terms.throughputMbps = static_cast<float>(sumRate / 1.0e6);
    terms.totalBufferMb = static_cast<float>(sumBufferBytes / 1.0e6);
    terms.tbErrRate = (tbValid > 0U) ? static_cast<float>(tbErr) / static_cast<float>(tbValid) : 0.0F;
    terms.fairnessJain = (sumRateSq > 0.0)
                             ? static_cast<float>((sumRate * sumRate) / (static_cast<double>(m_nActiveUe) * sumRateSq))
                             : 0.0F;
    terms.scalar = terms.throughputMbps - 0.05F * terms.totalBufferMb - 2.0F * terms.tbErrRate + 0.5F * terms.fairnessJain;
    return terms;
}

} // namespace cumac::online
