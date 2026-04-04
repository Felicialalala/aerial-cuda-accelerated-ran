/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OnlineFeatureCodec.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace cumac::online {
namespace {

float clampNonNeg(float value)
{
    return std::max(0.0F, value);
}

std::string toLowerCopy(const std::string& value)
{
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

} // namespace

OnlineFeatureCodec::RewardMode OnlineFeatureCodec::parseRewardMode(const std::string& value)
{
    const std::string lower = toLowerCopy(value);
    if (lower.empty() || lower == "goodput_only") {
        return RewardMode::GoodputOnly;
    }
    if (lower == "legacy" || lower == "legacy_throughput_backlog") {
        return RewardMode::LegacyThroughputBacklog;
    }
    if (lower == "goodput_soft_queue") {
        return RewardMode::GoodputSoftQueue;
    }
    if (lower == "goodput_reliability" || lower == "goodput_ttl_bler") {
        return RewardMode::GoodputReliability;
    }
    return RewardMode::GoodputOnly;
}

const char* OnlineFeatureCodec::rewardModeToString(const RewardMode mode)
{
    switch (mode) {
        case RewardMode::LegacyThroughputBacklog:
            return "legacy";
        case RewardMode::GoodputSoftQueue:
            return "goodput_soft_queue";
        case RewardMode::GoodputReliability:
            return "goodput_reliability";
        case RewardMode::GoodputOnly:
        default:
            return "goodput_only";
    }
}

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
                                          const ObservationExtras* extras,
                                          std::vector<float>& cellFeatures,
                                          std::vector<float>& ueFeatures,
                                          std::vector<float>& prgFeatures,
                                          std::vector<float>& edgeAttr) const
{
    cellFeatures.assign(static_cast<size_t>(m_nCell) * kCellFeatDim, 0.0F);
    ueFeatures.assign(static_cast<size_t>(m_nActiveUe) * kUeFeatDim, 0.0F);
    prgFeatures.assign(static_cast<size_t>(m_nCell) * static_cast<size_t>(m_nPrbGrp) * kPrgFeatDim, 0.0F);
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
        if (extras != nullptr && extras->hasUeExtra(m_nActiveUe)) {
            const size_t extraBase = static_cast<size_t>(ueIdx) * ObservationFeatureLayout::kUeExtraFeatDim;
            ueFeatures[base + 8] = extras->ueExtraFeatures[extraBase + 0U];
            ueFeatures[base + 9] = extras->ueExtraFeatures[extraBase + 1U];
            ueFeatures[base + 10] = extras->ueExtraFeatures[extraBase + 2U];
            ueFeatures[base + 11] = extras->ueExtraFeatures[extraBase + 3U];
        }

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

    if (extras != nullptr && extras->hasPrgFeatures(m_nCell, m_nPrbGrp)) {
        prgFeatures = extras->prgFeatures;
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
            // Keep cellUeMask as pure association so PRG-only Type-0 mode can
            // reconstruct the native "all active UE slots" baseline exactly.
            cellUeMask[cIdx * m_nActiveUe + ueIdx] = 1U;
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

OnlineFeatureCodec::RewardTerms OnlineFeatureCodec::buildReward(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                                                const std::vector<unsigned long long>& servedBytesThisTti,
                                                                const std::vector<unsigned long long>& goodputBytesThisTti,
                                                                unsigned long long flowQueuedBytes,
                                                                unsigned long long expiredBytesThisTti,
                                                                unsigned long long expiredPacketsThisTti,
                                                                float slotDurationSec,
                                                                float tbErrRate,
                                                                float schedWbSinrDb,
                                                                float prgUtilizationRatio,
                                                                float goodputSpectralEfficiencyBpsHz,
                                                                float prgReuseRatio,
                                                                float expiryDropRate) const
{
    RewardTerms terms;
    if (cellGrpUeStatusCpu == nullptr) {
        return terms;
    }

    double sumBufferBytes = 0.0;
    double sumServedBytes = 0.0;
    double sumGoodputBytes = 0.0;
    double sumGoodputBytesSq = 0.0;
    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        if (cellGrpUeStatusCpu->bufferSize != nullptr) {
            sumBufferBytes += static_cast<double>(cellGrpUeStatusCpu->bufferSize[ueIdx]);
        }
        if (ueIdx < servedBytesThisTti.size()) {
            sumServedBytes += static_cast<double>(servedBytesThisTti[ueIdx]);
        }
        if (ueIdx < goodputBytesThisTti.size()) {
            const double goodputBytes = static_cast<double>(goodputBytesThisTti[ueIdx]);
            sumGoodputBytes += goodputBytes;
            sumGoodputBytesSq += goodputBytes * goodputBytes;
        }
    }

    const double totalPendingBytes = sumBufferBytes + static_cast<double>(flowQueuedBytes);
    const double slotDuration = std::max(1.0e-6, static_cast<double>(slotDurationSec));
    terms.throughputMbps = static_cast<float>((sumServedBytes * 8.0) / slotDuration / 1.0e6);
    terms.goodputMbps = static_cast<float>((sumGoodputBytes * 8.0) / slotDuration / 1.0e6);
    terms.totalBufferMb = static_cast<float>(totalPendingBytes / 1.0e6);
    terms.tbErrRate = std::max(0.0F, std::min(1.0F, tbErrRate));
    terms.schedWbSinrDb = schedWbSinrDb;
    terms.prgUtilizationRatio = std::max(0.0F, std::min(1.0F, prgUtilizationRatio));
    terms.goodputSpectralEfficiencyBpsHz = std::max(0.0F, goodputSpectralEfficiencyBpsHz);
    terms.prgReuseRatio = std::max(0.0F, std::min(1.0F, prgReuseRatio));
    terms.expiredBytes = clampNonNeg(static_cast<float>(expiredBytesThisTti));
    terms.expiredPackets = clampNonNeg(static_cast<float>(expiredPacketsThisTti));
    terms.expiryDropRate = std::max(0.0F, std::min(1.0F, expiryDropRate));
    terms.fairnessJain = (sumGoodputBytesSq > 0.0)
                             ? static_cast<float>(
                                   (sumGoodputBytes * sumGoodputBytes) /
                                   (static_cast<double>(std::max<uint32_t>(1U, m_nActiveUe)) * sumGoodputBytesSq))
                             : 0.0F;

    switch (m_rewardMode) {
        case RewardMode::LegacyThroughputBacklog:
            if (terms.throughputMbps > 1.0e-6F) {
                terms.queueDelayMs = static_cast<float>(
                    (totalPendingBytes * 8.0) / (static_cast<double>(terms.throughputMbps) * 1.0e6) * 1.0e3);
            } else if (totalPendingBytes > 0.0) {
                terms.queueDelayMs = 5000.0F;
            } else {
                terms.queueDelayMs = 0.0F;
            }
            terms.scalar = 0.05F * terms.throughputMbps - 0.05F * terms.totalBufferMb - 0.002F * terms.queueDelayMs
                           - 1.5F * terms.tbErrRate + 0.5F * terms.fairnessJain;
            break;
        case RewardMode::GoodputSoftQueue:
            terms.scalar = 0.05F * terms.goodputMbps - 0.5F * std::log1p(std::max(0.0F, terms.totalBufferMb));
            break;
        case RewardMode::GoodputReliability:
            terms.scalar = 0.05F * terms.goodputMbps - 2.0F * terms.tbErrRate - 4.0F * terms.expiryDropRate
                           + 0.25F * terms.fairnessJain;
            break;
        case RewardMode::GoodputOnly:
        default:
            terms.scalar = 0.05F * terms.goodputMbps;
            break;
    }
    return terms;
}

} // namespace cumac::online
