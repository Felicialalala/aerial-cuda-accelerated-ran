/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "OnlineObservationTypes.h"
#include "api.h"
#include "../trafficModel/trafficService.hpp"

namespace cumac::online {

class ObservationExtrasBuilder {
public:
    bool initialize(const cumacCellGrpPrms* cellGrpPrmsCpu)
    {
        if (cellGrpPrmsCpu == nullptr) {
            return false;
        }
        m_nCell = cellGrpPrmsCpu->nCell;
        m_nActiveUe = cellGrpPrmsCpu->nActiveUe;
        m_nPrg = cellGrpPrmsCpu->nPrbGrp;
        if (m_nCell == 0U || m_nActiveUe == 0U || m_nPrg == 0U) {
            return false;
        }
        m_recentSchedRatio.assign(m_nActiveUe, 0.0f);
        m_recentGoodputMbps.assign(m_nActiveUe, 0.0f);
        m_prevPrgAssigned.assign(static_cast<size_t>(m_nCell) * m_nPrg, 0U);
        m_snapshot.ueExtraFeatures.assign(
            static_cast<size_t>(m_nActiveUe) * ObservationFeatureLayout::kUeExtraFeatDim, 0.0f);
        m_snapshot.prgFeatures.assign(
            static_cast<size_t>(m_nCell) * m_nPrg * ObservationFeatureLayout::kPrgFeatDim, 0.0f);
        return true;
    }

    void setSlotDurationMs(const float valueMs)
    {
        if (valueMs > 0.0f) {
            m_slotDurationMs = valueMs;
        }
    }

    void setPacketTtlTti(const int valueTti)
    {
        m_packetTtlTti = std::max(0, valueTti);
    }

    const ObservationExtras& buildSnapshot(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                           const cumacCellGrpPrms* cellGrpPrmsCpu,
                                           const TrafficService* trafSvc)
    {
        const size_t ueFeatSize =
            static_cast<size_t>(m_nActiveUe) * ObservationFeatureLayout::kUeExtraFeatDim;
        const size_t prgFeatSize =
            static_cast<size_t>(m_nCell) * static_cast<size_t>(m_nPrg) * ObservationFeatureLayout::kPrgFeatDim;
        if (m_snapshot.ueExtraFeatures.size() != ueFeatSize) {
            m_snapshot.ueExtraFeatures.assign(ueFeatSize, 0.0f);
        } else {
            std::fill(m_snapshot.ueExtraFeatures.begin(), m_snapshot.ueExtraFeatures.end(), 0.0f);
        }
        if (m_snapshot.prgFeatures.size() != prgFeatSize) {
            m_snapshot.prgFeatures.assign(prgFeatSize, 0.0f);
        } else {
            std::fill(m_snapshot.prgFeatures.begin(), m_snapshot.prgFeatures.end(), 0.0f);
        }

        if (cellGrpUeStatusCpu == nullptr || cellGrpPrmsCpu == nullptr || m_nCell == 0U || m_nActiveUe == 0U || m_nPrg == 0U) {
            return m_snapshot;
        }

        std::vector<int32_t> ueServingCell(m_nActiveUe, -1);
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                if (assocToCell(cellGrpPrmsCpu, cIdx, ueIdx) && ueServingCell[ueIdx] < 0) {
                    ueServingCell[ueIdx] = static_cast<int32_t>(cIdx);
                }
            }
        }

        std::vector<PacketHeadSummary> packetHead;
        if (trafSvc != nullptr) {
            trafSvc->GetPacketHeadStats(packetHead);
        }

        std::vector<float> cellRecentGoodputMean(m_nCell, 0.0f);
        std::vector<uint32_t> cellRecentGoodputCount(m_nCell, 0U);
        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            const int32_t cellId = ueServingCell[ueIdx];
            if (cellId >= 0 && static_cast<uint32_t>(cellId) < m_nCell) {
                cellRecentGoodputMean[static_cast<uint32_t>(cellId)] += m_recentGoodputMbps[ueIdx];
                cellRecentGoodputCount[static_cast<uint32_t>(cellId)] += 1U;
            }
        }
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            if (cellRecentGoodputCount[cIdx] > 0U) {
                cellRecentGoodputMean[cIdx] /= static_cast<float>(cellRecentGoodputCount[cIdx]);
            }
        }

        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            const size_t base = static_cast<size_t>(ueIdx) * ObservationFeatureLayout::kUeExtraFeatDim;
            float holDelayMs = 0.0f;
            float ttlSlackMs = -1.0f;
            if (ueIdx < packetHead.size()) {
                if (packetHead[ueIdx].hol_age_tti > 0) {
                    holDelayMs = static_cast<float>(packetHead[ueIdx].hol_age_tti) * m_slotDurationMs;
                }
                if (packetHead[ueIdx].ttl_slack_tti >= 0 && m_packetTtlTti > 0) {
                    ttlSlackMs = static_cast<float>(packetHead[ueIdx].ttl_slack_tti) * m_slotDurationMs;
                }
            }
            const int32_t cellId = ueServingCell[ueIdx];
            float recentDeficitNorm = 0.0f;
            if (cellId >= 0 && static_cast<uint32_t>(cellId) < m_nCell) {
                const float cellMean = cellRecentGoodputMean[static_cast<uint32_t>(cellId)];
                if (cellMean > 1.0f) {
                    recentDeficitNorm = std::max(0.0f, cellMean - m_recentGoodputMbps[ueIdx]) / cellMean;
                }
            }
            m_snapshot.ueExtraFeatures[base + 0U] = holDelayMs;
            m_snapshot.ueExtraFeatures[base + 1U] = ttlSlackMs;
            m_snapshot.ueExtraFeatures[base + 2U] = std::max(0.0f, std::min(1.0f, m_recentSchedRatio[ueIdx]));
            m_snapshot.ueExtraFeatures[base + 3U] = recentDeficitNorm;
        }

        const uint8_t nUeAnt = std::max<uint8_t>(1U, cellGrpPrmsCpu->nUeAnt);
        std::vector<float> ueWbSinrDb(m_nActiveUe, -20.0f);
        if (cellGrpPrmsCpu->wbSinr != nullptr) {
            for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                float wbSinrLin = 0.0f;
                for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
                    wbSinrLin += std::max(cellGrpPrmsCpu->wbSinr[ueIdx * nUeAnt + ant], 1.0e-9f);
                }
                wbSinrLin /= static_cast<float>(nUeAnt);
                ueWbSinrDb[ueIdx] = 10.0f * std::log10(std::max(wbSinrLin, 1.0e-9f));
            }
        }

        const size_t prgCellCount = static_cast<size_t>(m_nCell) * static_cast<size_t>(m_nPrg);
        std::vector<float> top1PerCellPrgDb(prgCellCount, -20.0f);
        std::vector<float> top2GapPerCellPrgDb(prgCellCount, 0.0f);
        std::vector<float> top1WinnerWbSinrDb(prgCellCount, -20.0f);
        std::vector<uint8_t> top1ValidPerCellPrg(prgCellCount, 0U);

        for (uint32_t prgIdx = 0; prgIdx < m_nPrg; ++prgIdx) {
            uint32_t reuseCount = 0U;
            for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
                if (m_prevPrgAssigned[cIdx * m_nPrg + prgIdx] != 0U) {
                    reuseCount += 1U;
                }
            }
            const float reuseRatio =
                m_nCell > 0U ? static_cast<float>(reuseCount) / static_cast<float>(m_nCell) : 0.0f;
            for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
                float top1SinrDb = -20.0f;
                float top2SinrDb = -20.0f;
                uint32_t top1UeIdx = 0U;
                bool any = false;
                bool secondFound = false;
                for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                    if (!assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                        continue;
                    }
                    float prgSinrLin = 0.0f;
                    if (cellGrpPrmsCpu->postEqSinr != nullptr) {
                        const size_t base = static_cast<size_t>(ueIdx) * m_nPrg * nUeAnt +
                                            static_cast<size_t>(prgIdx) * nUeAnt;
                        for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
                            prgSinrLin += std::max(cellGrpPrmsCpu->postEqSinr[base + ant], 1.0e-9f);
                        }
                        prgSinrLin /= static_cast<float>(nUeAnt);
                    } else if (cellGrpPrmsCpu->wbSinr != nullptr) {
                        for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
                            prgSinrLin += std::max(cellGrpPrmsCpu->wbSinr[ueIdx * nUeAnt + ant], 1.0e-9f);
                        }
                        prgSinrLin /= static_cast<float>(nUeAnt);
                    }
                    const float sinrDb = 10.0f * std::log10(std::max(prgSinrLin, 1.0e-9f));
                    if (!any || sinrDb > top1SinrDb) {
                        top2SinrDb = top1SinrDb;
                        secondFound = any;
                        top1SinrDb = sinrDb;
                        top1UeIdx = ueIdx;
                        any = true;
                    } else if (sinrDb > top2SinrDb) {
                        top2SinrDb = sinrDb;
                        secondFound = true;
                    }
                }
                const float gapDb = (any && secondFound) ? std::max(0.0f, top1SinrDb - top2SinrDb) : 0.0f;
                const size_t idx = static_cast<size_t>(cIdx) * static_cast<size_t>(m_nPrg) + prgIdx;
                top1PerCellPrgDb[idx] = top1SinrDb;
                top2GapPerCellPrgDb[idx] = gapDb;
                top1ValidPerCellPrg[idx] = any ? 1U : 0U;
                if (any && top1UeIdx < ueWbSinrDb.size()) {
                    top1WinnerWbSinrDb[idx] = ueWbSinrDb[top1UeIdx];
                }
            }

            constexpr float kConflictMarginDb = 3.0f;
            constexpr float kSinrDropScaleDb = 12.0f;
            for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
                const size_t idx = static_cast<size_t>(cIdx) * static_cast<size_t>(m_nPrg) + prgIdx;
                const float top1SinrDb = top1PerCellPrgDb[idx];
                const float gapDb = top2GapPerCellPrgDb[idx];
                const bool any = top1ValidPerCellPrg[idx] != 0U;
                float neighborMaxTop1SinrDb = -20.0f;
                float neighborMeanTop1SinrDb = -20.0f;
                float neighborTop1SumDb = 0.0f;
                uint32_t neighborTop1Count = 0U;
                uint32_t conflictCount = 0U;
                for (uint32_t otherCIdx = 0; otherCIdx < m_nCell; ++otherCIdx) {
                    if (otherCIdx == cIdx) {
                        continue;
                    }
                    const size_t otherIdx = static_cast<size_t>(otherCIdx) * static_cast<size_t>(m_nPrg) + prgIdx;
                    if (top1ValidPerCellPrg[otherIdx] == 0U) {
                        continue;
                    }
                    const float otherTop1SinrDb = top1PerCellPrgDb[otherIdx];
                    neighborMaxTop1SinrDb = std::max(neighborMaxTop1SinrDb, otherTop1SinrDb);
                    neighborTop1SumDb += otherTop1SinrDb;
                    neighborTop1Count += 1U;
                    if (any && otherTop1SinrDb >= (top1SinrDb - kConflictMarginDb)) {
                        conflictCount += 1U;
                    }
                }
                if (neighborTop1Count > 0U) {
                    neighborMeanTop1SinrDb = neighborTop1SumDb / static_cast<float>(neighborTop1Count);
                }
                const float samePrgConflictRatio =
                    (m_nCell > 1U) ? static_cast<float>(conflictCount) / static_cast<float>(m_nCell - 1U) : 0.0f;
                float prgSinrDropNorm = 0.0f;
                if (any) {
                    const float sinrDropDb = std::max(0.0f, top1WinnerWbSinrDb[idx] - top1SinrDb);
                    prgSinrDropNorm = std::min(1.0f, sinrDropDb / kSinrDropScaleDb);
                }
                const float iciProxy = 0.5f * samePrgConflictRatio + 0.5f * prgSinrDropNorm;
                const size_t base = (static_cast<size_t>(cIdx) * m_nPrg + prgIdx) *
                                    ObservationFeatureLayout::kPrgFeatDim;
                m_snapshot.prgFeatures[base + 0U] = top1SinrDb;
                m_snapshot.prgFeatures[base + 1U] = gapDb;
                m_snapshot.prgFeatures[base + 2U] = (m_prevPrgAssigned[cIdx * m_nPrg + prgIdx] != 0U) ? 1.0f : 0.0f;
                m_snapshot.prgFeatures[base + 3U] = reuseRatio;
                m_snapshot.prgFeatures[base + 4U] = neighborMaxTop1SinrDb;
                m_snapshot.prgFeatures[base + 5U] = neighborMeanTop1SinrDb;
                m_snapshot.prgFeatures[base + 6U] = std::max(0.0f, std::min(1.0f, samePrgConflictRatio));
                m_snapshot.prgFeatures[base + 7U] = std::max(0.0f, std::min(1.0f, iciProxy));
            }
        }

        return m_snapshot;
    }

    void observeTtiResult(const cumacSchdSol* schdSolCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu,
                          const std::vector<unsigned long long>& goodputBytesThisTti)
    {
        if (cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr || m_nCell == 0U || m_nActiveUe == 0U || m_nPrg == 0U) {
            return;
        }

        std::vector<uint8_t> scheduledThisTti(m_nActiveUe, 0U);
        std::fill(m_prevPrgAssigned.begin(), m_prevPrgAssigned.end(), 0U);

        if (schdSolCpu->allocSol != nullptr && schdSolCpu->setSchdUePerCellTTI != nullptr) {
            for (uint32_t prgIdx = 0; prgIdx < m_nPrg; ++prgIdx) {
                for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
                    const int16_t schedSlot = schdSolCpu->allocSol[prgIdx * cellGrpPrmsCpu->totNumCell + cIdx];
                    if (schedSlot < 0 || static_cast<uint32_t>(schedSlot) >= cellGrpPrmsCpu->nUe) {
                        continue;
                    }
                    const uint16_t activeUeId = schdSolCpu->setSchdUePerCellTTI[static_cast<uint32_t>(schedSlot)];
                    if (activeUeId == 0xFFFF || activeUeId >= m_nActiveUe) {
                        continue;
                    }
                    scheduledThisTti[activeUeId] = 1U;
                    m_prevPrgAssigned[cIdx * m_nPrg + prgIdx] = 1U;
                }
            }
        }

        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            const float scheduledFlag = (scheduledThisTti[ueIdx] != 0U) ? 1.0f : 0.0f;
            const float goodputMbps =
                (ueIdx < goodputBytesThisTti.size() && m_slotDurationMs > 0.0f)
                    ? static_cast<float>((static_cast<double>(goodputBytesThisTti[ueIdx]) * 8.0) /
                                         (static_cast<double>(m_slotDurationMs) * 1.0e3))
                    : 0.0f;
            m_recentSchedRatio[ueIdx] =
                (1.0f - m_recentAlpha) * m_recentSchedRatio[ueIdx] + m_recentAlpha * scheduledFlag;
            m_recentGoodputMbps[ueIdx] =
                (1.0f - m_recentAlpha) * m_recentGoodputMbps[ueIdx] + m_recentAlpha * goodputMbps;
        }
    }

private:
    static bool assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx)
    {
        if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
            return cellGrpPrmsCpu->cellAssocActUe[cIdx * cellGrpPrmsCpu->nActiveUe + ueIdx] != 0;
        }
        if (cellGrpPrmsCpu->nCell > 0U && (cellGrpPrmsCpu->nActiveUe % cellGrpPrmsCpu->nCell) == 0U) {
            const uint32_t uePerCell = cellGrpPrmsCpu->nActiveUe / cellGrpPrmsCpu->nCell;
            return (ueIdx / uePerCell) == cIdx;
        }
        return false;
    }

    uint32_t m_nCell = 0U;
    uint32_t m_nActiveUe = 0U;
    uint32_t m_nPrg = 0U;
    float m_slotDurationMs = 0.5f;
    int m_packetTtlTti = 0;
    float m_recentAlpha = 0.05f;
    std::vector<float> m_recentSchedRatio;
    std::vector<float> m_recentGoodputMbps;
    std::vector<uint8_t> m_prevPrgAssigned;
    ObservationExtras m_snapshot;
};

} // namespace cumac::online
