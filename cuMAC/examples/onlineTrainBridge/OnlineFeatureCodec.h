/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "api.h"
#include "OnlineObservationTypes.h"

namespace cumac::online {

class OnlineFeatureCodec {
public:
    static constexpr uint32_t kCellFeatDim = ObservationFeatureLayout::kCellFeatDim;
    static constexpr uint32_t kUeFeatDim = ObservationFeatureLayout::kUeFeatDim;
    static constexpr uint32_t kEdgeFeatDim = ObservationFeatureLayout::kEdgeFeatDim;
    static constexpr uint32_t kPrgFeatDim = ObservationFeatureLayout::kPrgFeatDim;

    enum class RewardMode {
        LegacyThroughputBacklog = 0,
        GoodputOnly,
        GoodputSoftQueue,
        GoodputReliability,
    };

    struct RewardTerms {
        float scalar = 0.0F;
        float throughputMbps = 0.0F;
        float goodputMbps = 0.0F;
        float totalBufferMb = 0.0F;
        float tbErrRate = 0.0F;
        float fairnessJain = 0.0F;
        float queueDelayMs = 0.0F;
        float schedWbSinrDb = 0.0F;
        float prgUtilizationRatio = 0.0F;
        float goodputSpectralEfficiencyBpsHz = 0.0F;
        float prgReuseRatio = 0.0F;
        float expiredBytes = 0.0F;
        float expiredPackets = 0.0F;
        float expiryDropRate = 0.0F;
    };

    bool initialize(const cumacCellGrpPrms* cellGrpPrmsCpu);
    void setRewardMode(RewardMode mode) { m_rewardMode = mode; }
    RewardMode rewardMode() const { return m_rewardMode; }
    static RewardMode parseRewardMode(const std::string& value);
    static const char* rewardModeToString(RewardMode mode);

    uint32_t nCell() const { return m_nCell; }
    uint32_t nActiveUe() const { return m_nActiveUe; }
    uint32_t nSchedUe() const { return m_nSchedUe; }
    uint32_t nPrg() const { return m_nPrbGrp; }
    uint32_t nTotCell() const { return m_totNumCell; }
    uint32_t nEdges() const { return m_nEdges; }
    uint32_t allocType() const { return m_allocType; }
    uint32_t actionAllocLen() const { return m_actionAllocLen; }
    uint32_t cellFeatDim() const { return kCellFeatDim; }
    uint32_t ueFeatDim() const { return kUeFeatDim; }
    uint32_t edgeFeatDim() const { return kEdgeFeatDim; }
    uint32_t prgFeatDim() const { return kPrgFeatDim; }

    const std::vector<int16_t>& edgeIndex() const { return m_edgeIndex; }

    void buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu,
                          const ObservationExtras* extras,
                          std::vector<float>& cellFeatures,
                          std::vector<float>& ueFeatures,
                          std::vector<float>& prgFeatures,
                          std::vector<float>& edgeAttr) const;

    void buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                         const cumacCellGrpPrms* cellGrpPrmsCpu,
                         std::vector<uint8_t>& ueMask,
                         std::vector<uint8_t>& cellUeMask,
                         std::vector<uint8_t>& prgMask) const;

    RewardTerms buildReward(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
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
                            float expiryDropRate) const;

private:
    uint32_t m_nCell = 0U;
    uint32_t m_nActiveUe = 0U;
    uint32_t m_nSchedUe = 0U;
    uint32_t m_nPrbGrp = 0U;
    uint32_t m_totNumCell = 0U;
    uint32_t m_allocType = 0U;
    uint32_t m_nEdges = 0U;
    uint32_t m_actionAllocLen = 0U;
    RewardMode m_rewardMode = RewardMode::GoodputOnly;

    std::vector<int16_t> m_edgeIndex;

    bool assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx) const;
};

} // namespace cumac::online
