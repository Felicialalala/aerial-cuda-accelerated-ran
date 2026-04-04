/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace cumac::online {

struct ObservationFeatureLayout {
    static constexpr uint32_t kCellFeatDim = 5U;
    static constexpr uint32_t kUeBaseFeatDim = 8U;
    static constexpr uint32_t kUeExtraFeatDim = 4U;
    static constexpr uint32_t kUeFeatDim = kUeBaseFeatDim + kUeExtraFeatDim;
    static constexpr uint32_t kEdgeFeatDim = 2U;
    static constexpr uint32_t kPrgFeatDim = 4U;
};

struct ObservationExtras {
    std::vector<float> ueExtraFeatures;
    std::vector<float> prgFeatures;

    void clear()
    {
        ueExtraFeatures.clear();
        prgFeatures.clear();
    }

    bool hasUeExtra(uint32_t nActiveUe) const
    {
        return ueExtraFeatures.size() ==
               static_cast<size_t>(nActiveUe) * ObservationFeatureLayout::kUeExtraFeatDim;
    }

    bool hasPrgFeatures(uint32_t nCell, uint32_t nPrg) const
    {
        return prgFeatures.size() ==
               static_cast<size_t>(nCell) * static_cast<size_t>(nPrg) * ObservationFeatureLayout::kPrgFeatDim;
    }
};

} // namespace cumac::online
