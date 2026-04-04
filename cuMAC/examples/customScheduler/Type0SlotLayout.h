/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace cumac {

struct Type0SlotLayout {
    std::vector<uint32_t> slotToCell;
    std::vector<uint32_t> slotLocalIdx;
    std::vector<uint8_t> slotValid;
    std::vector<uint32_t> slotCountPerCell;
    std::vector<uint32_t> cellSlotStart;
    uint32_t totalValidSlots = 0U;

    bool validSlot(uint32_t slot) const
    {
        return slot < slotValid.size() && slotValid[slot] != 0U;
    }

    bool slotBelongsToCell(uint32_t slot, uint32_t cell) const
    {
        return validSlot(slot) && slotToCell[slot] == cell;
    }
};

template <typename AssocFn>
inline Type0SlotLayout buildType0SlotLayout(uint32_t nCell, uint32_t nActiveUe, uint32_t nSchedUe, AssocFn&& assocFn)
{
    Type0SlotLayout layout;
    layout.slotToCell.assign(nSchedUe, 0U);
    layout.slotLocalIdx.assign(nSchedUe, 0U);
    layout.slotValid.assign(nSchedUe, 0U);
    layout.slotCountPerCell.assign(nCell, 0U);
    layout.cellSlotStart.assign(nCell, 0U);

    std::vector<uint32_t> assocCounts(nCell, 0U);
    for (uint32_t cIdx = 0; cIdx < nCell; ++cIdx) {
        for (uint32_t ueIdx = 0; ueIdx < nActiveUe; ++ueIdx) {
            if (assocFn(cIdx, ueIdx)) {
                assocCounts[cIdx] += 1U;
            }
        }
    }

    uint32_t clippedEnd = 0U;
    for (uint32_t cIdx = 0; cIdx < nCell; ++cIdx) {
        layout.cellSlotStart[cIdx] = clippedEnd;
        const uint32_t unclippedEnd = clippedEnd + assocCounts[cIdx];
        const uint32_t nextClippedEnd = std::min(unclippedEnd, nSchedUe);
        layout.slotCountPerCell[cIdx] = nextClippedEnd - clippedEnd;
        clippedEnd = nextClippedEnd;
    }

    uint32_t cursor = 0U;
    for (uint32_t cIdx = 0; cIdx < nCell; ++cIdx) {
        for (uint32_t localIdx = 0; localIdx < layout.slotCountPerCell[cIdx] && cursor < nSchedUe; ++localIdx, ++cursor) {
            layout.slotToCell[cursor] = cIdx;
            layout.slotLocalIdx[cursor] = localIdx;
            layout.slotValid[cursor] = 1U;
        }
    }
    layout.totalValidSlots = cursor;
    return layout;
}

} // namespace cumac
