/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "api.h"
#include "Type0SlotLayout.h"

namespace cumac {

struct Type0RejectDiagnostics {
    std::vector<int32_t> rejectWrongCellSlotCount;
    std::vector<int32_t> rejectEmptySlotCount;
    std::vector<int32_t> rejectPrgMaskCount;
    std::vector<int32_t> rejectOobSlotCount;
    std::vector<int32_t> rejectDuplicateUeCount;
    std::vector<int32_t> nativeSlotCountPerCell;
    std::vector<int32_t> nativeCellSlotStart;
    std::vector<int32_t> nativeSlotToCell;
    std::vector<int32_t> nativeSlotValid;

    void reset(uint32_t nCell)
    {
        rejectWrongCellSlotCount.assign(nCell, 0);
        rejectEmptySlotCount.assign(nCell, 0);
        rejectPrgMaskCount.assign(nCell, 0);
        rejectOobSlotCount.assign(nCell, 0);
        rejectDuplicateUeCount.assign(nCell, 0);
        nativeSlotCountPerCell.clear();
        nativeCellSlotStart.clear();
        nativeSlotToCell.clear();
        nativeSlotValid.clear();
    }
};

template <typename AssocFn>
inline void applyType0SampledActionToSchedule(const std::vector<int32_t>& ueAction,
                                              const std::vector<int16_t>& prgAction,
                                              const std::vector<uint8_t>& ueMask,
                                              const Type0SlotLayout& slotLayout,
                                              uint32_t nCell,
                                              uint32_t nActiveUe,
                                              uint32_t nSchedUe,
                                              uint32_t nPrg,
                                              uint32_t nTotCell,
                                              const cumacCellGrpPrms* cellGrpPrmsCpu,
                                              cumacSchdSol* schdSolCpu,
                                              AssocFn&& assocFn,
                                              Type0RejectDiagnostics* rejectDiag = nullptr)
{
    if (rejectDiag != nullptr) {
        rejectDiag->reset(nCell);
    }
    if (cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr || schdSolCpu->setSchdUePerCellTTI == nullptr ||
        schdSolCpu->allocSol == nullptr || nCell == 0U || nTotCell == 0U) {
        return;
    }
    if (rejectDiag != nullptr) {
        rejectDiag->nativeSlotCountPerCell.assign(slotLayout.slotCountPerCell.begin(), slotLayout.slotCountPerCell.end());
        rejectDiag->nativeCellSlotStart.assign(slotLayout.cellSlotStart.begin(), slotLayout.cellSlotStart.end());
        rejectDiag->nativeSlotToCell.assign(slotLayout.slotToCell.begin(), slotLayout.slotToCell.end());
        rejectDiag->nativeSlotValid.assign(slotLayout.slotValid.begin(), slotLayout.slotValid.end());
    }

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + nSchedUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (nTotCell * nPrg), static_cast<int16_t>(-1));

    std::vector<uint8_t> usedUe(nActiveUe, 0U);
    const auto bump = [](std::vector<int32_t>& counters, uint32_t cIdx) {
        if (cIdx < counters.size()) {
            counters[cIdx] += 1;
        }
    };
    const uint32_t ueLen = std::min<uint32_t>(nSchedUe, static_cast<uint32_t>(ueAction.size()));
    for (uint32_t slot = 0; slot < ueLen; ++slot) {
        if (!slotLayout.validSlot(slot)) {
            continue;
        }
        const int32_t ue = ueAction[slot];
        if (ue < 0 || static_cast<uint32_t>(ue) >= nActiveUe) {
            continue;
        }
        if (static_cast<uint32_t>(ue) >= ueMask.size() || ueMask[static_cast<uint32_t>(ue)] == 0U) {
            continue;
        }
        const uint32_t cIdx = slotLayout.slotToCell[slot];
        if (!assocFn(cIdx, static_cast<uint32_t>(ue))) {
            continue;
        }
        if (usedUe[static_cast<uint32_t>(ue)] != 0U) {
            if (rejectDiag != nullptr) {
                bump(rejectDiag->rejectDuplicateUeCount, cIdx);
            }
            continue;
        }
        usedUe[static_cast<uint32_t>(ue)] = 1U;
        schdSolCpu->setSchdUePerCellTTI[slot] = static_cast<uint16_t>(ue);
    }

    const uint32_t actionAllocLen = nTotCell * nPrg;
    const uint32_t prgLen = std::min<uint32_t>(actionAllocLen, static_cast<uint32_t>(prgAction.size()));
    for (uint32_t idx = 0; idx < prgLen; ++idx) {
        const int16_t v = prgAction[idx];
        if (v < 0) {
            continue;
        }
        const uint32_t cIdx = idx % nCell;
        const uint32_t prgIdx = idx / nTotCell;
        if (cellGrpPrmsCpu->prgMsk != nullptr && cellGrpPrmsCpu->prgMsk[cIdx] != nullptr &&
            cellGrpPrmsCpu->prgMsk[cIdx][prgIdx] == 0) {
            if (rejectDiag != nullptr) {
                bump(rejectDiag->rejectPrgMaskCount, cIdx);
            }
            continue;
        }
        const uint32_t slot = static_cast<uint32_t>(v);
        if (slot >= nSchedUe) {
            if (rejectDiag != nullptr) {
                bump(rejectDiag->rejectOobSlotCount, cIdx);
            }
            continue;
        }
        if (!slotLayout.slotBelongsToCell(slot, cIdx)) {
            if (rejectDiag != nullptr) {
                bump(rejectDiag->rejectWrongCellSlotCount, cIdx);
            }
            continue;
        }
        if (schdSolCpu->setSchdUePerCellTTI[slot] == 0xFFFF) {
            if (rejectDiag != nullptr) {
                bump(rejectDiag->rejectEmptySlotCount, cIdx);
            }
            continue;
        }
        schdSolCpu->allocSol[idx] = v;
    }
}

} // namespace cumac
