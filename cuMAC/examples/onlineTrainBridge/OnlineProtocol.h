/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cumac::online {

constexpr uint32_t kMagic = 0x524c4e4fU; // "ONLR"
constexpr uint16_t kVersion = 10U;
constexpr uint32_t kRewardTerms = 17U;
constexpr uint32_t kStateFlagHasPostEqSinr = 1U << 0;
constexpr uint32_t kStateFlagHasTeacherAction = 1U << 1;
constexpr uint32_t kStateFlagHasActualUeDiagnostics = 1U << 2;
constexpr uint32_t kStateFlagHasAcceptedPrePdschUeDiagnostics = 1U << 3;
constexpr uint32_t kStateFlagHasNativeRejectDiagnostics = 1U << 4;
constexpr uint32_t kStateFlagHasNativeSlotLayoutDiagnostics = 1U << 5;

enum class MsgType : uint16_t {
    ResetReq = 1,
    ResetRsp = 2,
    StepReq = 3,
    StepRsp = 4,
    CloseReq = 5,
    CloseRsp = 6,
    ErrorRsp = 7,
};

#pragma pack(push, 1)

struct MsgHeader {
    uint32_t magic = kMagic;
    uint16_t version = kVersion;
    uint16_t type = 0;
    uint32_t payloadBytes = 0;
};

struct ResetReqPayload {
    int32_t seed = 0;
    int32_t episodeHorizon = 0;
    uint32_t flags = 0;
};

struct StepReqHeader {
    int32_t stepIdx = 0;
    uint32_t ueActionLen = 0;
    uint32_t prgActionLen = 0;
};

struct EnvDimsPayload {
    uint32_t nCell = 0;
    uint32_t nActiveUe = 0;
    uint32_t nSchedUe = 0;
    uint32_t nTotCell = 0;
    uint32_t nPrg = 0;
    uint32_t nEdges = 0;
    uint32_t allocType = 0;
    uint32_t actionAllocLen = 0;
    uint32_t cellFeatDim = 0;
    uint32_t ueFeatDim = 0;
    uint32_t edgeFeatDim = 0;
    uint32_t prgFeatDim = 0;
    uint32_t postEqLayerDim = 0;
};

struct StepStateHeader {
    int32_t tti = 0;
    uint8_t done = 0;
    uint8_t reserved[3] = {0, 0, 0};
    float rewardScalar = 0.0F;
    float rewardTerms[kRewardTerms] = {};
    EnvDimsPayload dims;
    uint32_t stateFlags = 0;
};

struct ErrorPayload {
    int32_t code = -1;
};

#pragma pack(pop)

static_assert(sizeof(MsgHeader) == 12U, "unexpected MsgHeader size");

} // namespace cumac::online
