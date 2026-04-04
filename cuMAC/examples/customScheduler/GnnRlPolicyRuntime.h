/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "api.h"
#include "Type0SlotLayout.h"
#include "../ml/trtEngine.h"
#include "../onlineTrainBridge/OnlineObservationTypes.h"

namespace cumac {

class GnnRlPolicyRuntime {
public:
    enum class ActionMode : uint8_t {
        Joint = 0,
        PrgOnlyType0 = 1,
    };

    enum class DecodeMode : uint8_t {
        Sample = 0,
        Argmax = 1,
    };

    struct Config {
        std::string modelPath;
        int timeoutMs = 0;
        ActionMode actionMode = ActionMode::Joint;
        // `sample` mirrors online PPO action sampling + post-processing,
        // while `argmax` preserves the older deterministic decode path.
        DecodeMode decodeMode = DecodeMode::Sample;
        uint64_t sampleSeed = 0U;
        // Subtracted from NO_UE class logit during decode (positive encourages scheduling).
        float noUeBias = 0.0f;
        // Minimum per-cell scheduled-slot ratio to enforce after argmax decode.
        float minSchedRatio = 1.0f;
        // Subtracted from NO_PRG class logit during decode (positive encourages PRG assignment).
        float noPrgBias = 0.0f;
        // Minimum per-cell PRG-assignment ratio to enforce after argmax decode.
        // Default 0 keeps NO_PRG reachable so the policy may intentionally
        // leave part of the PRG grid empty.
        float minPrgRatio = 0.0f;
        // Per-cell max PRG share allowed for a single UE slot during decode (0<share<=1).
        // Non-positive enables runtime auto-guardrails for prg_only_type0 decode.
        float maxPrgSharePerUe = -1.0f;
    };

    explicit GnnRlPolicyRuntime(const Config& cfg);
    ~GnnRlPolicyRuntime();

    bool initialize(const cumacCellGrpPrms* cellGrpPrmsCpu);
    bool available() const { return m_initialized; }

    // Returns true when model inference + decode succeeds.
    bool inferType0(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                    const cumacCellGrpPrms* cellGrpPrmsCpu,
                    cumacSchdSol* schdSolCpu,
                    cumacSchdSol* schdSolGpu,
                    cudaStream_t stream);
    void setObservationExtras(const cumac::online::ObservationExtras& extras);

private:
    static constexpr uint32_t kCellFeatDim = cumac::online::ObservationFeatureLayout::kCellFeatDim;
    static constexpr uint32_t kUeFeatDim = cumac::online::ObservationFeatureLayout::kUeFeatDim;
    static constexpr uint32_t kEdgeFeatDim = cumac::online::ObservationFeatureLayout::kEdgeFeatDim;
    static constexpr uint32_t kPrgFeatDim = cumac::online::ObservationFeatureLayout::kPrgFeatDim;

    Config m_cfg;
    bool m_initialized = false;

    uint32_t m_nCell = 0;
    uint32_t m_nActiveUe = 0;
    uint32_t m_nSchedUe = 0;
    uint32_t m_nPrbGrp = 0;
    uint32_t m_totNumCell = 0;
    uint32_t m_nEdges = 0;
    uint32_t m_numUeSchdPerCellTTI = 0;

    std::vector<int64_t> m_edgeIndexHost;

    std::vector<float> m_obsCellHost;
    std::vector<float> m_obsUeHost;
    std::vector<float> m_obsPrgHost;
    std::vector<float> m_obsEdgeAttrHost;
    std::vector<float> m_actionMaskUeHost;
    std::vector<float> m_actionMaskCellUeHost;
    std::vector<uint8_t> m_ueMaskHost;
    std::vector<uint8_t> m_prgMaskHost;
    cumac::online::ObservationExtras m_obsExtras;

    std::vector<float> m_ueLogitsHost;
    std::vector<float> m_prgLogitsHost;

    float* m_obsCellDev = nullptr;
    float* m_obsUeDev = nullptr;
    float* m_obsPrgDev = nullptr;
    int64_t* m_obsEdgeIndexDev = nullptr;
    float* m_obsEdgeAttrDev = nullptr;
    float* m_actionMaskUeDev = nullptr;
    float* m_actionMaskCellUeDev = nullptr;
    float* m_ueLogitsDev = nullptr;
    float* m_prgLogitsDev = nullptr;

    std::unique_ptr<cumac_ml::trtEngine> m_trtEngine;

    bool buildAndRunModel(cudaStream_t stream);
    void buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu);
    void buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                         const cumacCellGrpPrms* cellGrpPrmsCpu);
    Type0SlotLayout buildSlotLayout(const cumacCellGrpPrms* cellGrpPrmsCpu) const;
    void populateType0AllUeSelection(const cumacCellGrpPrms* cellGrpPrmsCpu,
                                     const Type0SlotLayout& slotLayout,
                                     cumacSchdSol* schdSolCpu) const;
    void applySampledType0Action(const std::vector<int32_t>& ueAction,
                                 const std::vector<int16_t>& prgAction,
                                 const Type0SlotLayout& slotLayout,
                                 const cumacCellGrpPrms* cellGrpPrmsCpu,
                                 cumacSchdSol* schdSolCpu) const;
    bool decodeType0Sampled(const cumacCellGrpPrms* cellGrpPrmsCpu,
                            cumacSchdSol* schdSolCpu);
    bool decodeType0Argmax(const cumacCellGrpPrms* cellGrpPrmsCpu,
                           cumacSchdSol* schdSolCpu) const;
    bool decodeType0(const cumacCellGrpPrms* cellGrpPrmsCpu,
                     cumacSchdSol* schdSolCpu);

    bool assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx) const;
    void releaseCudaBuffers();

    std::mt19937_64 m_rng;
};

} // namespace cumac
