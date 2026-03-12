/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "api.h"
#include "../ml/trtEngine.h"

namespace cumac {

class GnnRlPolicyRuntime {
public:
    struct Config {
        std::string modelPath;
        int timeoutMs = 0;
        // Subtracted from NO_UE class logit during decode (positive encourages scheduling).
        float noUeBias = 0.0f;
        // Minimum per-cell scheduled-slot ratio to enforce after argmax decode.
        float minSchedRatio = 1.0f;
        // Subtracted from NO_PRG class logit during decode (positive encourages PRG assignment).
        float noPrgBias = 0.0f;
        // Minimum per-cell PRG-assignment ratio to enforce after argmax decode.
        float minPrgRatio = 1.0f;
        // Per-cell max PRG share allowed for a single UE slot during decode (0<share<=1).
        float maxPrgSharePerUe = 1.0f;
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

private:
    static constexpr uint32_t kCellFeatDim = 5;
    static constexpr uint32_t kUeFeatDim = 8;
    static constexpr uint32_t kEdgeFeatDim = 2;

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
    std::vector<float> m_obsEdgeAttrHost;
    std::vector<uint8_t> m_ueMaskHost;
    std::vector<uint8_t> m_prgMaskHost;

    std::vector<float> m_ueLogitsHost;
    std::vector<float> m_prgLogitsHost;

    float* m_obsCellDev = nullptr;
    float* m_obsUeDev = nullptr;
    int64_t* m_obsEdgeIndexDev = nullptr;
    float* m_obsEdgeAttrDev = nullptr;
    float* m_ueLogitsDev = nullptr;
    float* m_prgLogitsDev = nullptr;

    std::unique_ptr<cumac_ml::trtEngine> m_trtEngine;

    bool buildAndRunModel(cudaStream_t stream);
    void buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu);
    void buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                         const cumacCellGrpPrms* cellGrpPrmsCpu);
    bool decodeType0(const cumacCellGrpPrms* cellGrpPrmsCpu,
                     cumacSchdSol* schdSolCpu) const;

    bool assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx) const;
    void releaseCudaBuffers();
};

} // namespace cumac
