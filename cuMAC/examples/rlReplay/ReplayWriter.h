/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/api.h"

namespace cumac {

class ReplayWriter {
public:
    struct Config {
        bool enabled = false;
        std::string outDir;
        std::string scenario;
        std::string policy;
        int seed = 0;
    };

    explicit ReplayWriter(const Config& cfg);
    ~ReplayWriter();

    bool enabled() const { return m_enabled; }
    bool initialize(const cumacCellGrpPrms* cellGrpPrmsCpu);
    void capturePreActionObs(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                             const cumacCellGrpPrms* cellGrpPrmsCpu,
                             int tti);
    void appendTransition(int tti,
                          bool done,
                          const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu,
                          const cumacSchdSol* schdSolCpu);
    void finalize();

private:
    static constexpr uint32_t kVersion = 2;
    static constexpr uint32_t kCellFeatDim = 5;
    static constexpr uint32_t kUeFeatDim = 8;
    static constexpr uint32_t kEdgeFeatDim = 2;
    static constexpr uint32_t kRewardTerms = 4;

    Config m_cfg;
    bool m_enabled = false;
    bool m_initialized = false;
    bool m_hasPreObs = false;
    int m_preObsTti = -1;

    uint32_t m_recordCount = 0;
    uint32_t m_nCell = 0;
    uint32_t m_nActiveUe = 0;
    uint32_t m_nSchedUe = 0;
    uint32_t m_nPrbGrp = 0;
    uint32_t m_totNumCell = 0;
    uint32_t m_allocType = 0;
    uint32_t m_nEdges = 0;
    uint32_t m_actionAllocLen = 0;
    uint64_t m_recordBytes = 0;

    std::string m_recordsPath;
    std::string m_metaPath;
    std::string m_schemaPath;
    std::ofstream m_records;

    std::vector<int16_t> m_edgeIndex;
    std::vector<float> m_preCellFeatures;
    std::vector<float> m_preUeFeatures;
    std::vector<float> m_preEdgeAttr;
    std::vector<uint8_t> m_preUeMask;
    std::vector<uint8_t> m_preCellUeMask;
    std::vector<uint8_t> m_prePrgMask;

    struct RewardTerms {
        float scalar = 0.0f;
        float throughput_mbps = 0.0f;
        float total_buffer_mb = 0.0f;
        float tb_err_rate = 0.0f;
        float fairness_jain = 0.0f;
    };

    bool ensureDirRecursive(const std::string& path) const;
    std::string joinPath(const std::string& base, const std::string& child) const;

    void buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                          const cumacCellGrpPrms* cellGrpPrmsCpu,
                          std::vector<float>& cellFeatures,
                          std::vector<float>& ueFeatures,
                          std::vector<float>& edgeAttr) const;
    void buildEdgeIndex();
    void buildAction(const cumacSchdSol* schdSolCpu,
                     std::vector<int32_t>& ueSelect,
                     std::vector<int16_t>& prgAlloc) const;
    void buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                         const cumacCellGrpPrms* cellGrpPrmsCpu,
                         std::vector<uint8_t>& ueMask,
                         std::vector<uint8_t>& cellUeMask,
                         std::vector<uint8_t>& prgMask) const;
    RewardTerms buildReward(const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const;
    void writeMeta() const;
    void writeSchema() const;
};

} // namespace cumac
