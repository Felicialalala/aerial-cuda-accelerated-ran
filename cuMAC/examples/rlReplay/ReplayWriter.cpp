/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ReplayWriter.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace cumac {
namespace {

template <typename T>
void writeScalar(std::ofstream& ofs, const T& value)
{
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void writeVector(std::ofstream& ofs, const std::vector<T>& values)
{
    if (!values.empty()) {
        ofs.write(reinterpret_cast<const char*>(values.data()), sizeof(T) * values.size());
    }
}

float clampNonNeg(float value)
{
    return std::max(0.0f, value);
}

} // namespace

ReplayWriter::ReplayWriter(const Config& cfg) : m_cfg(cfg), m_enabled(cfg.enabled)
{
}

ReplayWriter::~ReplayWriter()
{
    finalize();
}

bool ReplayWriter::ensureDirRecursive(const std::string& path) const
{
    if (path.empty()) {
        return false;
    }

    std::string current;
    if (path[0] == '/') {
        current = "/";
    }

    std::stringstream ss(path);
    std::string token;
    while (std::getline(ss, token, '/')) {
        if (token.empty()) {
            continue;
        }
        if (!current.empty() && current.back() != '/') {
            current.push_back('/');
        }
        current += token;
        if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

std::string ReplayWriter::joinPath(const std::string& base, const std::string& child) const
{
    if (base.empty()) {
        return child;
    }
    if (base.back() == '/') {
        return base + child;
    }
    return base + "/" + child;
}

void ReplayWriter::buildEdgeIndex()
{
    m_edgeIndex.clear();
    m_edgeIndex.reserve(static_cast<size_t>(m_nEdges) * 2);
    for (uint32_t src = 0; src < m_nCell; ++src) {
        for (uint32_t dst = 0; dst < m_nCell; ++dst) {
            if (src == dst) {
                continue;
            }
            m_edgeIndex.push_back(static_cast<int16_t>(src));
            m_edgeIndex.push_back(static_cast<int16_t>(dst));
        }
    }
}

bool ReplayWriter::initialize(const cumacCellGrpPrms* cellGrpPrmsCpu)
{
    if (!m_enabled) {
        return false;
    }
    if (cellGrpPrmsCpu == nullptr) {
        m_enabled = false;
        return false;
    }

    m_nCell = cellGrpPrmsCpu->nCell;
    m_nActiveUe = cellGrpPrmsCpu->nActiveUe;
    m_nSchedUe = cellGrpPrmsCpu->nUe;
    m_nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    m_totNumCell = cellGrpPrmsCpu->totNumCell;
    m_allocType = cellGrpPrmsCpu->allocType;
    m_nEdges = m_nCell > 1 ? (m_nCell * (m_nCell - 1)) : 0;
    m_actionAllocLen = (m_allocType == 0) ? (m_totNumCell * m_nPrbGrp) : (2 * m_nSchedUe);

    if (m_nCell == 0 || m_nActiveUe == 0 || m_nSchedUe == 0 || m_nPrbGrp == 0 || m_totNumCell == 0) {
        m_enabled = false;
        return false;
    }

    if (m_cfg.outDir.empty()) {
        m_cfg.outDir = "./replay";
    }
    if (!ensureDirRecursive(m_cfg.outDir)) {
        m_enabled = false;
        return false;
    }

    m_recordsPath = joinPath(m_cfg.outDir, "rl_replay_records.bin");
    m_metaPath = joinPath(m_cfg.outDir, "rl_replay_meta.json");
    m_schemaPath = joinPath(m_cfg.outDir, "rl_replay_schema.json");

    m_records.open(m_recordsPath, std::ios::binary | std::ios::trunc);
    if (!m_records.is_open()) {
        m_enabled = false;
        return false;
    }

    buildEdgeIndex();

    m_preCellFeatures.assign(static_cast<size_t>(m_nCell) * kCellFeatDim, 0.0f);
    m_preUeFeatures.assign(static_cast<size_t>(m_nActiveUe) * kUeFeatDim, 0.0f);
    m_preEdgeAttr.assign(static_cast<size_t>(m_nEdges) * kEdgeFeatDim, 0.0f);
    m_preUeMask.assign(m_nActiveUe, 0U);
    m_prePrgMask.assign(static_cast<size_t>(m_nCell) * m_nPrbGrp, 1U);

    m_recordBytes = 0;
    m_recordBytes += sizeof(int32_t);                         // tti
    m_recordBytes += sizeof(uint8_t) * 4;                     // done + pad
    m_recordBytes += sizeof(float) * (1 + kRewardTerms);      // reward scalar + terms
    m_recordBytes += sizeof(float) * m_preCellFeatures.size();
    m_recordBytes += sizeof(float) * m_preUeFeatures.size();
    m_recordBytes += sizeof(int16_t) * m_edgeIndex.size();
    m_recordBytes += sizeof(float) * m_preEdgeAttr.size();
    m_recordBytes += sizeof(int32_t) * m_nSchedUe;
    m_recordBytes += sizeof(int16_t) * m_actionAllocLen;
    m_recordBytes += sizeof(uint8_t) * m_nActiveUe;
    m_recordBytes += sizeof(uint8_t) * (m_nCell * m_nPrbGrp);
    m_recordBytes += sizeof(float) * m_preCellFeatures.size();
    m_recordBytes += sizeof(float) * m_preUeFeatures.size();
    m_recordBytes += sizeof(float) * m_preEdgeAttr.size();

    writeSchema();
    m_initialized = true;
    return true;
}

void ReplayWriter::buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu,
                                    std::vector<float>& cellFeatures,
                                    std::vector<float>& ueFeatures,
                                    std::vector<float>& edgeAttr) const
{
    cellFeatures.assign(static_cast<size_t>(m_nCell) * kCellFeatDim, 0.0f);
    ueFeatures.assign(static_cast<size_t>(m_nActiveUe) * kUeFeatDim, 0.0f);
    edgeAttr.assign(static_cast<size_t>(m_nEdges) * kEdgeFeatDim, 0.0f);

    std::vector<int32_t> ueServingCell(m_nActiveUe, -1);
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                if (cellGrpPrmsCpu->cellAssocActUe[cIdx * m_nActiveUe + ueIdx] != 0 && ueServingCell[ueIdx] < 0) {
                    ueServingCell[ueIdx] = static_cast<int32_t>(cIdx);
                }
            }
        }
    } else if (m_nCell > 0 && (m_nActiveUe % m_nCell) == 0) {
        const uint32_t uePerCell = m_nActiveUe / m_nCell;
        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            ueServingCell[ueIdx] = static_cast<int32_t>(ueIdx / uePerCell);
        }
    }

    std::vector<float> cellLoadBytes(m_nCell, 0.0f);
    std::vector<float> cellAvgRateSum(m_nCell, 0.0f);
    std::vector<float> cellWbSinrLinSum(m_nCell, 0.0f);
    std::vector<uint32_t> cellUeCount(m_nCell, 0U);
    std::vector<uint32_t> cellWbSinrCount(m_nCell, 0U);
    std::vector<uint32_t> cellTbValid(m_nCell, 0U);
    std::vector<uint32_t> cellTbErr(m_nCell, 0U);

    const uint8_t nUeAnt = cellGrpPrmsCpu->nUeAnt > 0 ? cellGrpPrmsCpu->nUeAnt : 1;

    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        const int32_t cellId = ueServingCell[ueIdx];
        const float bufferBytes = (cellGrpUeStatusCpu->bufferSize != nullptr)
                                      ? static_cast<float>(cellGrpUeStatusCpu->bufferSize[ueIdx])
                                      : 0.0f;
        const float avgRateBps = (cellGrpUeStatusCpu->avgRatesActUe != nullptr)
                                     ? cellGrpUeStatusCpu->avgRatesActUe[ueIdx]
                                     : 0.0f;

        float wbSinrLin = 0.0f;
        if (cellGrpPrmsCpu->wbSinr != nullptr) {
            for (uint8_t ant = 0; ant < nUeAnt; ++ant) {
                wbSinrLin += std::max(cellGrpPrmsCpu->wbSinr[ueIdx * nUeAnt + ant], 1.0e-9f);
            }
            wbSinrLin /= static_cast<float>(nUeAnt);
        }

        float staleSlots = 0.0f;
        if (cellId >= 0 && cellGrpUeStatusCpu->lastSchdSlotActUe != nullptr && cellGrpPrmsCpu->currSlotIdxPerCell != nullptr) {
            const uint32_t currSlot = cellGrpPrmsCpu->currSlotIdxPerCell[static_cast<uint32_t>(cellId)];
            const uint32_t lastSlot = cellGrpUeStatusCpu->lastSchdSlotActUe[ueIdx];
            if (lastSlot == 0xFFFFFFFF) {
                staleSlots = 10000.0f;
            } else if (currSlot >= lastSlot) {
                staleSlots = static_cast<float>(currSlot - lastSlot);
            }
        }

        const float cqi = (cellGrpUeStatusCpu->cqiActUe != nullptr) ? static_cast<float>(cellGrpUeStatusCpu->cqiActUe[ueIdx]) : -1.0f;
        const float ri = (cellGrpUeStatusCpu->riActUe != nullptr) ? static_cast<float>(cellGrpUeStatusCpu->riActUe[ueIdx]) : -1.0f;
        const float tbErrAct = (cellGrpUeStatusCpu->tbErrLastActUe != nullptr)
                                   ? static_cast<float>(cellGrpUeStatusCpu->tbErrLastActUe[ueIdx])
                                   : -1.0f;
        const float newData = (cellGrpUeStatusCpu->newDataActUe != nullptr)
                                  ? static_cast<float>(cellGrpUeStatusCpu->newDataActUe[ueIdx])
                                  : -1.0f;

        const size_t base = static_cast<size_t>(ueIdx) * kUeFeatDim;
        ueFeatures[base + 0] = bufferBytes;
        ueFeatures[base + 1] = avgRateBps / 1.0e6f;
        ueFeatures[base + 2] = wbSinrLin;
        ueFeatures[base + 3] = cqi;
        ueFeatures[base + 4] = ri;
        ueFeatures[base + 5] = tbErrAct;
        ueFeatures[base + 6] = newData;
        ueFeatures[base + 7] = staleSlots;

        if (cellId >= 0 && static_cast<uint32_t>(cellId) < m_nCell) {
            const uint32_t cIdx = static_cast<uint32_t>(cellId);
            cellLoadBytes[cIdx] += bufferBytes;
            cellAvgRateSum[cIdx] += avgRateBps / 1.0e6f;
            cellUeCount[cIdx] += 1U;
            cellWbSinrLinSum[cIdx] += wbSinrLin;
            cellWbSinrCount[cIdx] += 1U;
            if (tbErrAct == 0.0f || tbErrAct == 1.0f) {
                cellTbValid[cIdx] += 1U;
                if (tbErrAct > 0.5f) {
                    cellTbErr[cIdx] += 1U;
                }
            }
        }
    }

    float totalLoad = 0.0f;
    for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
        totalLoad += cellLoadBytes[cIdx];
        const float ueCount = static_cast<float>(std::max<uint32_t>(1U, cellUeCount[cIdx]));
        const float meanWbSinrLin = cellWbSinrCount[cIdx] > 0
                                        ? cellWbSinrLinSum[cIdx] / static_cast<float>(cellWbSinrCount[cIdx])
                                        : 0.0f;
        const float tbErrRate = cellTbValid[cIdx] > 0
                                    ? static_cast<float>(cellTbErr[cIdx]) / static_cast<float>(cellTbValid[cIdx])
                                    : 0.0f;
        const size_t base = static_cast<size_t>(cIdx) * kCellFeatDim;
        cellFeatures[base + 0] = cellLoadBytes[cIdx];
        cellFeatures[base + 1] = static_cast<float>(cellUeCount[cIdx]);
        cellFeatures[base + 2] = meanWbSinrLin;
        cellFeatures[base + 3] = cellAvgRateSum[cIdx] / ueCount;
        cellFeatures[base + 4] = tbErrRate;
    }

    const float normLoad = std::max(totalLoad, 1.0f);
    size_t edgePos = 0;
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
}

void ReplayWriter::capturePreActionObs(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                       const cumacCellGrpPrms* cellGrpPrmsCpu,
                                       int tti)
{
    if (!m_enabled || !m_initialized) {
        return;
    }
    buildObservation(cellGrpUeStatusCpu, cellGrpPrmsCpu, m_preCellFeatures, m_preUeFeatures, m_preEdgeAttr);
    buildActionMask(cellGrpUeStatusCpu, cellGrpPrmsCpu, m_preUeMask, m_prePrgMask);
    m_hasPreObs = true;
    m_preObsTti = tti;
}

void ReplayWriter::buildAction(const cumacSchdSol* schdSolCpu,
                               std::vector<int32_t>& ueSelect,
                               std::vector<int16_t>& prgAlloc) const
{
    ueSelect.assign(m_nSchedUe, -1);
    prgAlloc.assign(m_actionAllocLen, static_cast<int16_t>(-1));
    if (schdSolCpu == nullptr) {
        return;
    }

    if (schdSolCpu->setSchdUePerCellTTI != nullptr) {
        for (uint32_t uIdx = 0; uIdx < m_nSchedUe; ++uIdx) {
            const uint16_t ueId = schdSolCpu->setSchdUePerCellTTI[uIdx];
            if (ueId != 0xFFFF) {
                ueSelect[uIdx] = static_cast<int32_t>(ueId);
            }
        }
    }

    if (schdSolCpu->allocSol != nullptr) {
        for (uint32_t idx = 0; idx < m_actionAllocLen; ++idx) {
            prgAlloc[idx] = schdSolCpu->allocSol[idx];
        }
    }
}

void ReplayWriter::buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                   const cumacCellGrpPrms* cellGrpPrmsCpu,
                                   std::vector<uint8_t>& ueMask,
                                   std::vector<uint8_t>& prgMask) const
{
    ueMask.assign(m_nActiveUe, 0U);
    prgMask.assign(static_cast<size_t>(m_nCell) * m_nPrbGrp, 1U);

    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        bool hasAssoc = true;
        if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
            hasAssoc = false;
            for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
                if (cellGrpPrmsCpu->cellAssocActUe[cIdx * m_nActiveUe + ueIdx] != 0) {
                    hasAssoc = true;
                    break;
                }
            }
        }
        const bool hasBuffer = (cellGrpUeStatusCpu->bufferSize != nullptr)
                                   ? (cellGrpUeStatusCpu->bufferSize[ueIdx] > 0U)
                                   : true;
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

ReplayWriter::RewardTerms ReplayWriter::buildReward(const cumacCellGrpUeStatus* cellGrpUeStatusCpu) const
{
    RewardTerms terms;
    if (cellGrpUeStatusCpu == nullptr) {
        return terms;
    }

    double sumRate = 0.0;
    double sumRateSq = 0.0;
    double sumBufferBytes = 0.0;
    uint32_t tbValid = 0U;
    uint32_t tbErr = 0U;
    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        const float rate = (cellGrpUeStatusCpu->avgRatesActUe != nullptr) ? clampNonNeg(cellGrpUeStatusCpu->avgRatesActUe[ueIdx]) : 0.0f;
        sumRate += static_cast<double>(rate);
        sumRateSq += static_cast<double>(rate) * static_cast<double>(rate);
        if (cellGrpUeStatusCpu->bufferSize != nullptr) {
            sumBufferBytes += static_cast<double>(cellGrpUeStatusCpu->bufferSize[ueIdx]);
        }
        if (cellGrpUeStatusCpu->tbErrLastActUe != nullptr) {
            const int8_t v = cellGrpUeStatusCpu->tbErrLastActUe[ueIdx];
            if (v == 0 || v == 1) {
                tbValid += 1U;
                if (v == 1) {
                    tbErr += 1U;
                }
            }
        }
    }

    terms.throughput_mbps = static_cast<float>(sumRate / 1.0e6);
    terms.total_buffer_mb = static_cast<float>(sumBufferBytes / 1.0e6);
    terms.tb_err_rate = (tbValid > 0U) ? static_cast<float>(tbErr) / static_cast<float>(tbValid) : 0.0f;
    terms.fairness_jain = (sumRateSq > 0.0)
                              ? static_cast<float>((sumRate * sumRate) / (static_cast<double>(m_nActiveUe) * sumRateSq))
                              : 0.0f;
    terms.scalar = terms.throughput_mbps - 0.05f * terms.total_buffer_mb - 2.0f * terms.tb_err_rate + 0.5f * terms.fairness_jain;
    return terms;
}

void ReplayWriter::appendTransition(int tti,
                                    bool done,
                                    const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu,
                                    const cumacSchdSol* schdSolCpu)
{
    if (!m_enabled || !m_initialized || !m_records.is_open()) {
        return;
    }

    if (!m_hasPreObs || m_preObsTti != tti) {
        capturePreActionObs(cellGrpUeStatusCpu, cellGrpPrmsCpu, tti);
    }

    std::vector<float> nextCellFeatures;
    std::vector<float> nextUeFeatures;
    std::vector<float> nextEdgeAttr;
    buildObservation(cellGrpUeStatusCpu, cellGrpPrmsCpu, nextCellFeatures, nextUeFeatures, nextEdgeAttr);

    std::vector<int32_t> ueSelect;
    std::vector<int16_t> prgAlloc;
    buildAction(schdSolCpu, ueSelect, prgAlloc);
    if (m_preUeMask.size() != m_nActiveUe || m_prePrgMask.size() != static_cast<size_t>(m_nCell) * m_nPrbGrp) {
        buildActionMask(cellGrpUeStatusCpu, cellGrpPrmsCpu, m_preUeMask, m_prePrgMask);
    }

    const RewardTerms reward = buildReward(cellGrpUeStatusCpu);
    const uint8_t doneFlag = done ? 1U : 0U;
    const uint8_t pad[3] = {0U, 0U, 0U};
    const float rewardTerms[kRewardTerms] = {
        reward.throughput_mbps,
        reward.total_buffer_mb,
        reward.tb_err_rate,
        reward.fairness_jain,
    };

    writeScalar(m_records, static_cast<int32_t>(tti));
    m_records.write(reinterpret_cast<const char*>(&doneFlag), sizeof(doneFlag));
    m_records.write(reinterpret_cast<const char*>(pad), sizeof(pad));
    writeScalar(m_records, reward.scalar);
    m_records.write(reinterpret_cast<const char*>(rewardTerms), sizeof(rewardTerms));
    writeVector(m_records, m_preCellFeatures);
    writeVector(m_records, m_preUeFeatures);
    writeVector(m_records, m_edgeIndex);
    writeVector(m_records, m_preEdgeAttr);
    writeVector(m_records, ueSelect);
    writeVector(m_records, prgAlloc);
    writeVector(m_records, m_preUeMask);
    writeVector(m_records, m_prePrgMask);
    writeVector(m_records, nextCellFeatures);
    writeVector(m_records, nextUeFeatures);
    writeVector(m_records, nextEdgeAttr);

    m_recordCount += 1U;
    m_hasPreObs = false;
}

void ReplayWriter::writeMeta() const
{
    std::ofstream meta(m_metaPath, std::ios::trunc);
    if (!meta.is_open()) {
        return;
    }

    meta << "{\n"
         << "  \"format\": \"cumac_rl_replay\",\n"
         << "  \"version\": " << kVersion << ",\n"
         << "  \"records_file\": \"rl_replay_records.bin\",\n"
         << "  \"schema_file\": \"rl_replay_schema.json\",\n"
         << "  \"record_count\": " << m_recordCount << ",\n"
         << "  \"record_bytes\": " << m_recordBytes << ",\n"
         << "  \"dims\": {\n"
         << "    \"n_cell\": " << m_nCell << ",\n"
         << "    \"n_active_ue\": " << m_nActiveUe << ",\n"
         << "    \"n_sched_ue\": " << m_nSchedUe << ",\n"
         << "    \"n_tot_cell\": " << m_totNumCell << ",\n"
         << "    \"n_prg\": " << m_nPrbGrp << ",\n"
         << "    \"alloc_type\": " << m_allocType << ",\n"
         << "    \"n_edges\": " << m_nEdges << ",\n"
         << "    \"action_alloc_len\": " << m_actionAllocLen << "\n"
         << "  },\n"
         << "  \"feature_dims\": {\n"
         << "    \"cell\": " << kCellFeatDim << ",\n"
         << "    \"ue\": " << kUeFeatDim << ",\n"
         << "    \"edge\": " << kEdgeFeatDim << ",\n"
         << "    \"reward_terms\": " << kRewardTerms << "\n"
         << "  },\n"
         << "  \"meta\": {\n"
         << "    \"seed\": " << m_cfg.seed << ",\n"
         << "    \"policy\": \"" << m_cfg.policy << "\",\n"
         << "    \"scenario\": \"" << m_cfg.scenario << "\"\n"
         << "  }\n"
         << "}\n";
}

void ReplayWriter::writeSchema() const
{
    std::ofstream schema(m_schemaPath, std::ios::trunc);
    if (!schema.is_open()) {
        return;
    }

    schema << "{\n"
           << "  \"record_layout\": [\n"
           << "    {\"name\":\"tti\",\"dtype\":\"int32\",\"shape\":[1]},\n"
           << "    {\"name\":\"done\",\"dtype\":\"uint8\",\"shape\":[1]},\n"
           << "    {\"name\":\"reward_scalar\",\"dtype\":\"float32\",\"shape\":[1]},\n"
           << "    {\"name\":\"reward_terms\",\"dtype\":\"float32\",\"shape\":[" << kRewardTerms << "]},\n"
           << "    {\"name\":\"obs_cell_features\",\"dtype\":\"float32\",\"shape\":[" << m_nCell << "," << kCellFeatDim << "]},\n"
           << "    {\"name\":\"obs_ue_features\",\"dtype\":\"float32\",\"shape\":[" << m_nActiveUe << "," << kUeFeatDim << "]},\n"
           << "    {\"name\":\"obs_edge_index\",\"dtype\":\"int16\",\"shape\":[" << m_nEdges << ",2]},\n"
           << "    {\"name\":\"obs_edge_attr\",\"dtype\":\"float32\",\"shape\":[" << m_nEdges << "," << kEdgeFeatDim << "]},\n"
           << "    {\"name\":\"action_ue_select\",\"dtype\":\"int32\",\"shape\":[" << m_nSchedUe << "]},\n"
           << "    {\"name\":\"action_prg_alloc\",\"dtype\":\"int16\",\"shape\":[" << m_actionAllocLen << "]},\n"
           << "    {\"name\":\"action_mask_ue\",\"dtype\":\"uint8\",\"shape\":[" << m_nActiveUe << "]},\n"
           << "    {\"name\":\"action_mask_prg_cell\",\"dtype\":\"uint8\",\"shape\":[" << m_nCell << "," << m_nPrbGrp << "]},\n"
           << "    {\"name\":\"next_cell_features\",\"dtype\":\"float32\",\"shape\":[" << m_nCell << "," << kCellFeatDim << "]},\n"
           << "    {\"name\":\"next_ue_features\",\"dtype\":\"float32\",\"shape\":[" << m_nActiveUe << "," << kUeFeatDim << "]},\n"
           << "    {\"name\":\"next_edge_attr\",\"dtype\":\"float32\",\"shape\":[" << m_nEdges << "," << kEdgeFeatDim << "]}\n"
           << "  ],\n"
           << "  \"notes\": {\n"
           << "    \"action_ue_select\": \"-1 indicates invalid slot\",\n"
           << "    \"action_prg_alloc\": \"type-0: totNumCell*nPrbGrp; type-1: 2*nUe\",\n"
           << "    \"reward_terms_order\": [\"throughput_mbps\",\"total_buffer_mb\",\"tb_err_rate\",\"fairness_jain\"]\n"
           << "  }\n"
           << "}\n";
}

void ReplayWriter::finalize()
{
    if (!m_enabled || !m_initialized) {
        return;
    }
    if (m_records.is_open()) {
        m_records.close();
    }
    writeMeta();
    m_initialized = false;
}

} // namespace cumac
