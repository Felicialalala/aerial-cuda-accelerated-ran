/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GnnRlPolicyRuntime.h"

#include "cumac.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cumac {
namespace {

float clampMin(float v, float lo)
{
    return std::max(v, lo);
}

const char* actionModeToString(const GnnRlPolicyRuntime::ActionMode mode)
{
    return mode == GnnRlPolicyRuntime::ActionMode::PrgOnlyType0 ? "prg_only_type0" : "joint";
}

const char* decodeModeToString(const GnnRlPolicyRuntime::DecodeMode mode)
{
    return mode == GnnRlPolicyRuntime::DecodeMode::Argmax ? "argmax" : "sample";
}

std::string maxPrgShareToString(float value)
{
    if (value > 0.0f) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
    return "auto";
}

template <typename ValidFn, typename LogitFn>
uint32_t sampleMaskedLogits(uint32_t classCount,
                            ValidFn&& isValid,
                            LogitFn&& logitAt,
                            uint32_t fallbackClass,
                            std::mt19937_64& rng)
{
    float maxLogit = std::numeric_limits<float>::lowest();
    uint32_t bestClass = fallbackClass;
    bool anyValid = false;

    for (uint32_t cls = 0; cls < classCount; ++cls) {
        if (!isValid(cls)) {
            continue;
        }
        const float logit = logitAt(cls);
        if (!std::isfinite(logit)) {
            continue;
        }
        anyValid = true;
        if (logit > maxLogit) {
            maxLogit = logit;
            bestClass = cls;
        }
    }
    if (!anyValid) {
        return fallbackClass;
    }

    double totalWeight = 0.0;
    for (uint32_t cls = 0; cls < classCount; ++cls) {
        if (!isValid(cls)) {
            continue;
        }
        const float logit = logitAt(cls);
        if (!std::isfinite(logit)) {
            continue;
        }
        totalWeight += std::exp(static_cast<double>(logit - maxLogit));
    }
    if (!(totalWeight > 0.0)) {
        return bestClass;
    }

    std::uniform_real_distribution<double> dist(0.0, totalWeight);
    const double target = dist(rng);
    double cumulative = 0.0;
    for (uint32_t cls = 0; cls < classCount; ++cls) {
        if (!isValid(cls)) {
            continue;
        }
        const float logit = logitAt(cls);
        if (!std::isfinite(logit)) {
            continue;
        }
        cumulative += std::exp(static_cast<double>(logit - maxLogit));
        if (target <= cumulative) {
            return cls;
        }
    }
    return bestClass;
}

} // namespace

GnnRlPolicyRuntime::GnnRlPolicyRuntime(const Config& cfg) : m_cfg(cfg), m_rng(cfg.sampleSeed)
{
}

GnnRlPolicyRuntime::~GnnRlPolicyRuntime()
{
    releaseCudaBuffers();
}

void GnnRlPolicyRuntime::releaseCudaBuffers()
{
    if (m_obsCellDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_obsCellDev));
        m_obsCellDev = nullptr;
    }
    if (m_obsUeDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_obsUeDev));
        m_obsUeDev = nullptr;
    }
    if (m_obsPrgDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_obsPrgDev));
        m_obsPrgDev = nullptr;
    }
    if (m_obsEdgeIndexDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_obsEdgeIndexDev));
        m_obsEdgeIndexDev = nullptr;
    }
    if (m_obsEdgeAttrDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_obsEdgeAttrDev));
        m_obsEdgeAttrDev = nullptr;
    }
    if (m_actionMaskUeDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_actionMaskUeDev));
        m_actionMaskUeDev = nullptr;
    }
    if (m_actionMaskCellUeDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_actionMaskCellUeDev));
        m_actionMaskCellUeDev = nullptr;
    }
    if (m_ueLogitsDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_ueLogitsDev));
        m_ueLogitsDev = nullptr;
    }
    if (m_prgLogitsDev != nullptr) {
        CUDA_CHECK_ERR(cudaFree(m_prgLogitsDev));
        m_prgLogitsDev = nullptr;
    }
}

bool GnnRlPolicyRuntime::initialize(const cumacCellGrpPrms* cellGrpPrmsCpu)
{
    if (cellGrpPrmsCpu == nullptr) {
        return false;
    }

    if (m_cfg.modelPath.empty()) {
        std::cerr << "[GNNRL_MODEL] empty model path\n";
        return false;
    }
    std::ifstream f(m_cfg.modelPath);
    if (!f.good()) {
        std::cerr << "[GNNRL_MODEL] model file not found: " << m_cfg.modelPath << "\n";
        return false;
    }

    if (cellGrpPrmsCpu->allocType != 0) {
        std::cerr << "[GNNRL_MODEL] supports allocType=0 only, got allocType=" << static_cast<int>(cellGrpPrmsCpu->allocType) << "\n";
        return false;
    }

    m_nCell = cellGrpPrmsCpu->nCell;
    m_nActiveUe = cellGrpPrmsCpu->nActiveUe;
    m_nSchedUe = cellGrpPrmsCpu->nUe;
    m_nPrbGrp = cellGrpPrmsCpu->nPrbGrp;
    m_totNumCell = cellGrpPrmsCpu->totNumCell;
    m_numUeSchdPerCellTTI = cellGrpPrmsCpu->numUeSchdPerCellTTI;
    m_nEdges = (m_nCell > 1U) ? (m_nCell * (m_nCell - 1U)) : 0U;

    if (m_nCell == 0U || m_nActiveUe == 0U || m_nSchedUe == 0U || m_nPrbGrp == 0U || m_totNumCell == 0U) {
        return false;
    }

    if (m_numUeSchdPerCellTTI == 0U && m_nCell > 0U) {
        m_numUeSchdPerCellTTI = m_nSchedUe / m_nCell;
    }
    if (m_numUeSchdPerCellTTI == 0U) {
        return false;
    }

    std::cerr << "[GNNRL_MODEL] init: modelPath=" << m_cfg.modelPath
              << " nCell=" << m_nCell
              << " nActiveUe=" << m_nActiveUe
              << " nSchedUe=" << m_nSchedUe
              << " nPrbGrp=" << m_nPrbGrp
              << " numUeSchdPerCellTTI=" << m_numUeSchdPerCellTTI
              << " actionMode=" << actionModeToString(m_cfg.actionMode)
              << " decodeMode=" << decodeModeToString(m_cfg.decodeMode)
              << " sampleSeed=" << m_cfg.sampleSeed
              << " noUeBias=" << m_cfg.noUeBias
              << " minSchedRatio=" << m_cfg.minSchedRatio
              << " noPrgBias=" << m_cfg.noPrgBias
              << " minPrgRatio=" << m_cfg.minPrgRatio
              << " maxPrgSharePerUe=" << maxPrgShareToString(m_cfg.maxPrgSharePerUe)
              << "\n";
    if (m_cfg.decodeMode == DecodeMode::Sample) {
        std::cerr << "[GNNRL_MODEL] sample decode follows online PPO action sampling semantics; "
                     "minSchedRatio/minPrgRatio/maxPrgSharePerUe apply in argmax mode only\n";
    }

    m_edgeIndexHost.clear();
    m_edgeIndexHost.reserve(static_cast<size_t>(m_nEdges) * 2U);
    for (uint32_t src = 0; src < m_nCell; ++src) {
        for (uint32_t dst = 0; dst < m_nCell; ++dst) {
            if (src == dst) {
                continue;
            }
            m_edgeIndexHost.push_back(static_cast<int64_t>(src));
            m_edgeIndexHost.push_back(static_cast<int64_t>(dst));
        }
    }

    m_obsCellHost.assign(static_cast<size_t>(m_nCell) * kCellFeatDim, 0.0f);
    m_obsUeHost.assign(static_cast<size_t>(m_nActiveUe) * kUeFeatDim, 0.0f);
    m_obsPrgHost.assign(static_cast<size_t>(m_nCell) * static_cast<size_t>(m_nPrbGrp) * kPrgFeatDim, 0.0f);
    m_obsEdgeAttrHost.assign(static_cast<size_t>(m_nEdges) * kEdgeFeatDim, 0.0f);
    m_actionMaskUeHost.assign(m_nActiveUe, 0.0f);
    m_actionMaskCellUeHost.assign(static_cast<size_t>(m_nCell) * m_nActiveUe, 0.0f);
    m_ueMaskHost.assign(m_nActiveUe, 0U);
    m_prgMaskHost.assign(static_cast<size_t>(m_nCell) * m_nPrbGrp, 1U);

    const size_t ueLogitsSize = static_cast<size_t>(m_nSchedUe) * (m_nActiveUe + 1U);
    const size_t prgLogitsSize = static_cast<size_t>(m_nCell) * m_nPrbGrp * (m_nSchedUe + 1U);
    m_ueLogitsHost.assign(ueLogitsSize, 0.0f);
    m_prgLogitsHost.assign(prgLogitsSize, 0.0f);

    releaseCudaBuffers();
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_obsCellDev), sizeof(float) * m_obsCellHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_obsUeDev), sizeof(float) * m_obsUeHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_obsPrgDev), sizeof(float) * m_obsPrgHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_obsEdgeIndexDev), sizeof(int64_t) * m_edgeIndexHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_obsEdgeAttrDev), sizeof(float) * m_obsEdgeAttrHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_actionMaskUeDev), sizeof(float) * m_actionMaskUeHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_actionMaskCellUeDev), sizeof(float) * m_actionMaskCellUeHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_ueLogitsDev), sizeof(float) * m_ueLogitsHost.size()));
    CUDA_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&m_prgLogitsDev), sizeof(float) * m_prgLogitsHost.size()));

    try {
        const std::vector<cumac_ml::trtTensorPrms_t> inputPrms = {
            {"obs_cell_features", {1, static_cast<int>(m_nCell), static_cast<int>(kCellFeatDim)}},
            {"obs_ue_features", {1, static_cast<int>(m_nActiveUe), static_cast<int>(kUeFeatDim)}},
            {"obs_prg_features", {1, static_cast<int>(m_nCell), static_cast<int>(m_nPrbGrp), static_cast<int>(kPrgFeatDim)}},
            {"obs_edge_index", {1, static_cast<int>(m_nEdges), 2}},
            {"obs_edge_attr", {1, static_cast<int>(m_nEdges), static_cast<int>(kEdgeFeatDim)}},
            {"action_mask_ue", {1, static_cast<int>(m_nActiveUe)}},
            {"action_mask_cell_ue", {1, static_cast<int>(m_nCell), static_cast<int>(m_nActiveUe)}},
        };
        const std::vector<cumac_ml::trtTensorPrms_t> outputPrms = {
            {"ue_logits", {1, static_cast<int>(m_nSchedUe), static_cast<int>(m_nActiveUe + 1U)}},
            {"prg_logits", {1, static_cast<int>(m_nCell), static_cast<int>(m_nPrbGrp), static_cast<int>(m_nSchedUe + 1U)}},
        };

        m_trtEngine = std::make_unique<cumac_ml::trtEngine>(m_cfg.modelPath.c_str(), true, 1U, inputPrms, outputPrms);
    } catch (const std::exception& e) {
        std::cerr << "[GNNRL_MODEL] failed to initialize TRT engine: " << e.what() << "\n";
        releaseCudaBuffers();
        m_trtEngine.reset();
        return false;
    }

    m_initialized = (m_trtEngine != nullptr);
    return m_initialized;
}

void GnnRlPolicyRuntime::setObservationExtras(const cumac::online::ObservationExtras& extras)
{
    m_obsExtras = extras;
}

bool GnnRlPolicyRuntime::assocToCell(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cIdx, uint32_t ueIdx) const
{
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        return cellGrpPrmsCpu->cellAssocActUe[cIdx * m_nActiveUe + ueIdx] != 0;
    }
    if (m_nCell > 0U && (m_nActiveUe % m_nCell) == 0U) {
        const uint32_t uePerCell = m_nActiveUe / m_nCell;
        return (ueIdx / uePerCell) == cIdx;
    }
    return false;
}

void GnnRlPolicyRuntime::buildObservation(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                          const cumacCellGrpPrms* cellGrpPrmsCpu)
{
    std::fill(m_obsCellHost.begin(), m_obsCellHost.end(), 0.0f);
    std::fill(m_obsUeHost.begin(), m_obsUeHost.end(), 0.0f);
    std::fill(m_obsPrgHost.begin(), m_obsPrgHost.end(), 0.0f);
    std::fill(m_obsEdgeAttrHost.begin(), m_obsEdgeAttrHost.end(), 0.0f);

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
                wbSinrLin += clampMin(cellGrpPrmsCpu->wbSinr[ueIdx * nUeAnt + ant], 1.0e-9f);
            }
            wbSinrLin /= static_cast<float>(nUeAnt);
        }

        float staleSlots = 0.0f;
        if (cellId >= 0 && cellGrpUeStatusCpu->lastSchdSlotActUe != nullptr && cellGrpPrmsCpu->currSlotIdxPerCell != nullptr) {
            const uint32_t currSlot = cellGrpPrmsCpu->currSlotIdxPerCell[static_cast<uint32_t>(cellId)];
            const uint32_t lastSlot = cellGrpUeStatusCpu->lastSchdSlotActUe[ueIdx];
            if (lastSlot == 0xFFFFFFFFU) {
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
        m_obsUeHost[base + 0] = bufferBytes;
        m_obsUeHost[base + 1] = avgRateBps / 1.0e6f;
        m_obsUeHost[base + 2] = wbSinrLin;
        m_obsUeHost[base + 3] = cqi;
        m_obsUeHost[base + 4] = ri;
        m_obsUeHost[base + 5] = tbErrAct;
        m_obsUeHost[base + 6] = newData;
        m_obsUeHost[base + 7] = staleSlots;
        if (m_obsExtras.hasUeExtra(m_nActiveUe)) {
            const size_t extraBase =
                static_cast<size_t>(ueIdx) * cumac::online::ObservationFeatureLayout::kUeExtraFeatDim;
            m_obsUeHost[base + 8] = m_obsExtras.ueExtraFeatures[extraBase + 0U];
            m_obsUeHost[base + 9] = m_obsExtras.ueExtraFeatures[extraBase + 1U];
            m_obsUeHost[base + 10] = m_obsExtras.ueExtraFeatures[extraBase + 2U];
            m_obsUeHost[base + 11] = m_obsExtras.ueExtraFeatures[extraBase + 3U];
        }

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
        m_obsCellHost[base + 0] = cellLoadBytes[cIdx];
        m_obsCellHost[base + 1] = static_cast<float>(cellUeCount[cIdx]);
        m_obsCellHost[base + 2] = meanWbSinrLin;
        m_obsCellHost[base + 3] = cellAvgRateSum[cIdx] / ueCount;
        m_obsCellHost[base + 4] = tbErrRate;
    }

    const float normLoad = std::max(totalLoad, 1.0f);
    size_t edgePos = 0;
    for (uint32_t src = 0; src < m_nCell; ++src) {
        for (uint32_t dst = 0; dst < m_nCell; ++dst) {
            if (src == dst) {
                continue;
            }
            m_obsEdgeAttrHost[edgePos * kEdgeFeatDim + 0] = cellLoadBytes[src] / normLoad;
            m_obsEdgeAttrHost[edgePos * kEdgeFeatDim + 1] = (cellLoadBytes[src] - cellLoadBytes[dst]) / normLoad;
            edgePos += 1U;
        }
    }

    if (m_obsExtras.hasPrgFeatures(m_nCell, m_nPrbGrp)) {
        m_obsPrgHost = m_obsExtras.prgFeatures;
    }
}

void GnnRlPolicyRuntime::buildActionMask(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                         const cumacCellGrpPrms* cellGrpPrmsCpu)
{
    std::fill(m_ueMaskHost.begin(), m_ueMaskHost.end(), 0U);
    std::fill(m_actionMaskUeHost.begin(), m_actionMaskUeHost.end(), 0.0f);
    std::fill(m_prgMaskHost.begin(), m_prgMaskHost.end(), 1U);
    std::fill(m_actionMaskCellUeHost.begin(), m_actionMaskCellUeHost.end(), 0.0f);

    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
        bool hasAssoc = false;
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            if (assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                hasAssoc = true;
                m_actionMaskCellUeHost[cIdx * m_nActiveUe + ueIdx] = 1.0f;
            }
        }
        const bool hasBuffer = (cellGrpUeStatusCpu->bufferSize != nullptr)
                                   ? (cellGrpUeStatusCpu->bufferSize[ueIdx] > 0U)
                                   : true;
        m_ueMaskHost[ueIdx] = (hasAssoc && hasBuffer) ? 1U : 0U;
        m_actionMaskUeHost[ueIdx] = (m_ueMaskHost[ueIdx] != 0U) ? 1.0f : 0.0f;
    }

    if (cellGrpPrmsCpu->prgMsk != nullptr) {
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            if (cellGrpPrmsCpu->prgMsk[cIdx] == nullptr) {
                continue;
            }
            for (uint32_t prgIdx = 0; prgIdx < m_nPrbGrp; ++prgIdx) {
                m_prgMaskHost[cIdx * m_nPrbGrp + prgIdx] = (cellGrpPrmsCpu->prgMsk[cIdx][prgIdx] != 0) ? 1U : 0U;
            }
        }
    }
}

bool GnnRlPolicyRuntime::buildAndRunModel(cudaStream_t stream)
{
    if (m_trtEngine == nullptr) {
        return false;
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_obsCellDev, m_obsCellHost.data(), sizeof(float) * m_obsCellHost.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_obsUeDev, m_obsUeHost.data(), sizeof(float) * m_obsUeHost.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_obsPrgDev, m_obsPrgHost.data(), sizeof(float) * m_obsPrgHost.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_obsEdgeIndexDev, m_edgeIndexHost.data(), sizeof(int64_t) * m_edgeIndexHost.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_obsEdgeAttrDev, m_obsEdgeAttrHost.data(), sizeof(float) * m_obsEdgeAttrHost.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_actionMaskUeDev,
        m_actionMaskUeHost.data(),
        sizeof(float) * m_actionMaskUeHost.size(),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_actionMaskCellUeDev,
        m_actionMaskCellUeHost.data(),
        sizeof(float) * m_actionMaskCellUeHost.size(),
        cudaMemcpyHostToDevice,
        stream));

    const std::vector<void*> inputs = {
        reinterpret_cast<void*>(m_obsCellDev),
        reinterpret_cast<void*>(m_obsUeDev),
        reinterpret_cast<void*>(m_obsPrgDev),
        reinterpret_cast<void*>(m_obsEdgeIndexDev),
        reinterpret_cast<void*>(m_obsEdgeAttrDev),
        reinterpret_cast<void*>(m_actionMaskUeDev),
        reinterpret_cast<void*>(m_actionMaskCellUeDev),
    };
    const std::vector<void*> outputs = {
        reinterpret_cast<void*>(m_ueLogitsDev),
        reinterpret_cast<void*>(m_prgLogitsDev),
    };

    if (!m_trtEngine->setup(inputs, outputs, 1U)) {
        std::cerr << "[GNNRL_MODEL] trt setup failed\n";
        return false;
    }
    if (!m_trtEngine->run(stream)) {
        std::cerr << "[GNNRL_MODEL] trt run failed\n";
        return false;
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_ueLogitsHost.data(), m_ueLogitsDev, sizeof(float) * m_ueLogitsHost.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        m_prgLogitsHost.data(), m_prgLogitsDev, sizeof(float) * m_prgLogitsHost.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));

    return true;
}

Type0SlotLayout GnnRlPolicyRuntime::buildSlotLayout(const cumacCellGrpPrms* cellGrpPrmsCpu) const
{
    return cumac::buildType0SlotLayout(m_nCell, m_nActiveUe, m_nSchedUe, [&](uint32_t cIdx, uint32_t ueIdx) -> bool {
        return m_ueMaskHost[ueIdx] != 0U && assocToCell(cellGrpPrmsCpu, cIdx, ueIdx);
    });
}

void GnnRlPolicyRuntime::populateType0AllUeSelection(const cumacCellGrpPrms* cellGrpPrmsCpu,
                                                     const Type0SlotLayout& slotLayout,
                                                     cumacSchdSol* schdSolCpu) const
{
    if (cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr || schdSolCpu->setSchdUePerCellTTI == nullptr) {
        return;
    }

    for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
        uint32_t localSlot = 0U;
        for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
            if (m_ueMaskHost[ueIdx] == 0U || !assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                continue;
            }
            if (localSlot >= slotLayout.slotCountPerCell[cIdx]) {
                break;
            }
            const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localSlot;
            if (schedSlot >= m_nSchedUe) {
                break;
            }
            schdSolCpu->setSchdUePerCellTTI[schedSlot] = static_cast<uint16_t>(ueIdx);
            localSlot += 1U;
        }
    }
}

void GnnRlPolicyRuntime::applySampledType0Action(const std::vector<int32_t>& ueAction,
                                                 const std::vector<int16_t>& prgAction,
                                                 const Type0SlotLayout& slotLayout,
                                                 const cumacCellGrpPrms* cellGrpPrmsCpu,
                                                 cumacSchdSol* schdSolCpu) const
{
    if (cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr || schdSolCpu->setSchdUePerCellTTI == nullptr ||
        schdSolCpu->allocSol == nullptr) {
        return;
    }

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + m_nSchedUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (m_totNumCell * m_nPrbGrp), static_cast<int16_t>(-1));

    std::vector<uint8_t> usedUe(m_nActiveUe, 0U);
    const uint32_t ueLen = std::min<uint32_t>(m_nSchedUe, static_cast<uint32_t>(ueAction.size()));
    for (uint32_t slot = 0; slot < ueLen; ++slot) {
        if (!slotLayout.validSlot(slot)) {
            continue;
        }
        const int32_t ue = ueAction[slot];
        if (ue < 0 || static_cast<uint32_t>(ue) >= m_nActiveUe) {
            continue;
        }
        if (m_ueMaskHost[static_cast<uint32_t>(ue)] == 0U) {
            continue;
        }
        const uint32_t cIdx = slotLayout.slotToCell[slot];
        if (!assocToCell(cellGrpPrmsCpu, cIdx, static_cast<uint32_t>(ue))) {
            continue;
        }
        if (usedUe[static_cast<uint32_t>(ue)] != 0U) {
            continue;
        }
        usedUe[static_cast<uint32_t>(ue)] = 1U;
        schdSolCpu->setSchdUePerCellTTI[slot] = static_cast<uint16_t>(ue);
    }

    const uint32_t actionAllocLen = m_totNumCell * m_nPrbGrp;
    const uint32_t prgLen = std::min<uint32_t>(actionAllocLen, static_cast<uint32_t>(prgAction.size()));
    for (uint32_t idx = 0; idx < prgLen; ++idx) {
        const int16_t v = prgAction[idx];
        if (v < 0) {
            continue;
        }
        const uint32_t cIdx = idx % m_nCell;
        const uint32_t prgIdx = idx / m_totNumCell;
        if (cellGrpPrmsCpu->prgMsk != nullptr && cellGrpPrmsCpu->prgMsk[cIdx] != nullptr) {
            if (cellGrpPrmsCpu->prgMsk[cIdx][prgIdx] == 0) {
                continue;
            }
        }
        const uint32_t slot = static_cast<uint32_t>(v);
        if (slot >= m_nSchedUe) {
            continue;
        }
        if (!slotLayout.slotBelongsToCell(slot, cIdx)) {
            continue;
        }
        if (schdSolCpu->setSchdUePerCellTTI[slot] == 0xFFFF) {
            continue;
        }
        schdSolCpu->allocSol[idx] = v;
    }
}

bool GnnRlPolicyRuntime::decodeType0Sampled(const cumacCellGrpPrms* cellGrpPrmsCpu,
                                            cumacSchdSol* schdSolCpu)
{
    if (cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr) {
        return false;
    }

    const Type0SlotLayout slotLayout = buildSlotLayout(cellGrpPrmsCpu);
    std::vector<int32_t> ueAction(m_nSchedUe, -1);
    std::vector<int16_t> prgAction(static_cast<size_t>(m_totNumCell) * m_nPrbGrp, static_cast<int16_t>(-1));

    if (m_cfg.actionMode == ActionMode::PrgOnlyType0) {
        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            uint32_t localSlot = 0U;
            for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                if (m_ueMaskHost[ueIdx] == 0U || !assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                    continue;
                }
                if (localSlot >= slotLayout.slotCountPerCell[cIdx]) {
                    break;
                }
                const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localSlot;
                if (schedSlot >= m_nSchedUe) {
                    break;
                }
                ueAction[schedSlot] = static_cast<int32_t>(ueIdx);
                localSlot += 1U;
            }
        }
    } else {
        const uint32_t ueNoClass = m_nActiveUe;
        const uint32_t ueClassCount = m_nActiveUe + 1U;
        for (uint32_t schedSlot = 0; schedSlot < m_nSchedUe; ++schedSlot) {
            if (!slotLayout.validSlot(schedSlot)) {
                continue;
            }
            const uint32_t cIdx = slotLayout.slotToCell[schedSlot];
            const size_t base = static_cast<size_t>(schedSlot) * ueClassCount;
            const uint32_t sampledClass = sampleMaskedLogits(
                ueClassCount,
                [&](uint32_t cls) -> bool {
                    if (cls == ueNoClass) {
                        return true;
                    }
                    return cls < m_nActiveUe && m_ueMaskHost[cls] != 0U && assocToCell(cellGrpPrmsCpu, cIdx, cls);
                },
                [&](uint32_t cls) -> float {
                    if (cls == ueNoClass) {
                        return m_ueLogitsHost[base + ueNoClass] - m_cfg.noUeBias;
                    }
                    return m_ueLogitsHost[base + cls];
                },
                ueNoClass,
                m_rng);
            if (sampledClass != ueNoClass) {
                ueAction[schedSlot] = static_cast<int32_t>(sampledClass);
            }
        }
    }

    std::vector<uint8_t> selectedSlotMask(m_nSchedUe, 0U);
    std::vector<uint8_t> usedUe(m_nActiveUe, 0U);
    for (uint32_t schedSlot = 0; schedSlot < m_nSchedUe; ++schedSlot) {
        if (!slotLayout.validSlot(schedSlot)) {
            continue;
        }
        const int32_t ue = ueAction[schedSlot];
        if (ue < 0 || static_cast<uint32_t>(ue) >= m_nActiveUe) {
            continue;
        }
        if (m_ueMaskHost[static_cast<uint32_t>(ue)] == 0U) {
            continue;
        }
        const uint32_t cIdx = slotLayout.slotToCell[schedSlot];
        if (!assocToCell(cellGrpPrmsCpu, cIdx, static_cast<uint32_t>(ue))) {
            continue;
        }
        if (usedUe[static_cast<uint32_t>(ue)] != 0U) {
            continue;
        }
        usedUe[static_cast<uint32_t>(ue)] = 1U;
        selectedSlotMask[schedSlot] = 1U;
    }

    const uint32_t prgNoClass = m_nSchedUe;
    const uint32_t prgClassCount = m_nSchedUe + 1U;

    for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
        for (uint32_t prgIdx = 0; prgIdx < m_nPrbGrp; ++prgIdx) {
            if (m_prgMaskHost[cIdx * m_nPrbGrp + prgIdx] == 0U) {
                continue;
            }
            const size_t base = (static_cast<size_t>(cIdx) * m_nPrbGrp + prgIdx) * prgClassCount;
            const uint32_t sampledClass = sampleMaskedLogits(
                prgClassCount,
                [&](uint32_t cls) -> bool {
                    if (cls == prgNoClass) {
                        return true;
                    }
                    if (cls >= m_nSchedUe) {
                        return false;
                    }
                    if (!slotLayout.slotBelongsToCell(cls, cIdx)) {
                        return false;
                    }
                    return selectedSlotMask[cls] != 0U;
                },
                [&](uint32_t cls) -> float {
                    if (cls == prgNoClass) {
                        return m_prgLogitsHost[base + prgNoClass] - m_cfg.noPrgBias;
                    }
                    return m_prgLogitsHost[base + cls];
                },
                prgNoClass,
                m_rng);
            if (sampledClass != prgNoClass) {
                prgAction[static_cast<size_t>(prgIdx) * m_totNumCell + cIdx] = static_cast<int16_t>(sampledClass);
            }
        }
    }

    applySampledType0Action(ueAction, prgAction, slotLayout, cellGrpPrmsCpu, schdSolCpu);
    return true;
}

bool GnnRlPolicyRuntime::decodeType0Argmax(const cumacCellGrpPrms* cellGrpPrmsCpu,
                                           cumacSchdSol* schdSolCpu) const
{
    if (schdSolCpu == nullptr || schdSolCpu->setSchdUePerCellTTI == nullptr || schdSolCpu->allocSol == nullptr) {
        return false;
    }

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + m_nSchedUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (m_totNumCell * m_nPrbGrp), static_cast<int16_t>(-1));
    const Type0SlotLayout slotLayout = buildSlotLayout(cellGrpPrmsCpu);

    if (m_cfg.actionMode == ActionMode::PrgOnlyType0) {
        populateType0AllUeSelection(cellGrpPrmsCpu, slotLayout, schdSolCpu);
    } else {
        const uint32_t ueNoClass = m_nActiveUe;
        const uint32_t ueClassCount = m_nActiveUe + 1U;

        for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
            const uint32_t numCellSched = slotLayout.slotCountPerCell[cIdx];
            std::vector<uint8_t> usedUe(m_nActiveUe, 0U);
            std::vector<uint32_t> unscheduledSlots;
            unscheduledSlots.reserve(numCellSched);
            uint32_t scheduledCnt = 0U;

            for (uint32_t localIdx = 0; localIdx < numCellSched; ++localIdx) {
                const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localIdx;
                if (localIdx >= numCellSched || schedSlot >= m_nSchedUe) {
                    continue;
                }

                const size_t base = static_cast<size_t>(schedSlot) * ueClassCount;
                uint32_t bestClass = ueNoClass;
                float bestScore = m_ueLogitsHost[base + ueNoClass] - m_cfg.noUeBias;

                for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                    if (m_ueMaskHost[ueIdx] == 0U || usedUe[ueIdx] != 0U || !assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                        continue;
                    }
                    const float s = m_ueLogitsHost[base + ueIdx];
                    if (s > bestScore) {
                        bestScore = s;
                        bestClass = ueIdx;
                    }
                }

                if (bestClass != ueNoClass) {
                    schdSolCpu->setSchdUePerCellTTI[schedSlot] = static_cast<uint16_t>(bestClass);
                    usedUe[bestClass] = 1U;
                    scheduledCnt += 1U;
                } else {
                    unscheduledSlots.push_back(schedSlot);
                }
            }

            // Guardrail against decode collapse: force-fill a minimum number of slots from legal UEs.
            const float ratio = std::max(0.0f, std::min(1.0f, m_cfg.minSchedRatio));
            const uint32_t minSched = static_cast<uint32_t>(std::floor(ratio * static_cast<float>(numCellSched) + 1.0e-6f));
            if (scheduledCnt < minSched && !unscheduledSlots.empty()) {
                for (uint32_t schedSlot : unscheduledSlots) {
                    if (scheduledCnt >= minSched) {
                        break;
                    }
                    const size_t base = static_cast<size_t>(schedSlot) * ueClassCount;
                    int32_t bestUe = -1;
                    float bestScore = std::numeric_limits<float>::lowest();
                    for (uint32_t ueIdx = 0; ueIdx < m_nActiveUe; ++ueIdx) {
                        if (m_ueMaskHost[ueIdx] == 0U || usedUe[ueIdx] != 0U || !assocToCell(cellGrpPrmsCpu, cIdx, ueIdx)) {
                            continue;
                        }
                        const float s = m_ueLogitsHost[base + ueIdx];
                        if (s > bestScore) {
                            bestScore = s;
                            bestUe = static_cast<int32_t>(ueIdx);
                        }
                    }
                    if (bestUe >= 0) {
                        schdSolCpu->setSchdUePerCellTTI[schedSlot] = static_cast<uint16_t>(bestUe);
                        usedUe[static_cast<uint32_t>(bestUe)] = 1U;
                        scheduledCnt += 1U;
                    }
                }
            }
        }
    }

    const uint32_t prgNoClass = m_nSchedUe;
    const uint32_t prgClassCount = m_nSchedUe + 1U;

    for (uint32_t cIdx = 0; cIdx < m_nCell; ++cIdx) {
        std::vector<uint32_t> noPrgList;
        noPrgList.reserve(m_nPrbGrp);
        uint32_t assignedPrg = 0U;
        uint32_t validPrg = 0U;
        const uint32_t numCellSched = slotLayout.slotCountPerCell[cIdx];
        std::vector<uint32_t> slotPrgCount(numCellSched, 0U);
        uint32_t activeSlotCount = 0U;

        for (uint32_t localIdx = 0; localIdx < numCellSched; ++localIdx) {
            const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localIdx;
            if (schedSlot < m_nSchedUe && schdSolCpu->setSchdUePerCellTTI[schedSlot] != 0xFFFF) {
                activeSlotCount += 1U;
            }
        }

        for (uint32_t prgIdx = 0; prgIdx < m_nPrbGrp; ++prgIdx) {
            if (m_prgMaskHost[cIdx * m_nPrbGrp + prgIdx] == 0U) {
                continue;
            }
            validPrg += 1U;
        }

        uint32_t perSlotPrgCap = validPrg;
        float maxPrgShare = m_cfg.maxPrgSharePerUe;
        if (maxPrgShare <= 0.0f) {
            if (m_cfg.actionMode == ActionMode::PrgOnlyType0 && activeSlotCount > 0U) {
                // Greedy argmax decode can collapse onto one slot even when the learned
                // stochastic policy was balanced. Keep at least a few live slots active
                // by default in prg_only_type0 unless the caller explicitly overrides it.
                maxPrgShare = std::min(1.0f, 2.0f / static_cast<float>(activeSlotCount));
            } else {
                maxPrgShare = 1.0f;
            }
        }
        maxPrgShare = std::max(0.0f, std::min(1.0f, maxPrgShare));
        if (validPrg > 0U && maxPrgShare > 0.0f && maxPrgShare < 1.0f) {
            perSlotPrgCap = std::max<uint32_t>(
                1U, static_cast<uint32_t>(std::floor(maxPrgShare * static_cast<float>(validPrg) + 1.0e-6f)));
        }

        for (uint32_t prgIdx = 0; prgIdx < m_nPrbGrp; ++prgIdx) {
            if (m_prgMaskHost[cIdx * m_nPrbGrp + prgIdx] == 0U) {
                continue;
            }
            const size_t base = (static_cast<size_t>(cIdx) * m_nPrbGrp + prgIdx) * prgClassCount;
            const float noClassScore = m_prgLogitsHost[base + prgNoClass] - m_cfg.noPrgBias;

            int32_t bestLocalAny = -1;
            float bestScoreAny = std::numeric_limits<float>::lowest();
            int32_t bestLocalCap = -1;
            float bestScoreCap = std::numeric_limits<float>::lowest();
            for (uint32_t localIdx = 0; localIdx < numCellSched; ++localIdx) {
                const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localIdx;
                if (schedSlot >= m_nSchedUe) {
                    continue;
                }
                if (schdSolCpu->setSchdUePerCellTTI[schedSlot] == 0xFFFF) {
                    continue;
                }

                const float s = m_prgLogitsHost[base + schedSlot];
                if (s > bestScoreAny) {
                    bestScoreAny = s;
                    bestLocalAny = static_cast<int32_t>(localIdx);
                }
                if (slotPrgCount[localIdx] < perSlotPrgCap && s > bestScoreCap) {
                    bestScoreCap = s;
                    bestLocalCap = static_cast<int32_t>(localIdx);
                }
            }

            int32_t chosenLocal = bestLocalCap;
            float chosenScore = bestScoreCap;
            if (chosenLocal < 0) {
                chosenLocal = bestLocalAny;
                chosenScore = bestScoreAny;
            }

            if (chosenLocal >= 0 && chosenScore > noClassScore) {
                const uint32_t bestClass = slotLayout.cellSlotStart[cIdx] + static_cast<uint32_t>(chosenLocal);
                schdSolCpu->allocSol[prgIdx * m_totNumCell + cIdx] = static_cast<int16_t>(bestClass);
                assignedPrg += 1U;
                slotPrgCount[static_cast<uint32_t>(chosenLocal)] += 1U;
            } else {
                noPrgList.push_back(prgIdx);
            }
        }

        // Guardrail against decode collapse on PRG head: force-fill a minimum share of valid PRGs.
        const float prgRatio = std::max(0.0f, std::min(1.0f, m_cfg.minPrgRatio));
        const uint32_t minPrgAssign = static_cast<uint32_t>(std::floor(prgRatio * static_cast<float>(validPrg) + 1.0e-6f));
        if (assignedPrg < minPrgAssign && !noPrgList.empty()) {
            for (uint32_t prgIdx : noPrgList) {
                if (assignedPrg >= minPrgAssign) {
                    break;
                }

                const size_t base = (static_cast<size_t>(cIdx) * m_nPrbGrp + prgIdx) * prgClassCount;
                int32_t bestSlot = -1;
                float bestScore = std::numeric_limits<float>::lowest();
                int32_t bestSlotAny = -1;
                float bestScoreAny = std::numeric_limits<float>::lowest();
                for (uint32_t localIdx = 0; localIdx < numCellSched; ++localIdx) {
                    const uint32_t schedSlot = slotLayout.cellSlotStart[cIdx] + localIdx;
                    if (schedSlot >= m_nSchedUe) {
                        continue;
                    }
                    if (schdSolCpu->setSchdUePerCellTTI[schedSlot] == 0xFFFF) {
                        continue;
                    }
                    const float s = m_prgLogitsHost[base + schedSlot];
                    if (s > bestScoreAny) {
                        bestScoreAny = s;
                        bestSlotAny = static_cast<int32_t>(schedSlot);
                    }
                    if (slotPrgCount[localIdx] < perSlotPrgCap && s > bestScore) {
                        bestScore = s;
                        bestSlot = static_cast<int32_t>(schedSlot);
                    }
                }
                if (bestSlot < 0) {
                    bestSlot = bestSlotAny;
                }
                if (bestSlot >= 0) {
                    schdSolCpu->allocSol[prgIdx * m_totNumCell + cIdx] = static_cast<int16_t>(bestSlot);
                    assignedPrg += 1U;
                    const uint32_t localIdx = static_cast<uint32_t>(bestSlot) - slotLayout.cellSlotStart[cIdx];
                    if (localIdx < slotPrgCount.size()) {
                        slotPrgCount[localIdx] += 1U;
                    }
                }
            }
        }
    }

    return true;
}

bool GnnRlPolicyRuntime::decodeType0(const cumacCellGrpPrms* cellGrpPrmsCpu,
                                     cumacSchdSol* schdSolCpu)
{
    if (m_cfg.decodeMode == DecodeMode::Argmax) {
        return decodeType0Argmax(cellGrpPrmsCpu, schdSolCpu);
    }
    return decodeType0Sampled(cellGrpPrmsCpu, schdSolCpu);
}

bool GnnRlPolicyRuntime::inferType0(const cumacCellGrpUeStatus* cellGrpUeStatusCpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu,
                                    cumacSchdSol* schdSolCpu,
                                    cumacSchdSol* schdSolGpu,
                                    cudaStream_t stream)
{
    if (!m_initialized || cellGrpUeStatusCpu == nullptr || cellGrpPrmsCpu == nullptr || schdSolCpu == nullptr || schdSolGpu == nullptr) {
        return false;
    }
    if (cellGrpPrmsCpu->allocType != 0) {
        return false;
    }

    buildObservation(cellGrpUeStatusCpu, cellGrpPrmsCpu);
    buildActionMask(cellGrpUeStatusCpu, cellGrpPrmsCpu);

    if (!buildAndRunModel(stream)) {
        return false;
    }

    if (!decodeType0(cellGrpPrmsCpu, schdSolCpu)) {
        return false;
    }

    const size_t ueSelBytes = static_cast<size_t>(m_nSchedUe) * sizeof(uint16_t);
    const size_t allocBytes = static_cast<size_t>(m_totNumCell) * m_nPrbGrp * sizeof(int16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->allocSol, schdSolCpu->allocSol, allocBytes, cudaMemcpyHostToDevice, stream));

    return true;
}

} // namespace cumac
