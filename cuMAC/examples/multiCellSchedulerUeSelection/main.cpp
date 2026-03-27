/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #define periodicLightWt_
 #define PDSCH_
// #define exitCheckFail_ //exit when solution check fails and create TV for debugging 

#include "network.h"
#include "api.h"
#include "cumac.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h"
#include "../customScheduler/CustomUePrgScheduler.h"
#include "../onlineTrainBridge/OnlineBridgeServer.h"
#include "../onlineTrainBridge/OnlineFeatureCodec.h"
#include "../rlReplay/ReplayWriter.h"
#include "trafficModel/trafficService.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <vector>

using namespace cumac;

/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuMAC DL/UL scheduler pipeline test with [Arguments]\n");
    printf("Arguments:\n");
    printf("  -d  [Indication for DL/UL: 0 - UL, 1 - DL (default 1)]\n");
    printf("  -b  [Indication for RR baseline/reference-check mode: 0 - PF reference check, 1 - RR baseline (default 0)]\n");
    printf("  -p  [Indication for using FP16 PRG allocation kernel: 0 - FP32, 1 - FP16 (default 0)]\n");
    printf("  -t  [Indication for saving TV before return: 0 - not saving TV, 1 - save TV for GPU scheduler, 2 - save TV for CPU scheduler, 3 - save per-cell TVs for testMAC/cuMAC-CP (default 0)]\n");
    printf("  -f  [Indication for choosing fast fading: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)]\n"); // currently only CFR on Prg is used in network class, so 2 / 4 is not recommended
    printf("  -x  [Use custom UE selection + PRG allocation: 0 - disable, 1 - enable (default 0)]\n");
    printf("  -g <percent> Enable traffic generation and specify percent of configured UEs to generate traffic on");
    printf("  -r <packet_size_bytes> Specify average packet size in bytes for traffic generation");
    printf("Example 1 (call cuMAC DL scheduler pipeline with CPU reference check): './multiCellSchedulerUeSelection'\n");
    printf("Example 2 (call cuMAC UL scheduler pipeline with CPU reference check): './multiCellSchedulerUeSelection -d 0'\n");
    printf("Example 3 (call cuMAC DL scheduler pipeline with RR baseline): './multiCellSchedulerUeSelection -b 1'\n");
    printf("Example 4 (call cuMAC DL scheduler pipeline using GPU TDL channel): './multiCellSchedulerUeSelection -f <1 or 2>'\n");
    printf("Example 4 (call cuMAC DL scheduler pipeline using GPU CDL channel): './multiCellSchedulerUeSelection -f <3 or 4>'\n");
    printf("Example 5 (create cuMAC test vector for DL: './multiCellSchedulerUeSelection -t 1'\n");
    // <channel_file> = ~/mnt/cuMAC/100randTTI570Ues2ta2raUMa_xpol_2.5GHz.mat
}

namespace {

enum class ExecMode {
    Both,
    Gpu,
};

ExecMode parseExecMode(const std::string& value)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lower.empty() || lower == "both") {
        return ExecMode::Both;
    }
    if (lower == "gpu") {
        return ExecMode::Gpu;
    }
    fprintf(stderr, "ERROR: Unsupported CUMAC_EXEC_MODE=%s. Supported values: both, gpu.\n", value.c_str());
    exit(1);
}

const char* execModeToString(const ExecMode mode)
{
    return mode == ExecMode::Gpu ? "gpu" : "both";
}

size_t allocSolBytes(const cumacCellGrpPrms* cellGrpPrms)
{
    if (cellGrpPrms == nullptr) {
        return 0U;
    }
    if (cellGrpPrms->allocType == 1U) {
        return static_cast<size_t>(2U) * cellGrpPrms->nUe * sizeof(int16_t);
    }
    return static_cast<size_t>(cellGrpPrms->totNumCell) * cellGrpPrms->nPrbGrp * sizeof(int16_t);
}

unsigned long long countAllocatedPrgs(const cumacSchdSol* schdSol, const cumacCellGrpPrms* cellGrpPrms)
{
    if (schdSol == nullptr || cellGrpPrms == nullptr || schdSol->allocSol == nullptr) {
        return 0ULL;
    }

    unsigned long long allocatedPrgs = 0ULL;
    if (cellGrpPrms->allocType == 1U) {
        const int maxPrg = static_cast<int>(cellGrpPrms->nPrbGrp);
        for (uint32_t uIdx = 0; uIdx < cellGrpPrms->nUe; ++uIdx) {
            const int startPrg = static_cast<int>(schdSol->allocSol[2U * uIdx]);
            const int endPrg = static_cast<int>(schdSol->allocSol[2U * uIdx + 1U]);
            if (startPrg < 0 || endPrg <= startPrg) {
                continue;
            }
            const int clippedStart = std::max(startPrg, 0);
            const int clippedEnd = std::min(endPrg, maxPrg);
            if (clippedEnd > clippedStart) {
                allocatedPrgs += static_cast<unsigned long long>(clippedEnd - clippedStart);
            }
        }
        return allocatedPrgs;
    }

    const uint32_t totalCells = cellGrpPrms->totNumCell;
    const uint32_t prgCount = cellGrpPrms->nPrbGrp;
    for (uint32_t prgIdx = 0; prgIdx < prgCount; ++prgIdx) {
        for (uint32_t cellIdx = 0; cellIdx < totalCells; ++cellIdx) {
            const size_t allocIdx = static_cast<size_t>(prgIdx) * totalCells + cellIdx;
            if (schdSol->allocSol[allocIdx] >= 0) {
                allocatedPrgs += 1ULL;
            }
        }
    }
    return allocatedPrgs;
}

void copyGpuUeSelectionToCpu(cumacSchdSol* schdSolCpu,
                             cumacSchdSol* schdSolGpu,
                             const cumacCellGrpPrms* cellGrpPrmsCpu,
                             cudaStream_t stream)
{
    if (schdSolCpu == nullptr || schdSolGpu == nullptr || cellGrpPrmsCpu == nullptr) {
        return;
    }
    const size_t ueSelBytes = static_cast<size_t>(cellGrpPrmsCpu->nUe) * sizeof(uint16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolCpu->setSchdUePerCellTTI, schdSolGpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));
}

void populateType0AllUeSelection(cumacSchdSol* schdSolCpu,
                                 cumacSchdSol* schdSolGpu,
                                 const cumacCellGrpPrms* cellGrpPrmsCpu,
                                 cudaStream_t stream)
{
    if (schdSolCpu == nullptr || schdSolGpu == nullptr || cellGrpPrmsCpu == nullptr) {
        return;
    }
    const uint32_t nSchedSlots = cellGrpPrmsCpu->nUe;
    const uint32_t nActiveUe = cellGrpPrmsCpu->nActiveUe;
    for (uint32_t uIdx = 0; uIdx < nSchedSlots; ++uIdx) {
        schdSolCpu->setSchdUePerCellTTI[uIdx] = (uIdx < nActiveUe) ? static_cast<uint16_t>(uIdx) : 0xFFFF;
    }
    const size_t ueSelBytes = static_cast<size_t>(nSchedSlots) * sizeof(uint16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));
}

void copyGpuSchedulingSolutionToCpu(cumacSchdSol* schdSolCpu,
                                    cumacSchdSol* schdSolGpu,
                                    const cumacCellGrpPrms* cellGrpPrmsCpu,
                                    cudaStream_t stream)
{
    if (schdSolCpu == nullptr || schdSolGpu == nullptr || cellGrpPrmsCpu == nullptr) {
        return;
    }
    const size_t ueSelBytes = static_cast<size_t>(cellGrpPrmsCpu->nUe) * sizeof(uint16_t);
    const size_t allocBytes = allocSolBytes(cellGrpPrmsCpu);
    const size_t layerBytes = static_cast<size_t>(cellGrpPrmsCpu->nUe) * sizeof(uint8_t);
    const size_t mcsBytes = static_cast<size_t>(cellGrpPrmsCpu->nUe) * sizeof(int16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolCpu->setSchdUePerCellTTI, schdSolGpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolCpu->allocSol, schdSolGpu->allocSol, allocBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolCpu->layerSelSol, schdSolGpu->layerSelSol, layerBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolCpu->mcsSelSol, schdSolGpu->mcsSelSol, mcsBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));
}

bool assocToCellOnline(const cumacCellGrpPrms* cellGrpPrmsCpu, uint32_t cellId, uint32_t activeUeId)
{
    if (cellGrpPrmsCpu->cellAssocActUe != nullptr) {
        return cellGrpPrmsCpu->cellAssocActUe[cellId * cellGrpPrmsCpu->nActiveUe + activeUeId] != 0;
    }
    if (cellGrpPrmsCpu->nCell > 0 && (cellGrpPrmsCpu->nActiveUe % cellGrpPrmsCpu->nCell) == 0) {
        const uint32_t uePerCell = cellGrpPrmsCpu->nActiveUe / cellGrpPrmsCpu->nCell;
        return (activeUeId / uePerCell) == cellId;
    }
    return false;
}

void applyOnlineActionToSchedule(const cumac::online::StepAction& action,
                                 cumacSchdSol* schdSolCpu,
                                 cumacSchdSol* schdSolGpu,
                                 cumacCellGrpPrms* cellGrpPrmsCpu,
                                 cudaStream_t stream)
{
    const uint32_t nCell = cellGrpPrmsCpu->nCell;
    const uint32_t nSchedUe = cellGrpPrmsCpu->nUe;
    const uint32_t nActiveUe = cellGrpPrmsCpu->nActiveUe;
    const uint32_t nPrg = cellGrpPrmsCpu->nPrbGrp;
    const uint32_t nTotCell = cellGrpPrmsCpu->totNumCell;
    const uint32_t slotsPerCell = std::max<uint32_t>(1U, nSchedUe / std::max<uint32_t>(1U, nCell));

    std::fill(schdSolCpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI + nSchedUe, 0xFFFF);
    std::fill(schdSolCpu->allocSol, schdSolCpu->allocSol + (nTotCell * nPrg), static_cast<int16_t>(-1));

    std::vector<uint8_t> usedUe(nActiveUe, 0U);
    const uint32_t ueLen = std::min<uint32_t>(nSchedUe, static_cast<uint32_t>(action.ueAction.size()));
    for (uint32_t slot = 0; slot < ueLen; ++slot) {
        const int32_t ue = action.ueAction[slot];
        if (ue < 0 || static_cast<uint32_t>(ue) >= nActiveUe) {
            continue;
        }
        const uint32_t cIdx = std::min<uint32_t>(slot / slotsPerCell, nCell - 1U);
        if (!assocToCellOnline(cellGrpPrmsCpu, cIdx, static_cast<uint32_t>(ue))) {
            continue;
        }
        if (usedUe[static_cast<uint32_t>(ue)] != 0U) {
            continue;
        }
        usedUe[static_cast<uint32_t>(ue)] = 1U;
        schdSolCpu->setSchdUePerCellTTI[slot] = static_cast<uint16_t>(ue);
    }

    const uint32_t actionAllocLen = nTotCell * nPrg;
    const uint32_t prgLen = std::min<uint32_t>(actionAllocLen, static_cast<uint32_t>(action.prgAction.size()));
    for (uint32_t idx = 0; idx < prgLen; ++idx) {
        const int16_t v = action.prgAction[idx];
        if (v < 0) {
            continue;
        }
        const uint32_t cIdx = idx % nCell;
        const uint32_t prgIdx = idx / nTotCell;
        if (cellGrpPrmsCpu->prgMsk != nullptr && cellGrpPrmsCpu->prgMsk[cIdx] != nullptr) {
            if (cellGrpPrmsCpu->prgMsk[cIdx][prgIdx] == 0) {
                continue;
            }
        }
        const uint32_t slot = static_cast<uint32_t>(v);
        if (slot >= nSchedUe) {
            continue;
        }
        if ((slot / slotsPerCell) != cIdx) {
            continue;
        }
        if (schdSolCpu->setSchdUePerCellTTI[slot] == 0xFFFF) {
            continue;
        }
        schdSolCpu->allocSol[idx] = v;
    }

    const size_t ueSelBytes = static_cast<size_t>(nSchedUe) * sizeof(uint16_t);
    const size_t allocBytes = static_cast<size_t>(nTotCell) * nPrg * sizeof(int16_t);
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->setSchdUePerCellTTI, schdSolCpu->setSchdUePerCellTTI, ueSelBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaMemcpyAsync(
        schdSolGpu->allocSol, schdSolCpu->allocSol, allocBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));
}

cumac::online::StepState buildOnlineState(int tti,
                                          bool done,
                                          const cumac::online::OnlineFeatureCodec& codec,
                                          const std::vector<float>& obsCell,
                                          const std::vector<float>& obsUe,
                                          const std::vector<float>& obsEdgeAttr,
                                          const std::vector<uint8_t>& maskUe,
                                          const std::vector<uint8_t>& maskCellUe,
                                          const std::vector<uint8_t>& maskPrg,
                                          const cumac::online::OnlineFeatureCodec::RewardTerms& reward)
{
    cumac::online::StepState state;
    state.header.tti = tti;
    state.header.done = done ? 1U : 0U;
    state.header.rewardScalar = reward.scalar;
    state.header.rewardTerms[0] = reward.throughputMbps;
    state.header.rewardTerms[1] = reward.totalBufferMb;
    state.header.rewardTerms[2] = reward.tbErrRate;
    state.header.rewardTerms[3] = reward.fairnessJain;
    state.header.dims.nCell = codec.nCell();
    state.header.dims.nActiveUe = codec.nActiveUe();
    state.header.dims.nSchedUe = codec.nSchedUe();
    state.header.dims.nTotCell = codec.nTotCell();
    state.header.dims.nPrg = codec.nPrg();
    state.header.dims.nEdges = codec.nEdges();
    state.header.dims.allocType = codec.allocType();
    state.header.dims.actionAllocLen = codec.actionAllocLen();

    state.obsCellFeatures = obsCell;
    state.obsUeFeatures = obsUe;
    state.obsEdgeIndex = codec.edgeIndex();
    state.obsEdgeAttr = obsEdgeAttr;
    state.actionMaskUe = maskUe;
    state.actionMaskCellUe = maskCellUe;
    state.actionMaskPrgCell = maskPrg;
    return state;
}

} // namespace


int main(int argc, char* argv[]) 
{
  int iArg = 1;

  // indicator for DL/UL
  std::string dlUlIndStr = std::string();
  uint8_t DL = 0;

  // indicator for RR baseline / PF reference-check mode
  std::string cpuRrRefIndStr = std::string();
  uint8_t baseline = 0;

  // indicator for using FP16 PRG allocation kernel
  std::string hpIndStr = std::string();
  uint8_t halfPrecision = 0;

  // indicator for saving TV before return
  std::string stvIndStr = std::string();
  uint8_t saveTv = 0;

  // fast fading mode: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)
  uint8_t fastFadingMode = 0;
  uint8_t useCustomUePrg = 0;

  float percent_ue_traffic = 0.0;
  float packetSizeBytes = 5000.0;

  while(iArg < argc) {
    if('-' == argv[iArg][0]) {
      switch(argv[iArg][1]) {
        case 'd': // indicator of scheduler modules being called
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No DL/UL indicator given.\n");
              exit(1);
            } else {
              dlUlIndStr.assign(argv[iArg++]);
            }
            break;
        case 'g':
            if((++iArg >= argc || (1 != sscanf(argv[iArg], "%f", &percent_ue_traffic)) || (percent_ue_traffic < 0) || (percent_ue_traffic > 100) )) {
              fprintf(stderr, "ERROR: Percent of UEs with traffic generation must be specified.\n");
              exit(1);
            }
            ++iArg;
            break;
        case 'r':
            if((++iArg >= argc || (1 != sscanf(argv[iArg], "%f", &packetSizeBytes)) || (packetSizeBytes < 0))) {
              fprintf(stderr, "ERROR: Invalid packet size for traffic generation.\n");
              exit(1);
            }
            ++iArg;
            break;
        case 'p': // indicator for using FP16 PRG allocation kernel
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No indicator for using FP16 PRG allocation kernel given.\n");
              exit(1);
            } else {
              hpIndStr.assign(argv[iArg++]);
            }
            break;
        case 't': // indicator for saving TV before return
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No indicator for saving TV before return given.\n");
              exit(1);
            } else {
              stvIndStr.assign(argv[iArg++]);
            }
            break;
        case 'b': // indicator of scheduler modules being called
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No RR baseline/reference-check indicator given.\n");
              exit(1);
            } else {
              cpuRrRefIndStr.assign(argv[iArg++]);
            }
            break;
        case 'f': // set fast fading mode
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhi", &fastFadingMode)) || (fastFadingMode < 0) || (fastFadingMode > 4) )
            {
                fprintf(stderr, "ERROR: Unsupported fast fading mode.\n");
                exit(1);
            }
            ++iArg;
            break;
        case 'x': // use custom UE+PRG scheduler
            if ((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhi", &useCustomUePrg)) ||
                (useCustomUePrg != 0 && useCustomUePrg != 1)) {
                fprintf(stderr, "ERROR: -x must be 0 or 1.\n");
                exit(1);
            }
            ++iArg;
            break;
        case 'h': // print help usage
            usage();
            exit(0);
            break;
        default:
            fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
            usage();
            exit(1);
            break;
      }
    } else {
      fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
      exit(1);
    }
  }

  // set GPU device with fallback mechanism
  int deviceCount{};
  CUDA_CHECK_ERR(cudaGetDeviceCount(&deviceCount));
  
  unsigned my_dev = gpuDeviceIdx;
  if (static_cast<int>(gpuDeviceIdx) >= deviceCount) {
      printf("WARNING: Requested GPU device %u exceeds available device count (%d). Falling back to GPU device 0.\n", 
             gpuDeviceIdx, deviceCount);
      my_dev = 0;
  }
  
  CUDA_CHECK_ERR(cudaSetDevice(my_dev));
  printf("cuMAC multi-cell scheduler: Running on GPU device %d (total devices: %d)\n", 
         my_dev, deviceCount);

  // setup randomness seed
  srand(seedConst);

  // create stream
  cudaStream_t cuStrmMain;
  CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

  if (dlUlIndStr.size() == 0) {
      DL = 1;
  } else {
      DL = static_cast<uint8_t>(dlUlIndStr[0] - '0');
  }
  if (DL == 1) {
      printf("cuMAC scheduler pipeline test: Downlink\n");
  } else {
      printf("cuMAC scheduler pipeline test: Uplink\n");
  }

  if (hpIndStr.size() == 0) {
      halfPrecision = 0;
  } else {
      halfPrecision = static_cast<uint8_t>(hpIndStr[0] - '0');
  }
  if (halfPrecision == 1) {
      printf("cuMAC scheduler pipeline test: FP16 half-precision kernels\n");
  } else {
      printf("cuMAC scheduler pipeline test: FP32 kernels\n");
  }

  if (stvIndStr.size() == 0) {
      saveTv = 0;
  } else {
      saveTv = static_cast<uint8_t>(stvIndStr[0] - '0');
  }

  if (cpuRrRefIndStr.size() == 0) {
      baseline = 0;
  } else {
      baseline = static_cast<uint8_t>(cpuRrRefIndStr[0] - '0');
  }
  if (baseline == 1) {
    printf("cuMAC scheduler pipeline test: RR baseline on CPU and GPU paths\n");
  } else {
    printf("cuMAC scheduler pipeline test: PF reference check (CPU and GPU)\n");
  }

  switch (fastFadingMode)
  {
    case 0:
      printf("cuMAC scheduler pipeline test: use Rayleigh fading\n");
      break;
    case 1:
      printf("cuMAC scheduler pipeline test: use GPU TDL channel, CFR on Prg\n");
      break;
    case 2:
      printf("cuMAC scheduler pipeline test: use GPU TDL channel, CFR on Sc and Prg\n");
      break;
    case 3:
      printf("cuMAC scheduler pipeline test: use GPU CDL channel, CFR on Prg\n");
      break;
    case 4:
      printf("cuMAC scheduler pipeline test: use GPU CDL channel, CFR on Sc and Prg\n");
      break;
    default:
      fprintf(stderr, "ERROR: Unsupported fast fading mode.\n");
      exit(1);
  }

  if (useCustomUePrg == 1) {
      printf("cuMAC scheduler pipeline test: custom UE selection + PRG allocation enabled\n");
  } else {
      printf("cuMAC scheduler pipeline test: native UE selection + PRG allocation\n");
  }

  auto getEnvInt = [](const char* name, int defaultValue) -> int {
      const char* v = std::getenv(name);
      if (v == nullptr || v[0] == '\0') {
          return defaultValue;
      }
      int parsed = std::atoi(v);
      return parsed >= 0 ? parsed : defaultValue;
  };
  auto getEnvFloat = [](const char* name, float defaultValue) -> float {
      const char* v = std::getenv(name);
      if (v == nullptr || v[0] == '\0') {
          return defaultValue;
      }
      char* end = nullptr;
      const float parsed = std::strtof(v, &end);
      if (end == v || (end != nullptr && *end != '\0') || parsed < 0.0f) {
          return defaultValue;
      }
      return parsed;
  };
  auto getEnvString = [](const char* name, const char* defaultValue) -> std::string {
      const char* v = std::getenv(name);
      if (v == nullptr || v[0] == '\0') {
          return std::string(defaultValue);
      }
      return std::string(v);
  };

  const bool compactTtiLog = getEnvInt("CUMAC_COMPACT_TTI_LOG", 1) != 0;
  const int progressTtiInterval = std::max(1, getEnvInt("CUMAC_PROGRESS_TTI_INTERVAL", 100));
  int compareTtiInterval = getEnvInt("CUMAC_COMPARE_TTI_INTERVAL", useCustomUePrg ? 0 : 1);
  const float trafficArrivalRate = getEnvFloat("CUMAC_TRAFFIC_ARRIVAL_RATE", 1.0f);
  const bool replayDumpEnabled = getEnvInt("CUMAC_RL_REPLAY_DUMP", 0) != 0;
  const std::string replayDir = getEnvString("CUMAC_RL_REPLAY_DIR", "./replay");
  const std::string replayScenario = getEnvString("CUMAC_RL_REPLAY_SCENARIO", "default");
  const std::string replayPolicy =
      getEnvString("CUMAC_CUSTOM_POLICY", (useCustomUePrg == 1) ? "gnnrl" : "native");
  const ExecMode execMode = parseExecMode(getEnvString("CUMAC_EXEC_MODE", "both"));
  const bool runCpuReferencePath = execMode == ExecMode::Both;
  const bool onlineBridgeEnabled = getEnvInt("CUMAC_ONLINE_BRIDGE", 0) != 0;
  const std::string onlineSocketPath = getEnvString("CUMAC_ONLINE_SOCKET", "/tmp/cumac_stageb_online.sock");
  const bool onlinePersistentMode = onlineBridgeEnabled && (getEnvInt("CUMAC_ONLINE_PERSISTENT", 1) != 0);
  if (compareTtiInterval < 0) {
      compareTtiInterval = 0;
  }
  if (onlineBridgeEnabled) {
      compareTtiInterval = 0;
      printf("cuMAC scheduler pipeline test: online bridge enabled, socket=%s\n", onlineSocketPath.c_str());
      printf("cuMAC scheduler pipeline test: online persistent mode=%s\n", onlinePersistentMode ? "on" : "off");
  }
  if (!runCpuReferencePath && useCustomUePrg == 1) {
      fprintf(stderr, "ERROR: CUMAC_EXEC_MODE=gpu is not supported with custom UE+PRG mode.\n");
      return 1;
  }
  if (!runCpuReferencePath && onlineBridgeEnabled) {
      fprintf(stderr, "ERROR: CUMAC_EXEC_MODE=gpu is not supported with online bridge mode.\n");
      return 1;
  }
  if (!runCpuReferencePath) {
      setenv("CUMAC_TTI_KPI_LOG_INTERVAL", "0", 1);
  }
  printf("cuMAC scheduler pipeline test: execution mode=%s\n", execModeToString(execMode));
  if (!runCpuReferencePath) {
      printf("cuMAC scheduler pipeline test: CPU reference scheduler path disabled; CPU KPI bookkeeping reuses GPU decisions\n");
      printf("cuMAC scheduler pipeline test: per-TTI throughput prints disabled in gpu-only mode; progress bar remains enabled\n");
  }

  std::ofstream nullStdout;
  std::streambuf* savedCoutBuf = nullptr;
  if (compactTtiLog) {
      nullStdout.open("/dev/null");
      savedCoutBuf = std::cout.rdbuf(nullStdout.rdbuf());
      printf("cuMAC scheduler pipeline test: compact per-TTI logging enabled\n");
      printf("cuMAC scheduler pipeline test: progress interval=%d, compare interval=%d (TTI)\n",
             progressTtiInterval, compareTtiInterval);
  }

  // specify scheduler type
  uint8_t schedulerType = 1; // multi-cell scheduler

  // specify channel matrix access type: 0 - row major, 1 - column major
  uint8_t columnMajor = 1;

  // use lightWeight kernels
  uint8_t lightWeight = 0;

  // percentage of SMs to determine the number of thread blocks used in light-weight kernels
  float percSmNumThrdBlk = 2.0;

  // CSI update indicator
  uint8_t csiUpdate = 1;

  // create network 
  network* net = new cumac::network(DL, schedulerType, 1 /*fixCellAssoc*/, fastFadingMode, percent_ue_traffic > 0.0, cuStrmMain);

  net->genNetTopology();
  net->genLSFading();

  // create API
  net->createAPI();


  // Initialize traffic service
  printf("Traffic configuration: active_ue_percent=%.3f packet_bytes=%.3f arrival_rate_pkt_per_tti=%.3f\n",
         percent_ue_traffic, packetSizeBytes, trafficArrivalRate);
  TrafficType basic_traffic(packetSizeBytes, 0, trafficArrivalRate);
  int traffic_num_flows = static_cast<int>(totNumUesConst * percent_ue_traffic / 100.0);
  traffic_num_flows = std::max(0, std::min(traffic_num_flows, static_cast<int>(net->cellGrpPrmsGpu.get()->nActiveUe)));
  if (percent_ue_traffic > 0.0f && traffic_num_flows == 0) {
    traffic_num_flows = 1;
  }
  TrafficConfig traf_cfg(basic_traffic, traffic_num_flows);
  //TrafficType low_traffic(100, 0, 1);
  //traf_cfg.AddFlows(low_traffic,totNumUesConst/2);
  std::unique_ptr<TrafficService> trafSvc = std::make_unique<TrafficService>(traf_cfg,net->cellGrpUeStatusCpu.get(),net->cellGrpUeStatusGpu.get());
  trafSvc->SetSlotDurationMs(slotDurationConst * 1.0e3);

  // determine the number of interfering cells
  uint16_t nInterfCell = net->simParam.get()->totNumCell - net->cellGrpPrmsGpu.get()->nCell;
  printf("Cluster configuration: coordinated cells=%u, interferer cells=%u, total cells=%u\n",
         net->cellGrpPrmsGpu.get()->nCell, nInterfCell, net->simParam.get()->totNumCell);

  // post-eq SINR calculation
  cumac::mcSinrCalHndl_t mcSinrCalGpu = new cumac::multiCellSinrCal(net->cellGrpPrmsGpu.get());

  cumac::mcUeSelHndl_t mcUeSelGpu = nullptr;
  cumac::mcRRUeSelHndl_t rrUeSelGpu = nullptr;
  cumac::mcSchdHndl_t mcSchGpu = nullptr;
  cumac::mcRRSchdHndl_t rrSchGpu = nullptr;
  if (baseline == 1) {
    printf("Using GPU RR UE selection\n");
    rrUeSelGpu = new cumac::multiCellRRUeSel(net->cellGrpPrmsGpu.get());

    printf("Using GPU RR UE scheduler\n");
    rrSchGpu = new cumac::multiCellRRScheduler(net->cellGrpPrmsGpu.get());
  } else {
    printf("Using GPU multi-cell PF UE selection\n");
    mcUeSelGpu = new cumac::multiCellUeSelection(net->cellGrpPrmsGpu.get());

    printf("Using GPU multi-cell PF scheduler\n");
    mcSchGpu = new cumac::multiCellScheduler(net->cellGrpPrmsGpu.get());
  }
  std::unique_ptr<cumac::CustomUePrgScheduler> customUePrgScheduler;
  if (useCustomUePrg == 1) {
    customUePrgScheduler = std::make_unique<cumac::CustomUePrgScheduler>();
  }
  std::unique_ptr<cumac::ReplayWriter> replayWriter;
  if (replayDumpEnabled) {
    cumac::ReplayWriter::Config replayCfg;
    replayCfg.enabled = true;
    replayCfg.outDir = replayDir;
    replayCfg.scenario = replayScenario;
    replayCfg.policy = replayPolicy;
    replayCfg.seed = seedConst;
    replayWriter = std::make_unique<cumac::ReplayWriter>(replayCfg);
    if (!replayWriter->initialize(net->cellGrpPrmsCpu.get())) {
      replayWriter.reset();
      printf("Replay dump: disabled (initialization failed)\n");
    } else {
      printf("Replay dump: enabled, out_dir=%s\n", replayDir.c_str());
    }
  }

  std::unique_ptr<cumac::online::OnlineBridgeServer> onlineBridge;
  cumac::online::OnlineFeatureCodec onlineCodec;
  cumac::online::ResetReqPayload resetReq {};
  int onlineEpisodeHorizon = numSimChnRlz;
  if (onlineBridgeEnabled) {
    if (!onlineCodec.initialize(net->cellGrpPrmsCpu.get())) {
      fprintf(stderr, "ERROR: Failed to initialize online feature codec.\n");
      return 1;
    }
    if (onlineCodec.allocType() != 0U || onlineCodec.nTotCell() != onlineCodec.nCell()) {
      fprintf(stderr, "ERROR: Online bridge currently requires allocType=0 and nTotCell==nCell.\n");
      return 1;
    }
    onlineBridge = std::make_unique<cumac::online::OnlineBridgeServer>(onlineSocketPath);
    if (!onlineBridge->initialize()) {
      fprintf(stderr, "ERROR: Online bridge socket initialization failed.\n");
      return 1;
    }
    if (!onlineBridge->recvResetReq(&resetReq)) {
      fprintf(stderr, "ERROR: Online bridge failed to receive reset request.\n");
      return 1;
    }
    const int requestedHorizon = std::max(1, resetReq.episodeHorizon);
    onlineEpisodeHorizon = onlinePersistentMode
                               ? std::max(1, numSimChnRlz)
                               : std::min(std::max(1, numSimChnRlz), requestedHorizon);
    if (onlinePersistentMode) {
      printf("Online bridge reset: seed=%d requested_horizon=%d ignored (persistent), ring_horizon=%d\n",
             resetReq.seed, requestedHorizon, onlineEpisodeHorizon);
    } else if (requestedHorizon != onlineEpisodeHorizon) {
      printf("Online bridge reset: seed=%d requested_horizon=%d adjusted_horizon=%d\n",
             resetReq.seed, requestedHorizon, onlineEpisodeHorizon);
    } else {
      printf("Online bridge reset: seed=%d horizon=%d\n", resetReq.seed, onlineEpisodeHorizon);
    }
  }

#ifdef PDSCH_
  // GPU layer selection
  cumac::mcLayerSelHndl_t mcLayerSelGpu = new cumac::multiCellLayerSel(net->cellGrpPrmsGpu.get());

  // GPU MCS selection
  std::unique_ptr<cumac::mcsSelectionLUT> mcsSelGpu = std::make_unique<cumac::mcsSelectionLUT>(net->cellGrpPrmsGpu.get(), cuStrmMain);
  CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

  // CPU layer selection
  cumac::mcLayerSelCpuHndl_t mcLayerSelCpu = nullptr;
  if (runCpuReferencePath) {
    mcLayerSelCpu = new cumac::multiCellLayerSelCpu(net->cellGrpPrmsCpu.get());
  }

  // CPU MCS selection
  std::unique_ptr<cumac::mcsSelectionLUTCpu> mcsSelCpu;
  if (runCpuReferencePath) {
    mcsSelCpu = std::make_unique<cumac::mcsSelectionLUTCpu>(net->cellGrpPrmsCpu.get());
  }
#endif

  rrUeSelCpuHndl_t  rrUeSelCpu  = nullptr;
  rrSchdCpuHndl_t   rrSchCpu    = nullptr;
  mcUeSelCpuHndl_t  mcUeSelCpu  = nullptr;
  mcSchdCpuHndl_t   mcSchCpu    = nullptr;
  if (runCpuReferencePath) {
    if (baseline == 1) { // RR baseline
      printf("Using CPU RR UE selection\n");
      rrUeSelCpu = new roundRobinUeSelCpu(net->cellGrpPrmsCpu.get());

      printf("Using CPU RR UE scheduler\n");
      rrSchCpu = new roundRobinSchedulerCpu(net->cellGrpPrmsCpu.get());
    } else { // CPU reference check
      printf("Using CPU multi-cell PF UE selection\n");
      mcUeSelCpu = new multiCellUeSelectionCpu(net->cellGrpPrmsCpu.get());

      printf("Using CPU multi-cell PF scheduler\n");
      mcSchCpu = new multiCellSchedulerCpu(net->cellGrpPrmsCpu.get());
    }
  }
  
  // create SVD precoder
  svdPrdHndl_t svdPrd = nullptr;

  if (prdSchemeConst) {
    svdPrd = new svdPrecoding(net->cellGrpPrmsGpu.get());

    // Setup SVD precoder
    svdPrd->setup(net->cellGrpPrmsGpu.get(), cuStrmMain);
  }

  // begin to perform scheduling for numSimChnRlz TTIs
  const int nActiveUe = net->cellGrpPrmsCpu->nActiveUe;
  const int nUeSched = net->cellGrpPrmsCpu->nUe;
  const int nUeAnt = net->cellGrpPrmsCpu->nUeAnt;
  const bool type0AllUeScheduling = net->cellGrpPrmsCpu->allocType == 0;
  std::vector<double> ueMcsSum(nActiveUe, 0.0);
  std::vector<uint32_t> ueMcsCnt(nActiveUe, 0);
  std::vector<unsigned long long> ueTbTxPkts(nActiveUe, 0ULL);
  std::vector<unsigned long long> ueTbSuccPkts(nActiveUe, 0ULL);
  std::vector<unsigned long long> ueTbErrCount(nActiveUe, 0ULL);
  std::vector<double> uePredBlerSum(nActiveUe, 0.0);
  std::vector<uint32_t> uePredBlerCnt(nActiveUe, 0U);
  std::vector<double> ueWbSinrLinSumAll(nActiveUe, 0.0);
  std::vector<uint32_t> ueWbSinrCntAll(nActiveUe, 0U);
  std::vector<double> ueWbSinrLinSumSched(nActiveUe, 0.0);
  std::vector<uint32_t> ueWbSinrCntSched(nActiveUe, 0U);
  std::vector<unsigned long long> macBufferBeforeSched(nActiveUe, 0ULL);
  std::vector<unsigned long long> servedBytesThisTti(nActiveUe, 0ULL);
  unsigned long long totalAllocatedPrgCount = 0ULL;
  unsigned long long totalPrgCapacity = 0ULL;
  int executedTtiCount = 0;
  bool onlineResetSent = false;
  bool onlineCloseRequested = false;
  const bool stageTraceEnabled = getEnvInt("CUMAC_TTI_STAGE_TRACE", 0) != 0;
  auto stageTrace = [&](int tti, const char* stage) {
    if (stageTraceEnabled) {
      printf("TTI_STAGE t=%d %s\n", tti, stage);
      fflush(stdout);
    }
  };
  auto printTtiProgress = [&](int currentTti) {
    if (onlineBridgeEnabled && onlinePersistentMode) {
      printf("TTI_PROGRESS %d\n", currentTti);
      return;
    }
    const int totalTti = std::max(1, numSimChnRlz);
    const int finalTti = totalTti - 1;
    const double ratio = finalTti > 0 ? static_cast<double>(currentTti) / static_cast<double>(finalTti) : 1.0;
    if (!runCpuReferencePath) {
      constexpr int barWidth = 24;
      const int filled = std::max(0, std::min(barWidth, static_cast<int>(std::lround(ratio * barWidth))));
      const std::string bar(static_cast<size_t>(filled), '#');
      const std::string gap(static_cast<size_t>(barWidth - filled), '.');
      const bool isFinalUpdate = !onlineBridgeEnabled && currentTti >= finalTti;
      printf("\rTTI_PROGRESS [%s%s] %d/%d (%5.1f%%)%s",
             bar.c_str(),
             gap.c_str(),
             currentTti,
             finalTti,
             100.0 * ratio,
             isFinalUpdate ? "\n" : "");
      fflush(stdout);
    } else {
      printf("TTI_PROGRESS %d/%d\n", currentTti, finalTti);
    }
  };
  for (int t = 0;; ++t) {
    if (!onlineBridgeEnabled && t >= numSimChnRlz) {
      break;
    }
    const int slotIdx = (onlineBridgeEnabled && onlinePersistentMode) ? (t % numSimChnRlz) : t;
    if (compactTtiLog) {
      if ((t % progressTtiInterval) == 0 || (!onlineBridgeEnabled && t == (numSimChnRlz - 1))) {
        printTtiProgress(t);
      }
    } else {
      std::cout<<"~~~~~~~~~~~~~~~~~TTI "<<t<<"~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    }
#ifdef periodicLightWt_
    if (t%16 == 0) {
      lightWeight = 0;
      csiUpdate = 1;
    } else {
      lightWeight = 1;
      csiUpdate = 0;
    }
#endif

    // generate user data traffic
    stageTrace(t, "before_traffic_update");
    trafSvc->Update();
    stageTrace(t, "after_traffic_update");
    if (net->cellGrpUeStatusCpu.get()->bufferSize != nullptr) {
      for (int ueId = 0; ueId < nActiveUe; ++ueId) {
        macBufferBeforeSched[ueId] = static_cast<unsigned long long>(net->cellGrpUeStatusCpu.get()->bufferSize[ueId]);
      }
    } else {
      std::fill(macBufferBeforeSched.begin(), macBufferBeforeSched.end(), 0ULL);
    }

    // generate channel
    if(net->execStatus.get()->channelRenew)
    {
      stageTrace(t, "before_gen_fast_fading_gpu");
      net->genFastFadingGpu(slotIdx);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      stageTrace(t, "after_gen_fast_fading_gpu");
      std::cout<<"GPU channel generated"<<std::endl;

      if (prdSchemeConst) {
        // run SVD precoder
        svdPrd->run(net->cellGrpPrmsGpu.get());
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        std::cout<<"SVD precoder and singular values computed"<<std::endl;
      }
    }

    // setup API 
    stageTrace(t, "before_setup_api");
    net->setupAPI(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    stageTrace(t, "after_setup_api");
    std::cout<<"API setup completed"<<std::endl;

#ifdef periodicLightWt_
    if (csiUpdate == 1) {
#endif
      // GPU post-eq SINR calculation
      stageTrace(t, "before_sinr_setup");
      mcSinrCalGpu->setup(net->cellGrpPrmsGpu.get(), columnMajor, cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      stageTrace(t, "after_sinr_setup");
      std::cout<<"CSI update: subband SINR calculation setup completed"<<std::endl;

      stageTrace(t, "before_sinr_run_subband");
      mcSinrCalGpu->run(cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      stageTrace(t, "after_sinr_run_subband");
      std::cout<<"CSI update: subband SINR calculation run completed"<<std::endl;
      
      //if (t == 0)
      //  mcSinrCalGpu->debugLog();

      stageTrace(t, "before_sinr_setup_wb");
      mcSinrCalGpu->setup_wbSinr(net->cellGrpPrmsGpu.get(), cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      stageTrace(t, "after_sinr_setup_wb");
      std::cout<<"CSI update: wideband SINR calculation setup completed"<<std::endl;

      stageTrace(t, "before_sinr_run_wb");
      mcSinrCalGpu->run(cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      stageTrace(t, "after_sinr_run_wb");
      std::cout<<"CSI update: wideband SINR calculation run completed"<<std::endl;

      net->cpySinrGpu2Cpu();
      std::cout<<"CSI update: subband and wideband SINRS copied to CPU structures"<<std::endl;
      if (net->cellGrpPrmsCpu.get()->wbSinr != nullptr) {
        for (int ueId = 0; ueId < nActiveUe; ++ueId) {
          double sinrLin = 0.0;
          for (int antIdx = 0; antIdx < nUeAnt; ++antIdx) {
            const double antSinr = static_cast<double>(net->cellGrpPrmsCpu.get()->wbSinr[ueId * nUeAnt + antIdx]);
            sinrLin += std::max(antSinr, 1.0e-9);
          }
          sinrLin /= std::max(1, nUeAnt);
          ueWbSinrLinSumAll[ueId] += sinrLin;
          ueWbSinrCntAll[ueId] += 1U;
        }
      }
#ifdef periodicLightWt_
    }
#endif    

    std::vector<float> onlineObsCell;
    std::vector<float> onlineObsUe;
    std::vector<float> onlineObsEdgeAttr;
    std::vector<uint8_t> onlineMaskUe;
    std::vector<uint8_t> onlineMaskCellUe;
    std::vector<uint8_t> onlineMaskPrg;

    if (onlineBridgeEnabled) {
      onlineCodec.buildObservation(
          net->cellGrpUeStatusCpu.get(), net->cellGrpPrmsCpu.get(), onlineObsCell, onlineObsUe, onlineObsEdgeAttr);
      onlineCodec.buildActionMask(
          net->cellGrpUeStatusCpu.get(), net->cellGrpPrmsCpu.get(), onlineMaskUe, onlineMaskCellUe, onlineMaskPrg);

      if (!onlineResetSent) {
        const cumac::online::OnlineFeatureCodec::RewardTerms zeroReward {};
        const cumac::online::StepState resetState = buildOnlineState(
            t,
            false,
            onlineCodec,
            onlineObsCell,
            onlineObsUe,
            onlineObsEdgeAttr,
            onlineMaskUe,
            onlineMaskCellUe,
            onlineMaskPrg,
            zeroReward);
        if (!onlineBridge->sendResetRsp(resetState)) {
          fprintf(stderr, "ERROR: Failed to send online reset response.\n");
          return 1;
        }
        onlineResetSent = true;
      }
    }

    if (replayWriter) {
      replayWriter->capturePreActionObs(net->cellGrpUeStatusCpu.get(), net->cellGrpPrmsCpu.get(), slotIdx);
    }

    if (onlineBridgeEnabled) {
      cumac::online::StepAction actionReq;
      bool closeReq = false;
      if (!onlineBridge->recvStepReq(&actionReq, &closeReq)) {
        fprintf(stderr, "ERROR: Failed to receive online step request.\n");
        return 1;
      }
      if (closeReq) {
        onlineCloseRequested = true;
        break;
      }
      applyOnlineActionToSchedule(
          actionReq, net->schdSolCpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsCpu.get(), cuStrmMain);
      std::cout << "Online action applied" << std::endl;
    } else if (useCustomUePrg == 1) {
      std::cout << "GPU PF UE selection setup completed [custom]" << std::endl;
      customUePrgScheduler->run(net->cellGrpUeStatusCpu.get(),
                                net->schdSolCpu.get(),
                                net->cellGrpPrmsCpu.get(),
                                net->schdSolGpu.get(),
                                cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout << "GPU PF UE selection run completed [custom]" << std::endl;
      std::cout << "CPU PF UE selection completed [custom]" << std::endl;
    } else if (type0AllUeScheduling) {
      stageTrace(t, "before_type0_all_ue_selection");
      populateType0AllUeSelection(net->schdSolCpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsCpu.get(), cuStrmMain);
      stageTrace(t, "after_type0_all_ue_selection");
      std::cout<<"Type-0 scheduling: using all active UEs for RBG allocation"<<std::endl;
    } else {
      if (baseline == 1) {
        stageTrace(t, "before_gpu_rr_ue_sel_setup");
        rrUeSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_rr_ue_sel_setup");
        std::cout<<"GPU RR UE selection setup completed"<<std::endl;

        stageTrace(t, "before_gpu_rr_ue_sel_run");
        rrUeSelGpu->run(cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_rr_ue_sel_run");
        std::cout<<"GPU RR UE selection run completed"<<std::endl;
      } else {
        stageTrace(t, "before_gpu_ue_sel_setup");
        mcUeSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_ue_sel_setup");
        std::cout<<"GPU PF UE selection setup completed"<<std::endl;

        stageTrace(t, "before_gpu_ue_sel_run");
        mcUeSelGpu->run(cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_ue_sel_run");
        std::cout<<"GPU PF UE selection run completed"<<std::endl;
      }

      if (runCpuReferencePath) {
        if (baseline == 1) {
          rrUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
          rrUeSelCpu->run();
          std::cout<<"CPU RR UE selection completed"<<std::endl;
        } else {
          stageTrace(t, "before_cpu_ue_sel_setup");
          mcUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
          stageTrace(t, "after_cpu_ue_sel_setup");
          stageTrace(t, "before_cpu_ue_sel_run");
          mcUeSelCpu->run();
          stageTrace(t, "after_cpu_ue_sel_run");
          std::cout<<"CPU PF UE selection completed"<<std::endl;
        }
      } else {
        copyGpuUeSelectionToCpu(net->schdSolCpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsCpu.get(), cuStrmMain);
      }
    }

    net->ueDownSelectGpu();
    std::cout<<"GPU UE downselection completed"<<std::endl;
    net->ueDownSelectCpu();
    std::cout<<"CPU UE downselection completed"<<std::endl;

    // only set coordinate cell IDs and perform cell assocation at the first time slot //
    if (t == 0) {
      net->execStatus.get()->cellIdRenew    = false;
      net->execStatus.get()->cellAssocRenew = false;
    }

    // setup GPU multi-cell scheduler
    if (onlineBridgeEnabled) {
      std::cout << "GPU scheduler setup started [online-action]" << std::endl;
      std::cout << "GPU scheduler setup completed [online-action]" << std::endl;
    } else if (useCustomUePrg == 1) {
      std::cout << "GPU scheduler setup started [custom]" << std::endl;
      std::cout << "GPU scheduler setup completed [custom]" << std::endl;
    } else {
      if (lightWeight == 1) {
        printf("Multi-cell scheduler: calling light-weight kernel\n");
      }
      std::cout<<"GPU scheduler setup started"<<std::endl;
      if (baseline == 1) {
        stageTrace(t, "before_gpu_rr_sch_setup");
        rrSchGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_rr_sch_setup");
        std::cout<<"GPU RR scheduler setup completed"<<std::endl;
      } else {
        stageTrace(t, "before_gpu_sch_setup");
        mcSchGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), net->simParam.get(), columnMajor, halfPrecision, lightWeight, percSmNumThrdBlk, cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_sch_setup");
        std::cout<<"GPU PF scheduler setup completed"<<std::endl;
      }

      if (runCpuReferencePath) {
        if (baseline == 1) {
          rrSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
        } else {
          stageTrace(t, "before_cpu_sch_setup");
          mcSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get(), net->simParam.get(), columnMajor);
          stageTrace(t, "after_cpu_sch_setup");
        }
      }
    }

#ifdef PDSCH_
    // setup GPU layer selection
    mcLayerSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), 0, cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // setup GPU MCS selection
    mcsSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    if (runCpuReferencePath) {
      mcLayerSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
      mcsSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    }
#endif

    // run GPU multi-cell scheduler
    if (onlineBridgeEnabled) {
      std::cout << "GPU scheduler run started [online-action]" << std::endl;
      std::cout << "GPU scheduler run completed [online-action]" << std::endl;
      std::cout << "CPU scheduler run started [online-action]" << std::endl;
      std::cout << "CPU scheduler run completed [online-action]" << std::endl;
    } else if (useCustomUePrg == 1) {
      std::cout << "GPU scheduler run started [custom]" << std::endl;
      std::cout << "GPU scheduler run completed [custom]" << std::endl;
      std::cout << "CPU scheduler run started [custom]" << std::endl;
      std::cout << "CPU scheduler run completed [custom]" << std::endl;
    } else {
      std::cout<<"GPU scheduler run started"<<std::endl;
      if (baseline == 1) {
        stageTrace(t, "before_gpu_rr_sch_run");
        rrSchGpu->run(cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_rr_sch_run");
        std::cout<<"GPU RR scheduler run completed"<<std::endl;
      } else {
        stageTrace(t, "before_gpu_sch_run");
        mcSchGpu->run(cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        stageTrace(t, "after_gpu_sch_run");
        std::cout<<"GPU PF scheduler run completed"<<std::endl;
      }

      if (runCpuReferencePath) {
        if (baseline == 1) {
          std::cout<<"CPU scheduler run started"<<std::endl;
          rrSchCpu->run();
          std::cout<<"CPU scheduler run completed"<<std::endl;
        } else {
          std::cout<<"CPU scheduler run started"<<std::endl;
          stageTrace(t, "before_cpu_sch_run");
          mcSchCpu->run();
          stageTrace(t, "after_cpu_sch_run");
          std::cout<<"CPU scheduler run completed"<<std::endl;
        }
      }
    }
    std::cout<<"PRB scheduling solution computed"<<std::endl;

#ifdef PDSCH_
    // run GPU layer selection
    stageTrace(t, "before_gpu_layer_sel_run");
    mcLayerSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    stageTrace(t, "after_gpu_layer_sel_run");
    std::cout<<"GPU Layer selection solution computed"<<std::endl;

    // run GPU MCS selection
    stageTrace(t, "before_gpu_mcs_sel_run");
    mcsSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    stageTrace(t, "after_gpu_mcs_sel_run");
    std::cout<<"GPU MCS selection solution computed"<<std::endl;
    // mcsSelGpu->debugLog();

    if (runCpuReferencePath) {
      stageTrace(t, "before_cpu_layer_sel_run");
      mcLayerSelCpu->run();
      stageTrace(t, "after_cpu_layer_sel_run");
      std::cout<<"CPU Layer selection solution computed"<<std::endl;

      stageTrace(t, "before_cpu_mcs_sel_run");
      mcsSelCpu->run();
      stageTrace(t, "after_cpu_mcs_sel_run");
      std::cout<<"CPU MCS selection solution computed"<<std::endl;
    } else {
      copyGpuSchedulingSolutionToCpu(net->schdSolCpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsCpu.get(), cuStrmMain);
    }

    totalAllocatedPrgCount += countAllocatedPrgs(net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    totalPrgCapacity += static_cast<unsigned long long>(net->cellGrpPrmsCpu.get()->totNumCell) *
                        static_cast<unsigned long long>(net->cellGrpPrmsCpu.get()->nPrbGrp);

    // accumulate per-UE average MCS based on scheduled UE mapping
    for (int schdUidx = 0; schdUidx < nUeSched; ++schdUidx) {
      const uint16_t activeUeId = net->schdSolCpu.get()->setSchdUePerCellTTI[schdUidx];
      const int16_t mcs = net->schdSolCpu.get()->mcsSelSol[schdUidx];
      if (activeUeId != 0xFFFF && activeUeId < static_cast<uint16_t>(nActiveUe) && mcs >= 0) {
        ueMcsSum[activeUeId] += static_cast<double>(mcs);
        ueMcsCnt[activeUeId] += 1;
      }
    }
#endif

    // use scheduling solution
    stageTrace(t, "before_net_run");
    net->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    stageTrace(t, "after_net_run");
    std::cout<<"Scheduling solution transferred to host"<<std::endl;

    bool solCheckPass = true;
    if (runCpuReferencePath && baseline == 0 && compareTtiInterval > 0) {
      if (((t + 1) % compareTtiInterval) == 0 || (!onlineBridgeEnabled && t == (numSimChnRlz - 1))) {
        solCheckPass = net->compareCpuGpuAllocSol();
      }
    }
    (void)solCheckPass;

#ifdef exitCheckFail_
    if (!solCheckPass) {
      saveToH5("tvDebug.h5",
               net->cellGrpUeStatusGpu.get(),
               net->cellGrpPrmsGpu.get(),
               net->schdSolGpu.get());
      return 0;
    }
#endif 

#ifdef PDSCH_
    stageTrace(t, "before_phy_abstract");
    net->phyAbstract(2, slotIdx);
    stageTrace(t, "after_phy_abstract");
#else
    stageTrace(t, "before_update_data_rate_ue_sel_cpu");
    net->updateDataRateUeSelCpu(slotIdx);
    stageTrace(t, "after_update_data_rate_ue_sel_cpu");
    stageTrace(t, "before_update_data_rate_gpu");
    net->updateDataRateGpu(slotIdx);
    stageTrace(t, "after_update_data_rate_gpu");
#endif

    stageTrace(t, "before_update_data_rate_all_cpu");
    net->updateDataRateAllActiveUeCpu(slotIdx);
    stageTrace(t, "after_update_data_rate_all_cpu");
    stageTrace(t, "before_update_data_rate_all_gpu");
    net->updateDataRateAllActiveUeGpu(slotIdx);
    stageTrace(t, "after_update_data_rate_all_gpu");
    if (net->cellGrpUeStatusCpu.get()->bufferSize != nullptr) {
      for (int ueId = 0; ueId < nActiveUe; ++ueId) {
        const unsigned long long after_sched =
            static_cast<unsigned long long>(net->cellGrpUeStatusCpu.get()->bufferSize[ueId]);
        servedBytesThisTti[ueId] =
            macBufferBeforeSched[ueId] > after_sched ? (macBufferBeforeSched[ueId] - after_sched) : 0ULL;
      }
      trafSvc->RecordMacServedBytes(servedBytesThisTti, t);
    }

    const bool onlineDone = onlineBridgeEnabled && !onlinePersistentMode && (t + 1 >= onlineEpisodeHorizon);
    if (replayWriter) {
      replayWriter->appendTransition(
          slotIdx,
          onlineDone || (!onlineBridgeEnabled && t == (numSimChnRlz - 1)),
          net->cellGrpUeStatusCpu.get(),
          net->cellGrpPrmsCpu.get(),
          net->schdSolCpu.get());
    }

    for (int schdUidx = 0; schdUidx < nUeSched; ++schdUidx) {
      const uint16_t activeUeId = net->schdSolCpu.get()->setSchdUePerCellTTI[schdUidx];
      if (activeUeId == 0xFFFF || activeUeId >= static_cast<uint16_t>(nActiveUe)) {
        continue;
      }
      const int16_t mcs = net->schdSolCpu.get()->mcsSelSol[schdUidx];
      if (mcs < 0) {
        continue;
      }
      ueTbTxPkts[activeUeId] += 1ULL;
      const int8_t tbErr = net->cellGrpUeStatusCpu.get()->tbErrLast != nullptr
                               ? net->cellGrpUeStatusCpu.get()->tbErrLast[schdUidx]
                               : -1;
      if (tbErr == 0) {
        ueTbSuccPkts[activeUeId] += 1ULL;
      }
      if (tbErr == 1) {
        ueTbErrCount[activeUeId] += 1ULL;
      }
      const float predBler = net->getPredictedBlerCpu(schdUidx, slotIdx);
      if (predBler >= 0.0f) {
        uePredBlerSum[activeUeId] += static_cast<double>(predBler);
        uePredBlerCnt[activeUeId] += 1U;
      }
      if (net->cellGrpPrmsCpu.get()->wbSinr != nullptr) {
        double sinrLin = 0.0;
        for (int antIdx = 0; antIdx < nUeAnt; ++antIdx) {
          const double antSinr = static_cast<double>(net->cellGrpPrmsCpu.get()->wbSinr[activeUeId * nUeAnt + antIdx]);
          sinrLin += std::max(antSinr, 1.0e-9);
        }
        sinrLin /= std::max(1, nUeAnt);
        ueWbSinrLinSumSched[activeUeId] += sinrLin;
        ueWbSinrCntSched[activeUeId] += 1U;
      }
    }

    executedTtiCount = t + 1;

    if (onlineBridgeEnabled) {
      std::vector<float> nextObsCell;
      std::vector<float> nextObsUe;
      std::vector<float> nextObsEdgeAttr;
      std::vector<uint8_t> nextMaskUe;
      std::vector<uint8_t> nextMaskCellUe;
      std::vector<uint8_t> nextMaskPrg;
      onlineCodec.buildObservation(
          net->cellGrpUeStatusCpu.get(), net->cellGrpPrmsCpu.get(), nextObsCell, nextObsUe, nextObsEdgeAttr);
      onlineCodec.buildActionMask(
          net->cellGrpUeStatusCpu.get(), net->cellGrpPrmsCpu.get(), nextMaskUe, nextMaskCellUe, nextMaskPrg);
      const cumac::online::OnlineFeatureCodec::RewardTerms reward = onlineCodec.buildReward(net->cellGrpUeStatusCpu.get());
      const cumac::online::StepState stepState = buildOnlineState(
          t + 1,
          onlineDone,
          onlineCodec,
          nextObsCell,
          nextObsUe,
          nextObsEdgeAttr,
          nextMaskUe,
          nextMaskCellUe,
          nextMaskPrg,
          reward);
      if (!onlineBridge->sendStepRsp(stepState)) {
        fprintf(stderr, "ERROR: Failed to send online step response.\n");
        return 1;
      }
      if (onlineDone) {
        break;
      }
    }
  }

  const double prgUtilizationRatio =
      totalPrgCapacity > 0ULL ? static_cast<double>(totalAllocatedPrgCount) / static_cast<double>(totalPrgCapacity) : 0.0;
  printf("PRG_UTILIZATION allocated_prg_count=%llu total_prg_capacity=%llu utilization_ratio=%.6f\n",
         totalAllocatedPrgCount,
         totalPrgCapacity,
         prgUtilizationRatio);

  if (replayWriter) {
    replayWriter->finalize();
  }

  if (onlineBridgeEnabled && onlineBridge) {
    if (onlineCloseRequested) {
      if (!onlineBridge->sendCloseRsp()) {
        fprintf(stderr, "ERROR: Failed to send online close response.\n");
        return 1;
      }
    } else {
      if (!onlineBridge->recvCloseReq()) {
        fprintf(stderr, "ERROR: Failed to receive online close request.\n");
        return 1;
      }
      if (!onlineBridge->sendCloseRsp()) {
        fprintf(stderr, "ERROR: Failed to send online close response.\n");
        return 1;
      }
    }
  }

  if (compactTtiLog && savedCoutBuf != nullptr) {
    std::cout.rdbuf(savedCoutBuf);
  }

  bool cpuGpuPerfCheckPass;
  if (onlineBridgeEnabled) {
    cpuGpuPerfCheckPass = true;
    printf("CPU and GPU scheduler performance check result: BYPASS (online bridge mode)\n");
  } else if (useCustomUePrg == 1) {
    cpuGpuPerfCheckPass = true;
    printf("CPU and GPU scheduler performance check result: BYPASS (custom UE+PRG mode)\n");
  } else if (!runCpuReferencePath) {
    cpuGpuPerfCheckPass = true;
    printf("CPU and GPU scheduler performance check result: BYPASS (gpu-only execution mode)\n");
  } else if (baseline == 1) { 
    cpuGpuPerfCheckPass = 1;
  } else {
    cpuGpuPerfCheckPass = net->compareCpuGpuSchdPerf();
    if (cpuGpuPerfCheckPass) {
      printf("CPU and GPU scheduler performance check result: PASS\n");
    } else {
      printf("CPU and GPU scheduler performance check result: FAIL\n");
    }
  }

  if (percent_ue_traffic > 0.0f && net->cellGrpUeStatusCpu.get()->bufferSize != nullptr) {
    unsigned long long generated_bytes = trafSvc->GetTotalGeneratedBytes();
    unsigned long long generated_pkts = trafSvc->GetTotalGeneratedPkts();
    unsigned long long accepted_bytes = trafSvc->GetTotalAcceptedBytes();
    unsigned long long dropped_bytes = trafSvc->GetTotalDroppedBytes();
    unsigned long long flow_queued_bytes = trafSvc->GetTotalFlowQueuedBytes();
    PacketDelaySummary trafficPacketDelay;
    std::vector<PacketDelaySummary> perFlowPacketDelay;
    trafSvc->GetPacketDelayStats(trafficPacketDelay, perFlowPacketDelay);

    unsigned long long mac_buffer_bytes = 0;
    for (int uIdx = 0; uIdx < totNumActiveUesConst; uIdx++) {
      mac_buffer_bytes += net->cellGrpUeStatusCpu.get()->bufferSize[uIdx];
    }

    unsigned long long served_bytes_est = 0;
    if (accepted_bytes > (flow_queued_bytes + mac_buffer_bytes)) {
      served_bytes_est = accepted_bytes - flow_queued_bytes - mac_buffer_bytes;
    }

    const int kpiTtiCount = std::max(1, executedTtiCount);
    double total_time_s = static_cast<double>(kpiTtiCount) * slotDurationConst;
    double offered_mbps = total_time_s > 0.0 ? (generated_bytes * 8.0) / total_time_s / 1e6 : 0.0;
    double served_mbps_est = total_time_s > 0.0 ? (served_bytes_est * 8.0) / total_time_s / 1e6 : 0.0;
    double drop_rate = generated_bytes > 0 ? static_cast<double>(dropped_bytes) / static_cast<double>(generated_bytes) : 0.0;
    double queue_delay_est_ms = served_mbps_est > 0.0 ? (mac_buffer_bytes * 8.0) / (served_mbps_est * 1e6) * 1e3 : 0.0;

    printf("TRAFFIC_KPI flows=%d generated_pkts=%llu generated_bytes=%llu accepted_bytes=%llu dropped_bytes=%llu flow_queued_bytes=%llu mac_buffer_bytes=%llu served_bytes_est=%llu\n",
           traffic_num_flows, generated_pkts, generated_bytes, accepted_bytes, dropped_bytes, flow_queued_bytes, mac_buffer_bytes, served_bytes_est);
    printf("TRAFFIC_KPI offered_mbps=%.6f served_mbps_est=%.6f drop_rate=%.6f queue_delay_est_ms=%.6f\n",
           offered_mbps, served_mbps_est, drop_rate, queue_delay_est_ms);
    printf("TRAFFIC_PKT_DELAY served_pkt_count=%llu pending_pkt_count=%llu mean_ms=%.6f p50_ms=%.6f p90_ms=%.6f p95_ms=%.6f max_ms=%.6f\n",
           trafficPacketDelay.served_packets,
           trafficPacketDelay.pending_packets,
           trafficPacketDelay.mean_delay_ms,
           trafficPacketDelay.p50_delay_ms,
           trafficPacketDelay.p90_delay_ms,
           trafficPacketDelay.p95_delay_ms,
           trafficPacketDelay.max_delay_ms);
  } else {
    printf("TRAFFIC_KPI flows=0 generated_pkts=0 generated_bytes=0 accepted_bytes=0 dropped_bytes=0 flow_queued_bytes=0 mac_buffer_bytes=0 served_bytes_est=0\n");
    printf("TRAFFIC_KPI offered_mbps=0.000000 served_mbps_est=0.000000 drop_rate=0.000000 queue_delay_est_ms=0.000000\n");
    printf("TRAFFIC_PKT_DELAY served_pkt_count=0 pending_pkt_count=0 mean_ms=0.000000 p50_ms=0.000000 p90_ms=0.000000 p95_ms=0.000000 max_ms=0.000000\n");
  }

  std::vector<unsigned long long> ueGeneratedBytes;
  std::vector<unsigned long long> ueAcceptedBytes;
  std::vector<unsigned long long> ueDroppedBytes;
  std::vector<unsigned long long> ueFlowQueuedBytes;
  PacketDelaySummary trafficPacketDelaySummary;
  std::vector<PacketDelaySummary> uePacketDelay;
  trafSvc->GetPerFlowStats(ueGeneratedBytes, ueAcceptedBytes, ueDroppedBytes, ueFlowQueuedBytes);
  trafSvc->GetPacketDelayStats(trafficPacketDelaySummary, uePacketDelay);

  std::vector<int> ueCellId(nActiveUe, -1);
  std::vector<int> ueLocalId(nActiveUe, -1);
  if (net->cellGrpPrmsCpu.get()->cellAssocActUe != nullptr) {
    const int nCell = static_cast<int>(net->cellGrpPrmsCpu.get()->nCell);
    for (int ueId = 0; ueId < nActiveUe; ++ueId) {
      for (int cIdx = 0; cIdx < nCell; ++cIdx) {
        if (net->cellGrpPrmsCpu.get()->cellAssocActUe[cIdx * nActiveUe + ueId] != 0) {
          ueCellId[ueId] = cIdx;
          break;
        }
      }
    }
    std::vector<int> perCellLocalCnt(nCell, 0);
    for (int ueId = 0; ueId < nActiveUe; ++ueId) {
      const int cIdx = ueCellId[ueId];
      if (cIdx >= 0 && cIdx < nCell) {
        ueLocalId[ueId] = perCellLocalCnt[cIdx]++;
      }
    }
  } else if (net->cellGrpPrmsCpu.get()->nCell > 0 && (nActiveUe % net->cellGrpPrmsCpu.get()->nCell) == 0) {
    const int uePerCell = nActiveUe / net->cellGrpPrmsCpu.get()->nCell;
    for (int ueId = 0; ueId < nActiveUe; ++ueId) {
      ueCellId[ueId] = ueId / uePerCell;
      ueLocalId[ueId] = ueId % uePerCell;
    }
  }

  printf("UE_KPI_HEADER cell_id ue_local_id ue_id avg_thr_mbps avg_mcs_tx_only avg_mcs_all_tti0 scheduled_tti_count no_tx_tti_count scheduled_ratio avg_wb_sinr_db avg_sched_wb_sinr_db avg_predicted_bler tb_err_count tb_bler flow_drop_rate tx_success_rate tx_drop_rate tx_total_pkts tx_success_pkts queue_delay_est_ms generated_bytes accepted_bytes dropped_bytes flow_queued_bytes mac_buffer_bytes mcs_samples\n");
  printf("UE_PKT_DELAY_HEADER ue_id served_pkt_count pending_pkt_count packet_delay_mean_ms packet_delay_p50_ms packet_delay_p90_ms packet_delay_p95_ms packet_delay_max_ms\n");
  for (int ueId = 0; ueId < nActiveUe; ++ueId) {
    const double avgMcsTxOnly = ueMcsCnt[ueId] > 0 ? (ueMcsSum[ueId] / static_cast<double>(ueMcsCnt[ueId])) : -1.0;
    const int kpiTtiCount = std::max(1, executedTtiCount);
    const double totalTimeS = static_cast<double>(kpiTtiCount) * slotDurationConst;
    const double avgMcsAllTti0 = ueMcsSum[ueId] / static_cast<double>(kpiTtiCount);
    const unsigned long long generated = (ueId < static_cast<int>(ueGeneratedBytes.size())) ? ueGeneratedBytes[ueId] : 0ULL;
    const unsigned long long accepted = (ueId < static_cast<int>(ueAcceptedBytes.size())) ? ueAcceptedBytes[ueId] : 0ULL;
    const unsigned long long dropped = (ueId < static_cast<int>(ueDroppedBytes.size())) ? ueDroppedBytes[ueId] : 0ULL;
    const unsigned long long flowQueued = (ueId < static_cast<int>(ueFlowQueuedBytes.size())) ? ueFlowQueuedBytes[ueId] : 0ULL;
    const unsigned long long macBuffer = net->cellGrpUeStatusCpu.get()->bufferSize != nullptr
                                             ? static_cast<unsigned long long>(net->cellGrpUeStatusCpu.get()->bufferSize[ueId])
                                             : 0ULL;
    const unsigned long long txTotalPkts = ueTbTxPkts[ueId];
    const unsigned long long txSuccessPkts = ueTbSuccPkts[ueId];
    const unsigned long long tbErrCount = ueTbErrCount[ueId];
    const unsigned long long servedBytesEst =
        accepted > (flowQueued + macBuffer) ? (accepted - flowQueued - macBuffer) : 0ULL;
    const double avgThrBps = totalTimeS > 0.0 ? (static_cast<double>(servedBytesEst) * 8.0) / totalTimeS : 0.0;
    const double avgThrMbps = avgThrBps / 1.0e6;
    const double txSuccessRate = txTotalPkts > 0 ? static_cast<double>(txSuccessPkts) / static_cast<double>(txTotalPkts) : 0.0;
    const double txDropRate = txTotalPkts > 0 ? 1.0 - txSuccessRate : 0.0;
    const double avgPredictedBler = uePredBlerCnt[ueId] > 0
                                        ? uePredBlerSum[ueId] / static_cast<double>(uePredBlerCnt[ueId])
                                        : 0.0;
    const double tbBler = txTotalPkts > 0 ? static_cast<double>(tbErrCount) / static_cast<double>(txTotalPkts) : 0.0;
    const double flowDropRate = generated > 0 ? static_cast<double>(dropped) / static_cast<double>(generated) : 0.0;
    const uint32_t scheduledTtiCount = ueMcsCnt[ueId];
    const uint32_t noTxTtiCount = kpiTtiCount >= static_cast<int>(scheduledTtiCount)
                                      ? static_cast<uint32_t>(kpiTtiCount - static_cast<int>(scheduledTtiCount))
                                      : 0U;
    const double scheduledRatio = static_cast<double>(scheduledTtiCount) / static_cast<double>(kpiTtiCount);
    const double avgWbSinrDb = ueWbSinrCntAll[ueId] > 0
                                   ? 10.0 * std::log10(ueWbSinrLinSumAll[ueId] / static_cast<double>(ueWbSinrCntAll[ueId]))
                                   : 0.0;
    const double avgSchedWbSinrDb = ueWbSinrCntSched[ueId] > 0
                                        ? 10.0 * std::log10(ueWbSinrLinSumSched[ueId] / static_cast<double>(ueWbSinrCntSched[ueId]))
                                        : 0.0;
    const double ueDelayMs = avgThrBps > 0.0 ? (static_cast<double>(macBuffer) * 8.0 / avgThrBps * 1.0e3) : 0.0;

    printf("UE_KPI cell_id=%d ue_local_id=%d ue_id=%d avg_thr_mbps=%.6f avg_mcs_tx_only=%.6f avg_mcs_all_tti0=%.6f scheduled_tti_count=%u no_tx_tti_count=%u scheduled_ratio=%.6f avg_wb_sinr_db=%.6f avg_sched_wb_sinr_db=%.6f avg_predicted_bler=%.6f tb_err_count=%llu tb_bler=%.6f flow_drop_rate=%.6f tx_success_rate=%.6f tx_drop_rate=%.6f tx_total_pkts=%llu tx_success_pkts=%llu queue_delay_est_ms=%.6f generated_bytes=%llu accepted_bytes=%llu dropped_bytes=%llu flow_queued_bytes=%llu mac_buffer_bytes=%llu mcs_samples=%u\n",
           ueCellId[ueId],
           ueLocalId[ueId],
           ueId,
           avgThrMbps,
           avgMcsTxOnly,
           avgMcsAllTti0,
           scheduledTtiCount,
           noTxTtiCount,
           scheduledRatio,
           avgWbSinrDb,
           avgSchedWbSinrDb,
           avgPredictedBler,
           tbErrCount,
           tbBler,
           flowDropRate,
           txSuccessRate,
           txDropRate,
           txTotalPkts,
           txSuccessPkts,
           ueDelayMs,
           generated,
           accepted,
           dropped,
           flowQueued,
           macBuffer,
           ueMcsCnt[ueId]);
    const PacketDelaySummary pktSummary =
        ueId < static_cast<int>(uePacketDelay.size()) ? uePacketDelay[ueId] : PacketDelaySummary{};
    printf("UE_PKT_DELAY ue_id=%d served_pkt_count=%llu pending_pkt_count=%llu packet_delay_mean_ms=%.6f packet_delay_p50_ms=%.6f packet_delay_p90_ms=%.6f packet_delay_p95_ms=%.6f packet_delay_max_ms=%.6f\n",
           ueId,
           pktSummary.served_packets,
           pktSummary.pending_packets,
           pktSummary.mean_delay_ms,
           pktSummary.p50_delay_ms,
           pktSummary.p90_delay_ms,
           pktSummary.p95_delay_ms,
           pktSummary.max_delay_ms);
  }

  if (saveTv == 1) {
    std::string saveTvName = "TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsGpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5(saveTvName,
             net->cellGrpUeStatusGpu.get(),
             net->cellGrpPrmsGpu.get(),
             net->schdSolGpu.get());
  } else if (saveTv == 2) {
    std::string saveTvName = "CPU_TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsCpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5_CPU(saveTvName,
             net->cellGrpUeStatusCpu.get(),
             net->cellGrpPrmsCpu.get(),
             net->schdSolCpu.get());
  } else if (saveTv == 3) {
    std::string saveTvName = "TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsGpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5(saveTvName,
             net->cellGrpUeStatusGpu.get(),
             net->cellGrpPrmsGpu.get(),
             net->schdSolGpu.get());

    saveTvName = "TV_F08_";
    for (int cellIdx = 0; cellIdx < net->cellGrpPrmsGpu.get()->nCell; cellIdx++) {
      saveToH5_testMAC_perCell(saveTvName,
                               cellIdx,
                               net->cellGrpUeStatusGpu.get(),
                               net->cellGrpPrmsGpu.get(),
                               net->schdSolGpu.get());
    }
  }

  net->writeToFileLargeNumActUe();
  net->writetoFileLargeNumActUe_short();

  // clean up memory
  if (prdSchemeConst) {
    svdPrd->destroy();
  }
  net->destroyAPI();

  if (mcSinrCalGpu) delete mcSinrCalGpu;
  if (mcUeSelGpu) delete mcUeSelGpu;
  if (mcSchGpu) delete mcSchGpu;
  if (rrSchCpu) delete rrSchCpu;
  if (mcSchCpu) delete mcSchCpu;
  if (rrUeSelCpu) delete rrUeSelCpu;
  if (mcUeSelCpu) delete mcUeSelCpu;
  if (svdPrd) delete svdPrd;
  if (net) delete net;
  if (mcLayerSelGpu) delete mcLayerSelGpu;
  if (mcLayerSelCpu) delete mcLayerSelCpu;

  return !cpuGpuPerfCheckPass;
}
