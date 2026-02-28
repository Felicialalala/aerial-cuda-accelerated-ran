#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARAM_FILE="${ROOT_DIR}/cuMAC/examples/parameters.h"
UPDATE_TOOL="${ROOT_DIR}/cuMAC/scripts/update_parameter.py"
BUILD_AERIAL_SCRIPT="${ROOT_DIR}/testBenches/phase4_test_scripts/build_aerial_sdk.sh"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4" # phase4 | cmake | skip
GPU_ID=0
TTI_COUNT=200
DL_UL="dl"          # dl or ul
FADING_MODE=0       # 0: Rayleigh, 1: TDL(PRG), 3: CDL(PRG)
TRAFFIC_PERCENT=100
TRAFFIC_RATE=5000   # bytes per packet, arrival rate fixed at 1 pkt/TTI in example
RUN_TAG=""
ANALYZER_SCRIPT="${ROOT_DIR}/cuMAC/scripts/summarize_stageA_kpi.py"

usage() {
    cat <<EOF
Stage-A MVP baseline script (original scheduler only).

Usage:
  $(basename "$0") [options]

Options:
  --build-dir <path>   Build directory (default: ${BUILD_DIR})
  --build-method <m>   Build method: phase4 | cmake | skip (default: ${BUILD_METHOD})
  --gpu <id>           GPU device id (default: ${GPU_ID})
  --tti <count>        Number of simulated TTIs (default: ${TTI_COUNT})
  --mode <dl|ul>       Downlink or uplink (default: ${DL_UL})
  --fading <0|1|3>     0=Rayleigh, 1=TDL on PRG, 3=CDL on PRG (default: ${FADING_MODE})
  --traffic-percent <p> UE traffic percentage [0,100] (default: ${TRAFFIC_PERCENT})
  --traffic-rate <b>   Traffic packet size in bytes (default: ${TRAFFIC_RATE})
  --tag <name>         Optional run tag in output folder
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-method)
            BUILD_METHOD="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --tti)
            TTI_COUNT="$2"
            shift 2
            ;;
        --mode)
            DL_UL="$2"
            shift 2
            ;;
        --fading)
            FADING_MODE="$2"
            shift 2
            ;;
        --traffic-percent)
            TRAFFIC_PERCENT="$2"
            shift 2
            ;;
        --traffic-rate)
            TRAFFIC_RATE="$2"
            shift 2
            ;;
        --tag)
            RUN_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "${PARAM_FILE}" ]]; then
    echo "Cannot find parameters file: ${PARAM_FILE}" >&2
    exit 1
fi

if [[ ! -f "${UPDATE_TOOL}" ]]; then
    echo "Cannot find update tool: ${UPDATE_TOOL}" >&2
    exit 1
fi

if [[ "${DL_UL}" != "dl" && "${DL_UL}" != "ul" ]]; then
    echo "--mode must be 'dl' or 'ul'" >&2
    exit 1
fi

if [[ "${FADING_MODE}" != "0" && "${FADING_MODE}" != "1" && "${FADING_MODE}" != "3" ]]; then
    echo "--fading must be one of 0, 1, 3" >&2
    exit 1
fi

if ! [[ "${TRAFFIC_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--traffic-percent must be a number in [0,100]" >&2
    exit 1
fi

if ! [[ "${TRAFFIC_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--traffic-rate must be a positive number" >&2
    exit 1
fi

awk "BEGIN {exit !(${TRAFFIC_PERCENT} >= 0 && ${TRAFFIC_PERCENT} <= 100)}" || {
    echo "--traffic-percent must be within [0,100]" >&2
    exit 1
}

if [[ "${BUILD_METHOD}" != "phase4" && "${BUILD_METHOD}" != "cmake" && "${BUILD_METHOD}" != "skip" ]]; then
    echo "--build-method must be one of: phase4, cmake, skip" >&2
    exit 1
fi

PARAM_BAK="$(mktemp)"
cp "${PARAM_FILE}" "${PARAM_BAK}"
restore_params() {
    cp "${PARAM_BAK}" "${PARAM_FILE}"
    rm -f "${PARAM_BAK}"
}
trap restore_params EXIT

set_param() {
    local param="$1"
    local value="$2"
    python3 "${UPDATE_TOOL}" --file "${PARAM_FILE}" --param "${param}" --value "${value}"
}

echo "[Stage-A] Apply MVP baseline parameters (3 cells, 8 UE/cell, 4T4R, 30kHz, Rayleigh-ready)..."
set_param gpuDeviceIdx "${GPU_ID}"
set_param numSimChnRlz "${TTI_COUNT}"
set_param slotDurationConst "0.5e-3"
set_param scsConst "30000.0"
set_param cellRadiusConst "500"
set_param numCellConst "3"
set_param numCoorCellConst "numCellConst"
set_param numUePerCellConst "8"
set_param numUeForGrpConst "8"
set_param numActiveUePerCellConst "8"
set_param totNumUesConst "numCellConst*numUePerCellConst"
set_param totNumActiveUesConst "numCellConst*numActiveUePerCellConst"
set_param nBsAntConst "4"
set_param nUeAntConst "4"
set_param nPrbsPerGrpConst "4"
set_param nPrbGrpsConst "68"

case "${BUILD_METHOD}" in
    phase4)
        if [[ ! -f "/.dockerenv" ]]; then
            cat >&2 <<EOF
This script is configured to build with phase4 tooling and must run inside the dev container.

Recommended steps:
  ./cuPHY-CP/container/run_aerial.sh
  ./testBenches/phase4_test_scripts/build_aerial_sdk.sh
  ./cuMAC/scripts/run_stageA_mvp_baseline.sh --build-method skip
EOF
            exit 1
        fi
        if [[ ! -x "${BUILD_AERIAL_SCRIPT}" ]]; then
            echo "Build script not found: ${BUILD_AERIAL_SCRIPT}" >&2
            exit 1
        fi
        echo "[Stage-A] Building SDK via phase4 script..."
        "${BUILD_AERIAL_SCRIPT}" --build_dir "${BUILD_DIR}"
        ;;
    cmake)
        if ! command -v cmake >/dev/null 2>&1; then
            echo "cmake not found. Use --build-method phase4 inside container, or install cmake." >&2
            exit 1
        fi
        if [[ ! -d "${BUILD_DIR}" ]]; then
            echo "Build directory does not exist: ${BUILD_DIR}" >&2
            echo "Configure first, e.g.:" >&2
            echo "  cmake -B ${BUILD_DIR} -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native -DCMAKE_BUILD_TYPE=Release" >&2
            exit 1
        fi
        echo "[Stage-A] Build target with cmake: multiCellSchedulerUeSelection"
        cmake --build "${BUILD_DIR}" --target multiCellSchedulerUeSelection
        ;;
    skip)
        echo "[Stage-A] Skip build as requested (--build-method skip)."
        ;;
esac

BIN="${BUILD_DIR}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
if [[ ! -x "${BIN}" ]]; then
    echo "Executable not found: ${BIN}" >&2
    exit 1
fi

DL_IND=1
if [[ "${DL_UL}" == "ul" ]]; then
    DL_IND=0
fi

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_TAG}" ]]; then
    OUT_DIR="${ROOT_DIR}/output/stageA_mvp_baseline_${RUN_TAG}_${TS}"
else
    OUT_DIR="${ROOT_DIR}/output/stageA_mvp_baseline_${TS}"
fi
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run.log"

echo "[Stage-A] Running baseline scheduler..."
echo "  output_dir=${OUT_DIR}"
echo "  cmd=${BIN} -d ${DL_IND} -b 0 -f ${FADING_MODE} -g ${TRAFFIC_PERCENT} -r ${TRAFFIC_RATE}"

(
    cd "${OUT_DIR}"
    "${BIN}" -d "${DL_IND}" -b 0 -f "${FADING_MODE}" -g "${TRAFFIC_PERCENT}" -r "${TRAFFIC_RATE}"
) 2>&1 | tee "${LOG_FILE}"

if [[ -f "${ANALYZER_SCRIPT}" ]]; then
    echo "[Stage-A] Summarizing KPIs..."
    python3 "${ANALYZER_SCRIPT}" \
        --output-dir "${OUT_DIR}" \
        --slot-duration-ms 0.5 \
        --traffic-percent "${TRAFFIC_PERCENT}" \
        --traffic-rate "${TRAFFIC_RATE}"
fi

echo "[Stage-A] Done. Log: ${LOG_FILE}"
