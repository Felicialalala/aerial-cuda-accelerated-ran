#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARAM_FILE="${ROOT_DIR}/cuMAC/examples/parameters.h"
UPDATE_TOOL="${ROOT_DIR}/cuMAC/scripts/update_parameter.py"
BUILD_AERIAL_SCRIPT="${ROOT_DIR}/testBenches/phase4_test_scripts/build_aerial_sdk.sh"
KPI_SCRIPT="${ROOT_DIR}/cuMAC/scripts/summarize_stageA_kpi.py"
MATRIX_SCRIPT="${ROOT_DIR}/cuMAC/scripts/summarize_stageB_matrix.py"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4"   # phase4 | cmake | skip
RESTORE_PARAMS=0        # 0=keep applied compile-time params, 1=restore on exit

GPU_ID=0
TTI_COUNT=2000
DL_UL="dl"
FADING_MODE=0           # 0=Rayleigh, 1=TDL on PRBG, 2=TDL on SC+PRBG, 3=CDL on PRBG, 4=CDL on SC+PRBG
UE_PER_CELL=8
TOPOLOGY_SEED=0
UE_PLACEMENT="uniform"      # uniform | stratified
UE_RADIUS_SPLITS="0.33,0.66"
UE_STRATA_COUNTS=""
UE_VORONOI_CLIP=0
TRAFFIC_PERCENT=100
PACKET_SIZE_BYTES=5000
TRAFFIC_ARRIVAL_RATE=0.2
CDL_PROFILES="C,D"
CDL_DELAY_SPREADS_NS="300,1000"
ALLOW_PROFILE_D=0
RUN_TIMEOUT_SEC=0
KILL_AFTER_SEC=30
RUN_TAG=""
COMPACT_OUTPUT=1        # 1=regular profile, 0=keep full artifacts
CUSTOM_UE_PRG=0         # 1=use CustomUePrgScheduler for UE+PRG, 0=native
BASELINE_SCHEDULER="pf" # pf | rr (effective for native path; maps to binary -b)
CUSTOM_POLICY="gnnrl"   # gnnrl | legacy (effective only when --custom-ue-prg=1)
MODEL_PATH=""          # used when --custom-policy gnnrl_model
POLICY_TIMEOUT_MS=0    # policy timeout hint for model runtime (0=disable)
COMPACT_TTI_LOG=1       # 1=compact per-TTI stage logs
PROGRESS_TTI_INTERVAL=100
KPI_TTI_LOG_INTERVAL=100
COMPARE_TTI_INTERVAL=0  # per-TTI CPU/GPU solution compare interval, 0=disable
REPLAY_DUMP=0           # 1=export per-TTI RL replay transitions
REPLAY_DIR=""           # empty means <scenario_out_dir>/replay
ONLINE_BRIDGE=0         # 1=enable online bridge mode in binary
ONLINE_SOCKET="/tmp/cumac_stageb_online.sock"
EXEC_MODE="both"       # both | gpu

usage() {
    cat <<EOF
Stage-B main experiment script (7-site coordinated cluster only, no outer interferer ring, 4T4R, Type-0 bitmap allocation).

Usage:
  $(basename "$0") [options]

Options:
  --build-dir <path>          Build directory (default: ${BUILD_DIR})
  --build-method <m>          phase4 | cmake | skip (default: ${BUILD_METHOD})
  --gpu <id>                  GPU device id (default: ${GPU_ID})
  --tti <count>               Number of simulated TTIs (default: ${TTI_COUNT})
  --mode <dl|ul>              Downlink or uplink (default: ${DL_UL})
  --fading-mode <0|1|2|3|4>   0=Rayleigh, 1=TDL on PRBG, 2=TDL on SC+PRBG, 3=CDL on PRBG, 4=CDL on SC+PRBG (default: ${FADING_MODE})
  --ue-per-cell <n>           Active/scheduled UE per cell (default: ${UE_PER_CELL})
  --topology-seed <n>         Fixed topology/random seed (default: ${TOPOLOGY_SEED})
  --ue-placement <m>          UE placement mode: uniform | stratified (default: ${UE_PLACEMENT})
  --ue-radius-splits <a,b>    Stratified center/mid upper radius ratios (default: ${UE_RADIUS_SPLITS})
  --ue-strata-counts <a,b,c>  Per-cell UE counts for center/mid/edge; empty means balanced split
  --ue-voronoi-clip <0|1>     1 clips UE samples to serving-cell Voronoi region (default: ${UE_VORONOI_CLIP})
  --traffic-percent <p>       UE traffic percentage [0,100] (default: ${TRAFFIC_PERCENT})
  --packet-size-bytes <b>     Traffic packet size in bytes (default: ${PACKET_SIZE_BYTES})
  --traffic-rate <b>          Deprecated alias of --packet-size-bytes
  --traffic-arrival-rate <r>  Traffic arrival rate in pkt/TTI (default: ${TRAFFIC_ARRIVAL_RATE})
  --cdl-profiles <list>       Comma list, e.g. C,D (default: ${CDL_PROFILES})
  --cdl-delay-spreads <list>  Comma list (ns), aligned with profiles (default: ${CDL_DELAY_SPREADS_NS})
  --allow-profile-d <0|1>     1 to attempt CDL-D. 0 will skip D with note (default: ${ALLOW_PROFILE_D})
  --timeout-sec <sec>         Timeout per scenario, 0 means no timeout (default: ${RUN_TIMEOUT_SEC})
  --kill-after-sec <sec>      Extra grace period after timeout TERM (default: ${KILL_AFTER_SEC})
  --compact-output <0|1>      1 keeps regular artifacts only (default: ${COMPACT_OUTPUT})
  --custom-ue-prg <0|1>       1 use custom UE+PRG scheduler (default: ${CUSTOM_UE_PRG})
  --baseline-scheduler <m>    pf | rr for native baseline/reference path (default: ${BASELINE_SCHEDULER})
  --custom-policy <name>      gnnrl | legacy | gnnrl_model (default: ${CUSTOM_POLICY})
  --model-path <path>         ONNX model path for gnnrl_model policy (default: empty)
  --policy-timeout-ms <n>     Model policy timeout hint in ms, 0=disable (default: ${POLICY_TIMEOUT_MS})
  --compact-tti-log <0|1>     1 compact per-TTI logs (default: ${COMPACT_TTI_LOG})
  --progress-tti <n>          Progress print interval in TTI (default: ${PROGRESS_TTI_INTERVAL})
  --kpi-tti-log <n>           Throughput print interval in TTI (default: ${KPI_TTI_LOG_INTERVAL})
  --compare-tti <n>           CPU/GPU per-TTI compare interval, 0=disable (default: ${COMPARE_TTI_INTERVAL})
  --replay-dump <0|1>         Export RL replay transitions (default: ${REPLAY_DUMP})
  --replay-dir <path>         Base replay output dir (default: scenario local replay/)
  --online-bridge <0|1>       Enable online bridge mode (default: ${ONLINE_BRIDGE})
  --online-socket <path>      Unix socket path for online bridge (default: ${ONLINE_SOCKET})
  --exec-mode <mode>          both | gpu (default: ${EXEC_MODE})
  --tag <name>                Optional run tag
  --restore-params <0|1>      Restore parameters.h on exit (default: ${RESTORE_PARAMS})
  -h, --help                  Show this help

Notes:
  1) In pure 7-cell mode (no outer interferers), current code path is unstable for CDL (fading-mode 3/4) and may segfault.
     Use fading-mode 0/1/2 for stable baseline runs.
  2) Current chanModels implementation does not support LOS path for CDL-D/E and may exit.
     By default this script skips CDL-D unless --allow-profile-d=1.
  3) KPI summary is generated per scenario and also aggregated to stageB_kpi_matrix.{csv,txt}.
  4) With --compact-output=1, pass scenarios keep kpi_summary.json + run_key.log only.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --build-method) BUILD_METHOD="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --tti) TTI_COUNT="$2"; shift 2 ;;
        --mode) DL_UL="$2"; shift 2 ;;
        --fading-mode) FADING_MODE="$2"; shift 2 ;;
        --ue-per-cell) UE_PER_CELL="$2"; shift 2 ;;
        --topology-seed) TOPOLOGY_SEED="$2"; shift 2 ;;
        --ue-placement) UE_PLACEMENT="$2"; shift 2 ;;
        --ue-radius-splits) UE_RADIUS_SPLITS="$2"; shift 2 ;;
        --ue-strata-counts) UE_STRATA_COUNTS="$2"; shift 2 ;;
        --ue-voronoi-clip) UE_VORONOI_CLIP="$2"; shift 2 ;;
        --traffic-percent) TRAFFIC_PERCENT="$2"; shift 2 ;;
        --packet-size-bytes) PACKET_SIZE_BYTES="$2"; shift 2 ;;
        --traffic-rate) PACKET_SIZE_BYTES="$2"; shift 2 ;;
        --traffic-arrival-rate) TRAFFIC_ARRIVAL_RATE="$2"; shift 2 ;;
        --cdl-profiles) CDL_PROFILES="$2"; shift 2 ;;
        --cdl-delay-spreads) CDL_DELAY_SPREADS_NS="$2"; shift 2 ;;
        --allow-profile-d) ALLOW_PROFILE_D="$2"; shift 2 ;;
        --timeout-sec) RUN_TIMEOUT_SEC="$2"; shift 2 ;;
        --kill-after-sec) KILL_AFTER_SEC="$2"; shift 2 ;;
        --compact-output) COMPACT_OUTPUT="$2"; shift 2 ;;
        --custom-ue-prg) CUSTOM_UE_PRG="$2"; shift 2 ;;
        --baseline-scheduler) BASELINE_SCHEDULER="$2"; shift 2 ;;
        --custom-policy) CUSTOM_POLICY="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --policy-timeout-ms) POLICY_TIMEOUT_MS="$2"; shift 2 ;;
        --compact-tti-log) COMPACT_TTI_LOG="$2"; shift 2 ;;
        --progress-tti) PROGRESS_TTI_INTERVAL="$2"; shift 2 ;;
        --kpi-tti-log) KPI_TTI_LOG_INTERVAL="$2"; shift 2 ;;
        --compare-tti) COMPARE_TTI_INTERVAL="$2"; shift 2 ;;
        --replay-dump) REPLAY_DUMP="$2"; shift 2 ;;
        --replay-dir) REPLAY_DIR="$2"; shift 2 ;;
        --online-bridge) ONLINE_BRIDGE="$2"; shift 2 ;;
        --online-socket) ONLINE_SOCKET="$2"; shift 2 ;;
        --exec-mode) EXEC_MODE="$2"; shift 2 ;;
        --tag) RUN_TAG="$2"; shift 2 ;;
        --restore-params) RESTORE_PARAMS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -f "${PARAM_FILE}" || ! -f "${UPDATE_TOOL}" ]]; then
    echo "Missing required files under repo root." >&2
    exit 1
fi

if [[ "${DL_UL}" != "dl" && "${DL_UL}" != "ul" ]]; then
    echo "--mode must be dl or ul" >&2
    exit 1
fi
if [[ "${FADING_MODE}" != "0" && "${FADING_MODE}" != "1" && "${FADING_MODE}" != "2" && "${FADING_MODE}" != "3" && "${FADING_MODE}" != "4" ]]; then
    echo "--fading-mode must be one of: 0,1,2,3,4" >&2
    exit 1
fi
if [[ "${BUILD_METHOD}" != "phase4" && "${BUILD_METHOD}" != "cmake" && "${BUILD_METHOD}" != "skip" ]]; then
    echo "--build-method must be one of: phase4, cmake, skip" >&2
    exit 1
fi
if ! [[ "${UE_PER_CELL}" =~ ^[0-9]+$ ]] || [[ "${UE_PER_CELL}" -lt 1 ]]; then
    echo "--ue-per-cell must be a positive integer" >&2
    exit 1
fi
if ! [[ "${TOPOLOGY_SEED}" =~ ^[0-9]+$ ]]; then
    echo "--topology-seed must be a non-negative integer" >&2
    exit 1
fi
if [[ "${UE_PLACEMENT}" != "uniform" && "${UE_PLACEMENT}" != "stratified" ]]; then
    echo "--ue-placement must be uniform or stratified" >&2
    exit 1
fi
if ! [[ "${UE_RADIUS_SPLITS}" =~ ^[0-9]+([.][0-9]+)?,[0-9]+([.][0-9]+)?$ ]]; then
    echo "--ue-radius-splits must be in <center_max,mid_max> format" >&2
    exit 1
fi
IFS=',' read -r UE_CENTER_MAX UE_MID_MAX <<< "${UE_RADIUS_SPLITS}"
awk "BEGIN {exit !(${UE_CENTER_MAX} > 0 && ${UE_CENTER_MAX} < ${UE_MID_MAX} && ${UE_MID_MAX} < 1)}" || {
    echo "--ue-radius-splits must satisfy 0 < center_max < mid_max < 1" >&2
    exit 1
}
if [[ -n "${UE_STRATA_COUNTS}" ]]; then
    if ! [[ "${UE_STRATA_COUNTS}" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
        echo "--ue-strata-counts must be in <center_count,mid_count,edge_count> format" >&2
        exit 1
    fi
    IFS=',' read -r UE_CENTER_COUNT UE_MID_COUNT UE_EDGE_COUNT <<< "${UE_STRATA_COUNTS}"
    if [[ $((UE_CENTER_COUNT + UE_MID_COUNT + UE_EDGE_COUNT)) -ne ${UE_PER_CELL} ]]; then
        echo "--ue-strata-counts must sum to --ue-per-cell (${UE_PER_CELL})" >&2
        exit 1
    fi
fi
if ! [[ "${UE_VORONOI_CLIP}" =~ ^[01]$ ]]; then
    echo "--ue-voronoi-clip must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${ALLOW_PROFILE_D}" =~ ^[01]$ ]]; then
    echo "--allow-profile-d must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${RUN_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
    echo "--timeout-sec must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${KILL_AFTER_SEC}" =~ ^[0-9]+$ ]] || [[ "${KILL_AFTER_SEC}" -lt 1 ]]; then
    echo "--kill-after-sec must be a positive integer" >&2
    exit 1
fi
if ! [[ "${COMPACT_OUTPUT}" =~ ^[01]$ ]]; then
    echo "--compact-output must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${CUSTOM_UE_PRG}" =~ ^[01]$ ]]; then
    echo "--custom-ue-prg must be 0 or 1" >&2
    exit 1
fi
BASELINE_SCHEDULER="$(echo "${BASELINE_SCHEDULER}" | tr '[:upper:]' '[:lower:]')"
if [[ "${BASELINE_SCHEDULER}" != "pf" && "${BASELINE_SCHEDULER}" != "rr" ]]; then
    echo "--baseline-scheduler must be pf or rr" >&2
    exit 1
fi
CUSTOM_POLICY="$(echo "${CUSTOM_POLICY}" | tr '[:upper:]' '[:lower:]')"
if [[ "${CUSTOM_POLICY}" != "gnnrl" && "${CUSTOM_POLICY}" != "legacy" && "${CUSTOM_POLICY}" != "gnnrl_model" ]]; then
    echo "--custom-policy must be gnnrl, legacy, or gnnrl_model" >&2
    exit 1
fi
if ! [[ "${POLICY_TIMEOUT_MS}" =~ ^[0-9]+$ ]]; then
    echo "--policy-timeout-ms must be a non-negative integer" >&2
    exit 1
fi
if [[ -n "${MODEL_PATH}" && "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${ROOT_DIR}/${MODEL_PATH}"
fi
if [[ "${CUSTOM_POLICY}" == "gnnrl_model" && -z "${MODEL_PATH}" ]]; then
    echo "--model-path is required when --custom-policy=gnnrl_model" >&2
    exit 1
fi
if [[ "${CUSTOM_POLICY}" == "gnnrl_model" && ! -f "${MODEL_PATH}" ]]; then
    echo "--model-path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ "${CUSTOM_POLICY}" == "gnnrl_model" && "${CUSTOM_UE_PRG}" != "1" ]]; then
    echo "--custom-policy=gnnrl_model requires --custom-ue-prg=1" >&2
    exit 1
fi
if ! [[ "${COMPACT_TTI_LOG}" =~ ^[01]$ ]]; then
    echo "--compact-tti-log must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${PROGRESS_TTI_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "--progress-tti must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${KPI_TTI_LOG_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "--kpi-tti-log must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${COMPARE_TTI_INTERVAL}" =~ ^[0-9]+$ ]]; then
    echo "--compare-tti must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "${REPLAY_DUMP}" =~ ^[01]$ ]]; then
    echo "--replay-dump must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${ONLINE_BRIDGE}" =~ ^[01]$ ]]; then
    echo "--online-bridge must be 0 or 1" >&2
    exit 1
fi
if [[ -z "${ONLINE_SOCKET}" ]]; then
    echo "--online-socket cannot be empty" >&2
    exit 1
fi
EXEC_MODE="$(echo "${EXEC_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" != "both" && "${EXEC_MODE}" != "gpu" ]]; then
    echo "--exec-mode must be both or gpu" >&2
    exit 1
fi
if ! [[ "${RESTORE_PARAMS}" =~ ^[01]$ ]]; then
    echo "--restore-params must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${TRAFFIC_PERCENT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--traffic-percent must be a number in [0,100]" >&2
    exit 1
fi
awk "BEGIN {exit !(${TRAFFIC_PERCENT} >= 0 && ${TRAFFIC_PERCENT} <= 100)}" || {
    echo "--traffic-percent must be within [0,100]" >&2
    exit 1
}
if ! [[ "${PACKET_SIZE_BYTES}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--packet-size-bytes must be a positive number" >&2
    exit 1
fi
if ! [[ "${TRAFFIC_ARRIVAL_RATE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "--traffic-arrival-rate must be a non-negative number" >&2
    exit 1
fi

IFS=',' read -r -a PROFILE_ARR <<< "${CDL_PROFILES}"
IFS=',' read -r -a DELAY_ARR <<< "${CDL_DELAY_SPREADS_NS}"
if [[ ${#PROFILE_ARR[@]} -eq 0 ]]; then
    echo "--cdl-profiles is empty" >&2
    exit 1
fi
if [[ ${#DELAY_ARR[@]} -eq 0 ]]; then
    echo "--cdl-delay-spreads is empty" >&2
    exit 1
fi

if [[ "${FADING_MODE}" == "3" || "${FADING_MODE}" == "4" ]]; then
    cat >&2 <<EOF
[Stage-B] ERROR: pure 7-cell topology (numCellConst=7, numCoorCellConst=7) currently crashes with CDL (fading-mode ${FADING_MODE}).
Use --fading-mode 0/1/2 for stable baseline runs in this topology.
EOF
    exit 1
fi

if [[ "${FADING_MODE}" == "0" || "${FADING_MODE}" == "1" || "${FADING_MODE}" == "2" ]]; then
    PROFILE_ARR=("NA")
    DELAY_ARR=("0")
fi

if [[ -n "${REPLAY_DIR}" && "${REPLAY_DIR}" != /* ]]; then
    REPLAY_DIR="${ROOT_DIR}/${REPLAY_DIR}"
fi
if [[ "${ONLINE_SOCKET}" != /* ]]; then
    ONLINE_SOCKET="${ROOT_DIR}/${ONLINE_SOCKET}"
fi

PARAM_BAK="$(mktemp)"
cp "${PARAM_FILE}" "${PARAM_BAK}"
restore_params() {
    cp "${PARAM_BAK}" "${PARAM_FILE}"
    rm -f "${PARAM_BAK}"
}
cleanup_params() {
    if [[ "${RESTORE_PARAMS}" == "1" ]]; then
        restore_params
    else
        rm -f "${PARAM_BAK}"
    fi
}
trap cleanup_params EXIT

set_param() {
    local param="$1"
    local value="$2"
    python3 "${UPDATE_TOOL}" --file "${PARAM_FILE}" --param "${param}" --value "${value}"
}

print_effective_compile_params() {
    echo "[Stage-B] Compile-time parameter snapshot:"
    sed -n '/#define numCellConst/p; /#define numCoorCellConst/p; /#define numUePerCellConst/p; /#define numActiveUePerCellConst/p; /#define totNumUesConst/p; /#define totNumActiveUesConst/p; /#define nPrbsPerGrpConst/p; /#define nPrbGrpsConst/p; /#define gpuAllocTypeConst/p; /#define cpuAllocTypeConst/p' "${PARAM_FILE}" | sed 's/^/[Stage-B]   /'
}

sanitize_reason() {
    local s="$1"
    s="$(echo "${s}" | tr '[:upper:]' '[:lower:]')"
    s="$(echo "${s}" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
    if [[ -z "${s}" ]]; then
        s="unknown"
    fi
    echo "${s}"
}

have_rg() {
    command -v rg >/dev/null 2>&1
}

log_search() {
    local pattern="$1"
    local log_file="$2"
    if have_rg; then
        rg -N "${pattern}" "${log_file}" 2>/dev/null || true
    else
        grep -E "${pattern}" "${log_file}" 2>/dev/null || true
    fi
}

log_search_quiet() {
    local pattern="$1"
    local log_file="$2"
    if have_rg; then
        rg -q "${pattern}" "${log_file}" 2>/dev/null
    else
        grep -qE "${pattern}" "${log_file}" 2>/dev/null
    fi
}

log_search_count() {
    local pattern="$1"
    local log_file="$2"
    if have_rg; then
        rg -c "${pattern}" "${log_file}" 2>/dev/null || true
    else
        grep -cE "${pattern}" "${log_file}" 2>/dev/null || true
    fi
}

log_search_only_matches() {
    local pattern="$1"
    local log_file="$2"
    if have_rg; then
        rg -o "${pattern}" "${log_file}" 2>/dev/null || true
    else
        grep -oE "${pattern}" "${log_file}" 2>/dev/null || true
    fi
}

extract_last_tti() {
    local log_file="$1"
    local last_tti
    last_tti="$(log_search_only_matches "TTI [0-9]+" "${log_file}" | awk '{print $2}' | tail -n 1 || true)"
    if [[ -z "${last_tti}" ]]; then
        last_tti="$(sed -n 's/.*TTI_PROGRESS \([0-9][0-9]*\)\/[0-9][0-9]*/\1/p' "${log_file}" | tail -n 1)"
    fi
    if [[ -z "${last_tti}" ]]; then
        last_tti="unknown"
    fi
    echo "${last_tti}"
}

extract_last_stage() {
    local log_file="$1"
    log_search "GPU channel generated|API setup completed|CSI update: subband SINR calculation setup completed|CSI update: subband SINR calculation run completed|CSI update: wideband SINR calculation setup completed|CSI update: wideband SINR calculation run completed|CSI update: subband and wideband SINRS copied to CPU structures|GPU PF UE selection setup completed|GPU PF UE selection run completed|GPU UE downselection completed|CPU PF UE selection completed|CPU UE downselection completed|GPU scheduler setup started|GPU scheduler setup completed|GPU scheduler run started|GPU scheduler run completed|CPU scheduler run started|CPU scheduler run completed|PRB scheduling solution computed|GPU Layer selection solution computed|CPU Layer selection solution computed|GPU MCS selection solution computed|CPU MCS selection solution computed|Scheduling solution transferred to host" "${log_file}" | tail -n 1 || true
}

count_mismatch_lines() {
    local log_file="$1"
    local count
    count="$(log_search_count "^Failure: CPU and GPU .* do not match" "${log_file}")"
    if [[ -z "${count}" ]]; then
        count="0"
    fi
    echo "${count}"
}

classify_failure_reason() {
    local log_file="$1"
    local rc="$2"
    local last_tti
    local last_stage
    local stage_key
    local mismatch_count
    local reason

    last_tti="$(extract_last_tti "${log_file}")"
    last_stage="$(extract_last_stage "${log_file}")"
    stage_key="$(sanitize_reason "${last_stage}")"
    mismatch_count="$(count_mismatch_lines "${log_file}")"
    reason="exit_code_${rc}"

    if log_search_quiet "ERROR: CDL with LOS path is not supported yet" "${log_file}"; then
        reason="cdl_los_path_not_supported"
    elif log_search_quiet "cudaError|CUDA error|an illegal memory access was encountered|device-side assert triggered|launch timed out and was terminated" "${log_file}"; then
        reason="cuda_runtime_error"
    elif [[ "${rc}" -eq 124 ]]; then
        reason="timeout_after_${stage_key}_tti_${last_tti}"
    elif [[ "${rc}" -eq 137 ]]; then
        reason="killed_after_timeout_or_oom_tti_${last_tti}"
    fi

    if [[ "${mismatch_count}" =~ ^[0-9]+$ ]] && [[ "${mismatch_count}" -gt 0 ]]; then
        reason="${reason}_with_cpu_gpu_mismatch_${mismatch_count}"
    fi

    echo "${reason}"
}

scenario_completed_all_tti() {
    local log_file="$1"
    local completed_idx
    local final_idx=$((TTI_COUNT - 1))
    if [[ "${TTI_COUNT}" -gt 0 ]] && log_search_quiet "TTI_PROGRESS ${final_idx}/${final_idx}" "${log_file}"; then
        return 0
    fi
    if log_search_quiet "^TRAFFIC_KPI flows=" "${log_file}" && log_search_quiet "^UE_KPI_HEADER " "${log_file}"; then
        return 0
    fi
    completed_idx="$(sed -n 's/.*TTI_PROGRESS \([0-9][0-9]*\)\/\([0-9][0-9]*\).*/\1 \2/p' "${log_file}" | awk '$1==$2{print $1}' | tail -n 1)"
    [[ -n "${completed_idx}" ]]
}

is_perf_check_warn_case() {
    local log_file="$1"
    local rc="$2"
    if [[ "${rc}" -eq 0 ]]; then
        return 1
    fi
    scenario_completed_all_tti "${log_file}" || return 1
    log_search_quiet "CPU and GPU scheduler performance check result: FAIL" "${log_file}" || return 1
    if log_search_quiet "cudaError|CUDA error|an illegal memory access was encountered|device-side assert triggered|launch timed out and was terminated|terminate called after throwing|Segmentation fault|Aborted \\(core dumped\\)" "${log_file}"; then
        return 1
    fi
    return 0
}

extract_perf_warn_reason() {
    local log_file="$1"
    local per_ue_gap
    local sum_gap
    local observed_idx
    local observed_tti
    local mismatch_suffix=""
    per_ue_gap="$(sed -n 's/.*per-UE throughput CDFs = \([0-9.eE+-]*%\).*/\1/p' "${log_file}" | tail -n 1)"
    sum_gap="$(sed -n 's/.*sum throughput curves = \([0-9.eE+-]*%\).*/\1/p' "${log_file}" | tail -n 1)"
    observed_idx="$(sed -n 's/.*TTI_PROGRESS \([0-9][0-9]*\)\/\([0-9][0-9]*\).*/\1 \2/p' "${log_file}" | awk '$1==$2{print $1}' | tail -n 1)"
    if [[ -n "${observed_idx}" ]]; then
        observed_tti=$((observed_idx + 1))
        if [[ "${TTI_COUNT}" =~ ^[0-9]+$ ]] && [[ "${TTI_COUNT}" -gt 0 ]] && [[ "${observed_tti}" -ne "${TTI_COUNT}" ]]; then
            mismatch_suffix="_expected_tti_${TTI_COUNT}_observed_tti_${observed_tti}"
        fi
    fi
    if [[ -n "${per_ue_gap}" && -n "${sum_gap}" ]]; then
        echo "perf_check_fail_after_full_run_per_ue_gap_${per_ue_gap}_sum_gap_${sum_gap}${mismatch_suffix}" | tr '%.' 'pd'
    elif [[ -n "${per_ue_gap}" ]]; then
        echo "perf_check_fail_after_full_run_per_ue_gap_${per_ue_gap}${mismatch_suffix}" | tr '%.' 'pd'
    elif [[ -n "${mismatch_suffix}" ]]; then
        echo "perf_check_fail_after_full_run${mismatch_suffix}"
    else
        echo "perf_check_fail_after_full_run"
    fi
}

write_failure_diagnosis() {
    local out_file="$1"
    local log_file="$2"
    local rc="$3"
    local reason="$4"
    local last_tti
    local last_stage
    local mismatch_count

    last_tti="$(extract_last_tti "${log_file}")"
    last_stage="$(extract_last_stage "${log_file}")"
    mismatch_count="$(count_mismatch_lines "${log_file}")"
    {
        echo "rc=${rc}"
        echo "reason=${reason}"
        echo "last_tti=${last_tti}"
        echo "last_stage=${last_stage}"
        echo "cpu_gpu_mismatch_count=${mismatch_count}"
        echo "last_20_lines:"
        tail -n 20 "${log_file}" 2>/dev/null || true
    } > "${out_file}"
}

write_pass_key_log() {
    local out_dir="$1"
    local log_file="${out_dir}/run.log"
    local key_file="${out_dir}/run_key.log"
    if [[ ! -f "${log_file}" ]]; then
        return
    fi

    {
        echo "summary_head:"
        head -n 20 "${log_file}" || true
        echo
        echo "summary_key_lines:"
        log_search "GPU CDL:|GPU TDL:|CPU and GPU scheduler per-UE throughput performance check result|CPU and GPU scheduler sum throughput performance check result|CPU and GPU scheduler performance check result|Largest gap \\(in percentage\\)|TRAFFIC_KPI" "${log_file}"
    } > "${key_file}"
}

compact_pass_outputs() {
    local out_dir="$1"
    local kpi_json="${out_dir}/kpi_summary.json"

    if [[ ! -f "${kpi_json}" ]]; then
        echo "[Stage-B] compact-output: skip cleanup in ${out_dir} (missing kpi_summary.json)"
        return
    fi

    write_pass_key_log "${out_dir}"
    rm -f \
        "${out_dir}/run.log" \
        "${out_dir}/output.txt" \
        "${out_dir}/output_short.txt" \
        "${out_dir}/snr.txt" \
        "${out_dir}/kpi_summary.txt"
}

echo "[Stage-B] Apply main experiment parameters (7-site only, no outer interferer ring, 4T4R, 30kHz, Type-0 bitmap allocation)..."
set_param gpuDeviceIdx "${GPU_ID}"
set_param numSimChnRlz "${TTI_COUNT}"
set_param seedConst "${TOPOLOGY_SEED}"
set_param slotDurationConst "0.5e-3"
set_param scsConst "30000.0"
set_param cellRadiusConst "500"
set_param numCellConst "7"
set_param numCoorCellConst "7"
set_param numUePerCellConst "${UE_PER_CELL}"
set_param numUeForGrpConst "${UE_PER_CELL}"
set_param numActiveUePerCellConst "${UE_PER_CELL}"
set_param totNumUesConst "numCoorCellConst*numUePerCellConst"
set_param totNumActiveUesConst "numCoorCellConst*numActiveUePerCellConst"
set_param nBsAntConst "4"
set_param nUeAntConst "4"
set_param nPrbsPerGrpConst "4"
set_param nPrbGrpsConst "68"
set_param gpuAllocTypeConst "0"
set_param cpuAllocTypeConst "0"
print_effective_compile_params
if [[ "${RESTORE_PARAMS}" == "0" ]]; then
    echo "[Stage-B] parameters.h will be kept after run so external rebuilds use the same Stage-B config."
else
    echo "[Stage-B] parameters.h will be restored on exit."
fi
BIN="${BUILD_DIR}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
CUSTOM_SCHED_SRC="${ROOT_DIR}/cuMAC/examples/customScheduler/CustomUePrgScheduler.cpp"
MAIN_SRC="${ROOT_DIR}/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp"

case "${BUILD_METHOD}" in
    phase4)
        if [[ ! -f "/.dockerenv" ]]; then
            cat >&2 <<EOF
This script should run inside the cuBB dev container for phase4 build.

Recommended:
  ./cuPHY-CP/container/run_aerial.sh
  ./testBenches/phase4_test_scripts/build_aerial_sdk.sh
  ./cuMAC/scripts/run_stageB_main_experiment.sh --build-method skip
EOF
            exit 1
        fi
        if [[ ! -x "${BUILD_AERIAL_SCRIPT}" ]]; then
            echo "Build script not found: ${BUILD_AERIAL_SCRIPT}" >&2
            exit 1
        fi
        echo "[Stage-B] Building SDK via phase4 script..."
        "${BUILD_AERIAL_SCRIPT}" --build_dir "${BUILD_DIR}"
        ;;
    cmake)
        if ! command -v cmake >/dev/null 2>&1; then
            echo "cmake not found. Use --build-method phase4 inside container." >&2
            exit 1
        fi
        if [[ ! -d "${BUILD_DIR}" ]]; then
            echo "Build directory does not exist: ${BUILD_DIR}" >&2
            exit 1
        fi
        cmake --build "${BUILD_DIR}" --target multiCellSchedulerUeSelection
        ;;
    skip)
        echo "[Stage-B] Skip build as requested."
        echo "[Stage-B] Skip mode assumes ${BIN} was built with the Stage-B compile-time params shown above."
        if [[ -f "${BIN}" && "${PARAM_FILE}" -nt "${BIN}" ]]; then
            echo "[Stage-B] Warning: ${PARAM_FILE} is newer than ${BIN}. Binary may still use stale compile-time constants (for example TTI count)." >&2
        fi
        if [[ "${CUSTOM_UE_PRG}" == "1" && -f "${BIN}" ]] && \
           ([[ -f "${CUSTOM_SCHED_SRC}" && "${CUSTOM_SCHED_SRC}" -nt "${BIN}" ]] || [[ -f "${MAIN_SRC}" && "${MAIN_SRC}" -nt "${BIN}" ]]); then
            cat >&2 <<EOF
[Stage-B] ERROR: --custom-ue-prg=1 requested, but binary is older than custom scheduler sources.
[Stage-B] Rebuild first, for example:
[Stage-B]   ./cuMAC/scripts/run_stageB_main_experiment.sh --build-method cmake --custom-ue-prg 1 ...
[Stage-B] or use --build-method phase4 inside container.
EOF
            exit 1
        fi
        if [[ "${ONLINE_BRIDGE}" == "1" && -f "${BIN}" ]] && \
           ([[ -f "${MAIN_SRC}" && "${MAIN_SRC}" -nt "${BIN}" ]]); then
            cat >&2 <<EOF
[Stage-B] ERROR: --online-bridge=1 requested, but binary is older than online bridge sources.
[Stage-B] Rebuild first, for example:
[Stage-B]   ./cuMAC/scripts/run_stageB_main_experiment.sh --build-method cmake --online-bridge 1 ...
[Stage-B] or use --build-method phase4 inside container.
EOF
            exit 1
        fi
        if [[ "${EXEC_MODE}" == "gpu" && -f "${BIN}" ]] && \
           ([[ -f "${MAIN_SRC}" && "${MAIN_SRC}" -nt "${BIN}" ]]); then
            cat >&2 <<EOF
[Stage-B] ERROR: --exec-mode=gpu requested, but binary is older than main experiment sources.
[Stage-B] Rebuild first, for example:
[Stage-B]   ./cuMAC/scripts/run_stageB_main_experiment.sh --build-method cmake --exec-mode gpu ...
[Stage-B] or use --build-method phase4 inside container.
EOF
            exit 1
        fi
        ;;
esac

if [[ ! -x "${BIN}" ]]; then
    echo "Executable not found: ${BIN}" >&2
    exit 1
fi

DL_IND=1
if [[ "${DL_UL}" == "ul" ]]; then
    DL_IND=0
fi

BASELINE_IND=0
if [[ "${BASELINE_SCHEDULER}" == "rr" ]]; then
    BASELINE_IND=1
fi

TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RUN_TAG}" ]]; then
    OUT_BASE="${ROOT_DIR}/output/stageB_main_experiment_${RUN_TAG}_${TS}"
else
    OUT_BASE="${ROOT_DIR}/output/stageB_main_experiment_${TS}"
fi
mkdir -p "${OUT_BASE}"
STATUS_FILE="${OUT_BASE}/matrix_status.txt"
echo "scenario,status,reason" > "${STATUS_FILE}"

echo "[Stage-B] Start scenario matrix..."
if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]]; then
    if command -v timeout >/dev/null 2>&1; then
        echo "[Stage-B] Scenario timeout enabled: ${RUN_TIMEOUT_SEC}s (kill-after ${KILL_AFTER_SEC}s)"
    else
        echo "[Stage-B] Warning: timeout command not found, timeout is ignored."
    fi
fi

for i in "${!PROFILE_ARR[@]}"; do
    raw_profile="${PROFILE_ARR[$i]}"
    profile="$(echo "${raw_profile}" | tr '[:lower:]' '[:upper:]' | xargs)"
    if [[ -z "${profile}" ]]; then
        continue
    fi
    delay="${DELAY_ARR[0]}"
    if [[ ${#DELAY_ARR[@]} -gt $i ]]; then
        delay="${DELAY_ARR[$i]}"
    fi
    if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
        echo "Invalid delay spread '${delay}' for profile '${profile}', skip."
        echo "CDL_${profile}_DS${delay}ns,SKIP,invalid_delay_spread" >> "${STATUS_FILE}"
        continue
    fi

    if [[ "${profile}" == "NA" ]]; then
        if [[ "${FADING_MODE}" == "0" ]]; then
            SCENARIO="RAYLEIGH"
        elif [[ "${FADING_MODE}" == "1" ]]; then
            SCENARIO="TDL_PRBG"
        else
            SCENARIO="TDL_SC_PRBG"
        fi
        delay="0"
    elif [[ "${profile}" == "D" && "${ALLOW_PROFILE_D}" == "0" ]]; then
        echo "[Stage-B] Skip CDL-D (LOS path currently unsupported in chanModels)."
        echo "CDL_${profile}_DS${delay}ns,SKIP,cdl_d_los_not_supported" >> "${STATUS_FILE}"
        continue
    fi

    if [[ "${profile}" != "NA" && "${profile}" != "A" && "${profile}" != "B" && "${profile}" != "C" && "${profile}" != "D" && "${profile}" != "E" ]]; then
        echo "[Stage-B] Skip invalid profile '${profile}'."
        echo "CDL_${profile}_DS${delay}ns,SKIP,invalid_profile" >> "${STATUS_FILE}"
        continue
    fi

    if [[ "${profile}" != "NA" ]]; then
        SCENARIO="CDL_${profile}_DS${delay}ns"
    fi
    OUT_DIR="${OUT_BASE}/${SCENARIO}"
    mkdir -p "${OUT_DIR}"
    LOG_FILE="${OUT_DIR}/run.log"
    SCENARIO_REPLAY_DIR=""
    if [[ "${REPLAY_DUMP}" == "1" ]]; then
        if [[ -n "${REPLAY_DIR}" ]]; then
            SCENARIO_REPLAY_DIR="${REPLAY_DIR}/${SCENARIO}"
        else
            SCENARIO_REPLAY_DIR="${OUT_DIR}/replay"
        fi
        mkdir -p "${SCENARIO_REPLAY_DIR}"
    fi

    echo "[Stage-B] Running ${SCENARIO}"
    echo "  baseline_scheduler=${BASELINE_SCHEDULER}"
    if [[ "${CUSTOM_UE_PRG}" == "1" ]]; then
        echo "  note=baseline scheduler selection is ignored by custom UE+PRG mode"
    fi
    echo "  traffic_packet_bytes=${PACKET_SIZE_BYTES}"
    echo "  traffic_arrival_rate_pkt_per_tti=${TRAFFIC_ARRIVAL_RATE}"
    echo "  cmd=${BIN} -d ${DL_IND} -b ${BASELINE_IND} -f ${FADING_MODE} -x ${CUSTOM_UE_PRG} -g ${TRAFFIC_PERCENT} -r ${PACKET_SIZE_BYTES}"
    if [[ "${CUSTOM_UE_PRG}" == "1" ]]; then
        echo "  custom_policy=${CUSTOM_POLICY}"
        if [[ "${CUSTOM_POLICY}" == "gnnrl_model" ]]; then
            echo "  model_path=${MODEL_PATH}"
            echo "  policy_timeout_ms=${POLICY_TIMEOUT_MS}"
        fi
    fi
    if [[ "${REPLAY_DUMP}" == "1" ]]; then
        echo "  replay_dump=1 replay_dir=${SCENARIO_REPLAY_DIR}"
    fi
    if [[ "${ONLINE_BRIDGE}" == "1" ]]; then
        echo "  online_bridge=1 online_socket=${ONLINE_SOCKET}"
    fi
    echo "  exec_mode=${EXEC_MODE}"
    if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]]; then
        echo "  timeout=${RUN_TIMEOUT_SEC}s"
    fi

    set +e
    (
        if [[ "${profile}" == "NA" ]]; then
            unset CUMAC_CDL_PROFILE
            unset CUMAC_CDL_DELAY_SPREAD_NS
        else
            export CUMAC_CDL_PROFILE="${profile}"
            export CUMAC_CDL_DELAY_SPREAD_NS="${delay}"
        fi
        export CUMAC_TOPOLOGY_SEED="${TOPOLOGY_SEED}"
        export CUMAC_UE_PLACEMENT_MODE="${UE_PLACEMENT}"
        export CUMAC_UE_RADIUS_SPLITS="${UE_RADIUS_SPLITS}"
        export CUMAC_UE_STRATA_COUNTS="${UE_STRATA_COUNTS}"
        export CUMAC_UE_VORONOI_CLIP="${UE_VORONOI_CLIP}"
        export CUMAC_COMPACT_TTI_LOG="${COMPACT_TTI_LOG}"
        export CUMAC_PROGRESS_TTI_INTERVAL="${PROGRESS_TTI_INTERVAL}"
        export CUMAC_TTI_KPI_LOG_INTERVAL="${KPI_TTI_LOG_INTERVAL}"
        export CUMAC_COMPARE_TTI_INTERVAL="${COMPARE_TTI_INTERVAL}"
        export CUMAC_TRAFFIC_ARRIVAL_RATE="${TRAFFIC_ARRIVAL_RATE}"
        export CUMAC_CUSTOM_POLICY="${CUSTOM_POLICY}"
        export CUMAC_GNNRL_MODEL_PATH="${MODEL_PATH}"
        export CUMAC_POLICY_TIMEOUT_MS="${POLICY_TIMEOUT_MS}"
        export CUMAC_RL_REPLAY_DUMP="${REPLAY_DUMP}"
        export CUMAC_RL_REPLAY_DIR="${SCENARIO_REPLAY_DIR}"
        export CUMAC_RL_REPLAY_SCENARIO="${SCENARIO}"
        export CUMAC_ONLINE_BRIDGE="${ONLINE_BRIDGE}"
        export CUMAC_ONLINE_SOCKET="${ONLINE_SOCKET}"
        export CUMAC_EXEC_MODE="${EXEC_MODE}"
        cd "${OUT_DIR}"
        RUNNER=("${BIN}" -d "${DL_IND}" -b "${BASELINE_IND}" -f "${FADING_MODE}" -x "${CUSTOM_UE_PRG}" -g "${TRAFFIC_PERCENT}" -r "${PACKET_SIZE_BYTES}")
        if command -v stdbuf >/dev/null 2>&1; then
            RUNNER=(stdbuf -oL -eL "${RUNNER[@]}")
        fi
        if [[ "${RUN_TIMEOUT_SEC}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
            timeout --signal=TERM --kill-after="${KILL_AFTER_SEC}s" "${RUN_TIMEOUT_SEC}s" \
                "${RUNNER[@]}"
        else
            "${RUNNER[@]}"
        fi
    ) 2>&1 | tee "${LOG_FILE}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ ${rc} -ne 0 ]]; then
        if is_perf_check_warn_case "${LOG_FILE}" "${rc}"; then
            reason="$(extract_perf_warn_reason "${LOG_FILE}")"
            write_failure_diagnosis "${OUT_DIR}/fail_diagnosis.txt" "${LOG_FILE}" "${rc}" "${reason}"
            echo "[Stage-B] ${SCENARIO} completed all TTIs but perf check failed; mark as WARN, reason=${reason}"
            if [[ -f "${KPI_SCRIPT}" ]]; then
                python3 "${KPI_SCRIPT}" \
                    --output-dir "${OUT_DIR}" \
                    --slot-duration-ms 0.5 \
                    --traffic-percent "${TRAFFIC_PERCENT}" \
                    --packet-size-bytes "${PACKET_SIZE_BYTES}" \
                    --traffic-arrival-rate "${TRAFFIC_ARRIVAL_RATE}"
            fi
            if [[ "${COMPACT_OUTPUT}" == "1" ]]; then
                compact_pass_outputs "${OUT_DIR}"
            fi
            echo "${SCENARIO},WARN,${reason}" >> "${STATUS_FILE}"
            continue
        fi
        reason="$(classify_failure_reason "${LOG_FILE}" "${rc}")"
        write_failure_diagnosis "${OUT_DIR}/fail_diagnosis.txt" "${LOG_FILE}" "${rc}" "${reason}"
        echo "[Stage-B] ${SCENARIO} failed with code ${rc}, reason=${reason}"
        echo "${SCENARIO},FAIL,${reason}" >> "${STATUS_FILE}"
        continue
    fi

    if [[ -f "${KPI_SCRIPT}" ]]; then
        python3 "${KPI_SCRIPT}" \
            --output-dir "${OUT_DIR}" \
            --slot-duration-ms 0.5 \
            --traffic-percent "${TRAFFIC_PERCENT}" \
            --packet-size-bytes "${PACKET_SIZE_BYTES}" \
            --traffic-arrival-rate "${TRAFFIC_ARRIVAL_RATE}"
    fi

    if [[ "${COMPACT_OUTPUT}" == "1" ]]; then
        compact_pass_outputs "${OUT_DIR}"
    fi

    echo "${SCENARIO},PASS,ok" >> "${STATUS_FILE}"
done

if [[ -f "${MATRIX_SCRIPT}" ]]; then
    python3 "${MATRIX_SCRIPT}" --base-dir "${OUT_BASE}"
fi

if [[ "${COMPACT_OUTPUT}" == "1" ]]; then
    rm -f "${OUT_BASE}/stageB_kpi_matrix.txt"
fi

echo "[Stage-B] Done. Output base: ${OUT_BASE}"
