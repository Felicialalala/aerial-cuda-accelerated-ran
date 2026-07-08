#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
IMITATION_SCRIPT="${ROOT_DIR}/training/stageb_hgraph/online_imitation.py"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4"
GPU_ID=0
TTI_COUNT=2000
DL_UL="dl"
FADING_MODE=0
TOPOLOGY_SCENARIO="3cell"
CELL_RADIUS=500
UE_PER_CELL=12
TOTAL_UE_COUNT=""
TOPOLOGY_SEED=42
UE_PLACEMENT="uniform"
UE_RADIUS_SPLITS="0.33,0.66"
UE_STRATA_COUNTS=""
UE_VORONOI_CLIP=1
BS_TX_PATTERN="omni"
TRAFFIC_PERCENT=100
PACKET_SIZE_BYTES=3000
TRAFFIC_ARRIVAL_RATE=1.0
PACKET_TTL_TTI=0
PACKET_TTL_MS=200
CDL_PROFILES="NA"
CDL_DELAY_SPREADS_NS="0"
ALLOW_PROFILE_D=0
PRBS_PER_GROUP=16
CUSTOM_UE_PRG=0
BASELINE_SCHEDULER="pfq"
COMPACT_TTI_LOG=1
PROGRESS_TTI_INTERVAL=0
KPI_TTI_LOG_INTERVAL=100
COMPARE_TTI_INTERVAL=0
EXEC_MODE="both"
PRECODING="svd"
TAG=""
RESTORE_PARAMS=0

OUT_DIR="${ROOT_DIR}/training/stageb_hgraph/runs/online_imitation"
EPISODES=10
EPISODE_HORIZON=400
MAX_STEPS=0
TOPOLOGY_SEED_MODE="fixed"
SEED_LIST=""
SOCKET_PATH="/tmp/cumac_stageb_hgraph_imitation.sock"
CONNECT_TIMEOUT_S=20.0
SIM_WAIT_TIMEOUT=10.0
SIM_LOG_MODE="file"
ONLINE_PERSISTENT=1
UE_PRG_TOPK=4
UE_PRG_DIVERSITY_EXTRA=0
TB_ERR_EMA_ALPHA=0.10
HIDDEN_DIM=128
MESSAGE_LAYERS=2
DROPOUT=0.0
ACTION_HEAD_MODE="slot"
CANDIDATE_BLANK_LOGIT_BIAS=0.0
CANDIDATE_SUCCESS_LOGIT_SCALE=1.0
CANDIDATE_BLANK_SUCCESS_SCALE=1.0
CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
LR=3e-4
WEIGHT_DECAY=1e-5
GRAD_CLIP=5.0
PRINT_EVERY=50
TRAIN_SEED=42
DEVICE="auto"
TEACHER_MODE="pfq_passthrough"
SAVE_EVERY_EPISODES=1

EXTRA_SIM_ENV=()

usage() {
    cat <<EOF
Run Stage-B sparse-entity graph online imitation warm start.

Usage:
  $(basename "$0") [options]

Stage-B scenario options:
  --build-dir <path>
  --build-method <m>          phase4 | cmake | skip
  --gpu <id>
  --tti <count>
  --mode <dl|ul>
  --fading-mode <0|1|2|3|4>
  --topology-scenario <m>     7cell | 3cell
  --cell-radius <m>           Cell radius in meters (ISD/site spacing is 2x this value)
  --ue-per-cell <n>
  --total-ue-count <n>
  --topology-seed <n>
  --ue-placement <m>          uniform | stratified
  --ue-radius-splits <a,b>
  --ue-strata-counts <a,b,c>
  --ue-voronoi-clip <0|1>
  --bs-tx-pattern <m>         omni | sector
  --traffic-percent <p>
  --packet-size-bytes <b>
  --traffic-arrival-rate <r>
  --packet-ttl-tti <n>
  --packet-ttl-ms <v>
  --prbs-per-group <n>
  --cdl-profiles <list>
  --cdl-delay-spreads <list>
  --allow-profile-d <0|1>
  --custom-ue-prg <0|1>
  --baseline-scheduler <m>    pf | pfq | rr
  --compact-tti-log <0|1>
  --progress-tti <n>
  --kpi-tti-log <n>
  --compare-tti <n>
  --exec-mode <mode>          must be both
  --precoding <mode>         none | svd
  --tag <name>
  --restore-params <0|1>

Online imitation options:
  --out-dir <path>
  --episodes <n>
  --episode-horizon <n>
  --max-steps <n>             0 means use --episode-horizon
  --topology-seed-mode <m>    auto | fixed | sequential | list_cycle
  --seed-list <list>
  --socket-path <path>
  --connect-timeout-s <sec>
  --sim-wait-timeout <sec>
  --sim-log-mode <m>          file | inherit
  --online-persistent <0|1>
  --ue-prg-topk <n>
  --ue-prg-diversity-extra <n>
  --tb-err-ema-alpha <v>
  --hidden-dim <n>
  --message-layers <n>
  --dropout <v>
  --action-head-mode <m>      slot | candidate
  --candidate-blank-logit-bias <v>
  --candidate-success-logit-scale <v>
  --candidate-blank-success-scale <v>
  --candidate-pfq-anchor-logit-coef <v>
  --lr <v>
  --weight-decay <v>
  --grad-clip <v>
  --print-every <n>
  --seed <n>
  --device <auto|cpu|cuda>
  --teacher-mode <m>          heuristic_graph | pfq_passthrough
  --save-every-episodes <n>
  --sim-env <KEY=VALUE>       Extra simulator env var, may be repeated
  -h, --help
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
        --topology-scenario) TOPOLOGY_SCENARIO="$2"; shift 2 ;;
        --cell-radius) CELL_RADIUS="$2"; shift 2 ;;
        --ue-per-cell) UE_PER_CELL="$2"; shift 2 ;;
        --total-ue-count|--ue-count) TOTAL_UE_COUNT="$2"; shift 2 ;;
        --topology-seed) TOPOLOGY_SEED="$2"; shift 2 ;;
        --ue-placement) UE_PLACEMENT="$2"; shift 2 ;;
        --ue-radius-splits) UE_RADIUS_SPLITS="$2"; shift 2 ;;
        --ue-strata-counts) UE_STRATA_COUNTS="$2"; shift 2 ;;
        --ue-voronoi-clip) UE_VORONOI_CLIP="$2"; shift 2 ;;
        --bs-tx-pattern) BS_TX_PATTERN="$2"; shift 2 ;;
        --traffic-percent) TRAFFIC_PERCENT="$2"; shift 2 ;;
        --packet-size-bytes|--traffic-rate) PACKET_SIZE_BYTES="$2"; shift 2 ;;
        --traffic-arrival-rate) TRAFFIC_ARRIVAL_RATE="$2"; shift 2 ;;
        --packet-ttl-tti) PACKET_TTL_TTI="$2"; shift 2 ;;
        --packet-ttl-ms) PACKET_TTL_MS="$2"; shift 2 ;;
        --prbs-per-group) PRBS_PER_GROUP="$2"; shift 2 ;;
        --cdl-profiles) CDL_PROFILES="$2"; shift 2 ;;
        --cdl-delay-spreads) CDL_DELAY_SPREADS_NS="$2"; shift 2 ;;
        --allow-profile-d) ALLOW_PROFILE_D="$2"; shift 2 ;;
        --custom-ue-prg) CUSTOM_UE_PRG="$2"; shift 2 ;;
        --baseline-scheduler) BASELINE_SCHEDULER="$2"; shift 2 ;;
        --compact-tti-log) COMPACT_TTI_LOG="$2"; shift 2 ;;
        --progress-tti) PROGRESS_TTI_INTERVAL="$2"; shift 2 ;;
        --kpi-tti-log) KPI_TTI_LOG_INTERVAL="$2"; shift 2 ;;
        --compare-tti) COMPARE_TTI_INTERVAL="$2"; shift 2 ;;
        --exec-mode) EXEC_MODE="$2"; shift 2 ;;
        --precoding) PRECODING="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --restore-params) RESTORE_PARAMS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --episode-horizon) EPISODE_HORIZON="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --topology-seed-mode) TOPOLOGY_SEED_MODE="$2"; shift 2 ;;
        --seed-list) SEED_LIST="$2"; shift 2 ;;
        --socket-path) SOCKET_PATH="$2"; shift 2 ;;
        --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2 ;;
        --sim-wait-timeout) SIM_WAIT_TIMEOUT="$2"; shift 2 ;;
        --sim-log-mode) SIM_LOG_MODE="$2"; shift 2 ;;
        --online-persistent) ONLINE_PERSISTENT="$2"; shift 2 ;;
        --ue-prg-topk) UE_PRG_TOPK="$2"; shift 2 ;;
        --ue-prg-diversity-extra) UE_PRG_DIVERSITY_EXTRA="$2"; shift 2 ;;
        --tb-err-ema-alpha) TB_ERR_EMA_ALPHA="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --message-layers) MESSAGE_LAYERS="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --action-head-mode) ACTION_HEAD_MODE="$2"; shift 2 ;;
        --candidate-blank-logit-bias) CANDIDATE_BLANK_LOGIT_BIAS="$2"; shift 2 ;;
        --candidate-success-logit-scale) CANDIDATE_SUCCESS_LOGIT_SCALE="$2"; shift 2 ;;
        --candidate-blank-success-scale) CANDIDATE_BLANK_SUCCESS_SCALE="$2"; shift 2 ;;
        --candidate-pfq-anchor-logit-coef) CANDIDATE_PFQ_ANCHOR_LOGIT_COEF="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
        --print-every) PRINT_EVERY="$2"; shift 2 ;;
        --seed) TRAIN_SEED="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --teacher-mode) TEACHER_MODE="$2"; shift 2 ;;
        --save-every-episodes) SAVE_EVERY_EPISODES="$2"; shift 2 ;;
        --sim-env) EXTRA_SIM_ENV+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -x "${RUN_SCRIPT}" ]]; then
    echo "Missing run script: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${IMITATION_SCRIPT}" ]]; then
    echo "Missing online imitation script: ${IMITATION_SCRIPT}" >&2
    exit 1
fi

EXEC_MODE="$(echo "${EXEC_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" != "both" ]]; then
    echo "--exec-mode must be both for online bridge imitation." >&2
    exit 1
fi

TOPOLOGY_SCENARIO="$(echo "${TOPOLOGY_SCENARIO}" | tr '[:upper:]' '[:lower:]')"
case "${TOPOLOGY_SCENARIO}" in
    7|7cell|7cells|7site|7-site)
        TOPOLOGY_SCENARIO="7cell"
        TOPOLOGY_NUM_CELLS=7
        ;;
    3|3cell|3cells|3site|3-site)
        TOPOLOGY_SCENARIO="3cell"
        TOPOLOGY_NUM_CELLS=3
        ;;
    *)
        echo "--topology-scenario must be 7cell or 3cell" >&2
        exit 1
        ;;
esac

if [[ -n "${TOTAL_UE_COUNT}" ]]; then
    if (( TOTAL_UE_COUNT % TOPOLOGY_NUM_CELLS != 0 )); then
        echo "--total-ue-count must be divisible by the coordinated cell count (${TOPOLOGY_NUM_CELLS})" >&2
        exit 1
    fi
    UE_PER_CELL=$((TOTAL_UE_COUNT / TOPOLOGY_NUM_CELLS))
fi
TOTAL_UE_COUNT=$((UE_PER_CELL * TOPOLOGY_NUM_CELLS))

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
fi
if [[ "${OUT_DIR}" != /* ]]; then
    OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi
if [[ "${SOCKET_PATH}" != /* ]]; then
    SOCKET_PATH="${ROOT_DIR}/${SOCKET_PATH}"
fi

mkdir -p "${OUT_DIR}"
SIM_CWD="${OUT_DIR}/sim_runtime"
mkdir -p "${SIM_CWD}"

BIN="${BUILD_DIR}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"

BUILD_ARGS=(
    --build-dir "${BUILD_DIR}"
    --build-method "${BUILD_METHOD}"
    --build-only 1
    --gpu "${GPU_ID}"
    --tti "${TTI_COUNT}"
    --mode "${DL_UL}"
    --fading-mode "${FADING_MODE}"
    --topology-scenario "${TOPOLOGY_SCENARIO}"
    --cell-radius "${CELL_RADIUS}"
    --ue-per-cell "${UE_PER_CELL}"
    --topology-seed "${TOPOLOGY_SEED}"
    --ue-placement "${UE_PLACEMENT}"
    --ue-radius-splits "${UE_RADIUS_SPLITS}"
    --ue-strata-counts "${UE_STRATA_COUNTS}"
    --ue-voronoi-clip "${UE_VORONOI_CLIP}"
    --bs-tx-pattern "${BS_TX_PATTERN}"
    --traffic-percent "${TRAFFIC_PERCENT}"
    --packet-size-bytes "${PACKET_SIZE_BYTES}"
    --traffic-arrival-rate "${TRAFFIC_ARRIVAL_RATE}"
    --packet-ttl-tti "${PACKET_TTL_TTI}"
    --packet-ttl-ms "${PACKET_TTL_MS}"
    --prbs-per-group "${PRBS_PER_GROUP}"
    --cdl-profiles "${CDL_PROFILES}"
    --cdl-delay-spreads "${CDL_DELAY_SPREADS_NS}"
    --allow-profile-d "${ALLOW_PROFILE_D}"
    --custom-ue-prg "${CUSTOM_UE_PRG}"
    --baseline-scheduler "${BASELINE_SCHEDULER}"
    --custom-policy gnnrl
    --gnnrl-action-mode joint
    --compact-tti-log "${COMPACT_TTI_LOG}"
    --progress-tti "${PROGRESS_TTI_INTERVAL}"
    --kpi-tti-log "${KPI_TTI_LOG_INTERVAL}"
    --compare-tti "${COMPARE_TTI_INTERVAL}"
    --online-bridge 1
    --online-socket "${SOCKET_PATH}"
    --exec-mode both
    --precoding "${PRECODING}"
    --restore-params "${RESTORE_PARAMS}"
)
if [[ -n "${TAG}" ]]; then
    BUILD_ARGS+=(--tag "${TAG}")
fi

echo "[Stage-B HGraph] preparing binary with run_stageB_main_experiment.sh --build-only 1"
"${RUN_SCRIPT}" "${BUILD_ARGS[@]}"

if [[ ! -x "${BIN}" ]]; then
    echo "Executable not found after build: ${BIN}" >&2
    exit 1
fi

DL_IND=1
if [[ "${DL_UL}" == "ul" ]]; then
    DL_IND=0
fi

BASELINE_IND=0
if [[ "${BASELINE_SCHEDULER}" == "rr" ]]; then
    BASELINE_IND=1
elif [[ "${BASELINE_SCHEDULER}" == "pfq" ]]; then
    BASELINE_IND=2
fi

PROFILE="${CDL_PROFILES%%,*}"
PROFILE="$(echo "${PROFILE}" | tr '[:lower:]' '[:upper:]' | xargs)"
DELAY="${CDL_DELAY_SPREADS_NS%%,*}"
DELAY="$(echo "${DELAY}" | xargs)"
if [[ "${FADING_MODE}" == "0" || "${FADING_MODE}" == "1" || "${FADING_MODE}" == "2" ]]; then
    PROFILE="NA"
    DELAY="0"
fi

SIM_ENV=(
    "CUMAC_TOPOLOGY_SEED=${TOPOLOGY_SEED}"
    "CUMAC_UE_PLACEMENT_MODE=${UE_PLACEMENT}"
    "CUMAC_UE_RADIUS_SPLITS=${UE_RADIUS_SPLITS}"
    "CUMAC_UE_STRATA_COUNTS=${UE_STRATA_COUNTS}"
    "CUMAC_UE_VORONOI_CLIP=${UE_VORONOI_CLIP}"
    "CUMAC_BS_TX_PATTERN=${BS_TX_PATTERN}"
    "CUMAC_COMPACT_TTI_LOG=${COMPACT_TTI_LOG}"
    "CUMAC_PROGRESS_TTI_INTERVAL=${PROGRESS_TTI_INTERVAL}"
    "CUMAC_TTI_KPI_LOG_INTERVAL=${KPI_TTI_LOG_INTERVAL}"
    "CUMAC_COMPARE_TTI_INTERVAL=${COMPARE_TTI_INTERVAL}"
    "CUMAC_TRAFFIC_ARRIVAL_RATE=${TRAFFIC_ARRIVAL_RATE}"
    "CUMAC_PACKET_TTL_TTI=${PACKET_TTL_TTI}"
    "CUMAC_PACKET_TTL_MS=${PACKET_TTL_MS}"
    "CUMAC_EXEC_MODE=both"
    "CUMAC_ONLINE_REWARD_MODE=goodput_only"
    "CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1"
    "CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1"
)
if [[ "${PROFILE}" != "NA" ]]; then
    SIM_ENV+=(
        "CUMAC_CDL_PROFILE=${PROFILE}"
        "CUMAC_CDL_DELAY_SPREAD_NS=${DELAY}"
    )
fi
for kv in "${EXTRA_SIM_ENV[@]}"; do
    SIM_ENV+=("${kv}")
done

SIM_ARGS="-d ${DL_IND} -b ${BASELINE_IND} -f ${FADING_MODE} -x ${CUSTOM_UE_PRG} -g ${TRAFFIC_PERCENT} -r ${PACKET_SIZE_BYTES}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

echo "[Stage-B HGraph] sim_bin=${BIN}"
echo "[Stage-B HGraph] sim_cwd=${SIM_CWD}"
echo "[Stage-B HGraph] sim_args=${SIM_ARGS}"
echo "[Stage-B HGraph] cell_radius_m=${CELL_RADIUS} site_spacing_m=$((2 * CELL_RADIUS)) ue_per_cell=${UE_PER_CELL} total_ue_count=${TOTAL_UE_COUNT}"
echo "[Stage-B HGraph] topology_seed_mode=${TOPOLOGY_SEED_MODE} topology_seed=${TOPOLOGY_SEED} seed_list=${SEED_LIST:-none}"
echo "[Stage-B HGraph] episode_horizon=${EPISODE_HORIZON} max_steps=${MAX_STEPS} episodes=${EPISODES}"
echo "[Stage-B HGraph] prbs_per_group=${PRBS_PER_GROUP} packet_size_bytes=${PACKET_SIZE_BYTES} packet_ttl_ms=${PACKET_TTL_MS} precoding=${PRECODING}"
echo "[Stage-B HGraph] teacher_mode=${TEACHER_MODE} action_head_mode=${ACTION_HEAD_MODE} hidden_dim=${HIDDEN_DIM} message_layers=${MESSAGE_LAYERS} lr=${LR}"

CMD=(
    "${PYTHON_BIN}" "${IMITATION_SCRIPT}"
    --sim-bin "${BIN}"
    --sim-args "${SIM_ARGS}"
    --sim-cwd "${SIM_CWD}"
    --socket-path "${SOCKET_PATH}"
    --connect-timeout-s "${CONNECT_TIMEOUT_S}"
    --sim-wait-timeout "${SIM_WAIT_TIMEOUT}"
    --sim-log-mode "${SIM_LOG_MODE}"
    --online-persistent "${ONLINE_PERSISTENT}"
    --episode-horizon "${EPISODE_HORIZON}"
    --max-steps "${MAX_STEPS}"
    --episodes "${EPISODES}"
    --topology-seed "${TOPOLOGY_SEED}"
    --topology-seed-mode "${TOPOLOGY_SEED_MODE}"
    --seed-list "${SEED_LIST}"
    --out-dir "${OUT_DIR}"
    --ue-prg-topk "${UE_PRG_TOPK}"
    --ue-prg-diversity-extra "${UE_PRG_DIVERSITY_EXTRA}"
    --tb-err-ema-alpha "${TB_ERR_EMA_ALPHA}"
    --hidden-dim "${HIDDEN_DIM}"
    --message-layers "${MESSAGE_LAYERS}"
    --dropout "${DROPOUT}"
    --action-head-mode "${ACTION_HEAD_MODE}"
    --candidate-blank-logit-bias "${CANDIDATE_BLANK_LOGIT_BIAS}"
    --candidate-success-logit-scale "${CANDIDATE_SUCCESS_LOGIT_SCALE}"
    --candidate-blank-success-scale "${CANDIDATE_BLANK_SUCCESS_SCALE}"
    --candidate-pfq-anchor-logit-coef "${CANDIDATE_PFQ_ANCHOR_LOGIT_COEF}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --grad-clip "${GRAD_CLIP}"
    --print-every "${PRINT_EVERY}"
    --seed "${TRAIN_SEED}"
    --device "${DEVICE}"
    --teacher-mode "${TEACHER_MODE}"
    --save-every-episodes "${SAVE_EVERY_EPISODES}"
)
for kv in "${SIM_ENV[@]}"; do
    CMD+=(--sim-env "${kv}")
done

echo "[Stage-B HGraph] running online imitation warm start"
"${CMD[@]}"
