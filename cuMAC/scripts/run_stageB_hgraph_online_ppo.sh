#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
PPO_SCRIPT="${ROOT_DIR}/training/stageb_hgraph/online_ppo.py"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4"
GPU_ID=0
TTI_COUNT=2000
DL_UL="dl"
FADING_MODE=0
TOPOLOGY_SCENARIO="3cell"
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

OUT_DIR="${ROOT_DIR}/training/stageb_hgraph/runs/online_ppo"
ITERATIONS=50
EPISODE_HORIZON=400
ROLLOUT_STEPS=400
TOPOLOGY_SEED_MODE="fixed"
SEED_LIST=""
SOCKET_PATH="/tmp/cumac_stageb_hgraph_ppo.sock"
CONNECT_TIMEOUT_S=20.0
SIM_WAIT_TIMEOUT=10.0
SIM_LOG_MODE="file"
ONLINE_PERSISTENT=1
UE_PRG_TOPK=4
UE_PRG_DIVERSITY_EXTRA=0
CANDIDATE_MODE="all_demand_ues"
MAX_PRG_CANDIDATES=0
SLOT_REPAIR_MAX_PER_CELL=0
SLOT_REPAIR_MIN_BACKLOG_BYTES=1.0
TB_ERR_EMA_ALPHA=0.10
HIDDEN_DIM=128
MESSAGE_LAYERS=2
DROPOUT=0.0
LR=1e-4
ACTOR_LR=0
CRITIC_LR=0
WEIGHT_DECAY=1e-5
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_EPS=0.2
VALUE_COEF=0.5
ENTROPY_COEF=0.01
TARGET_KL=0.05
UPDATE_EPOCHS=4
MINIBATCH_SIZE=64
NORMALIZE_STATE=1
NORMALIZE_REWARD=1
NORMALIZE_ADV=1
MAX_GRAD_NORM=1.0
VALUE_LOSS="huber"
VALUE_HUBER_BETA=10.0
TRAIN_SEED=42
DEVICE="auto"
PRINT_EVERY_ITERS=1
ROLLOUT_LOG_EVERY_STEPS=500
UPDATE_LOG_EVERY_MINIBATCHES=2
INIT_CHECKPOINT=""
IMITATION_EPISODES=0
IMITATION_MAX_STEPS=0
IMITATION_LR=0
TEACHER_MODE="pfq_passthrough"
TEACHER_AUX_COEF=0.20
TEACHER_AUX_DECAY_ITERS=80
REWARD_MODE="stageb_v1"
BEST_METRIC="goodput"
EARLY_STOP_PATIENCE=60
EARLY_STOP_MIN_DELTA=1.0
EARLY_STOP_WARMUP_ITERS=50
GOODPUT_WEIGHT=1.0
TB_ERR_WEIGHT=8.0
EXPIRY_WEIGHT=6.0
CONFLICT_WEIGHT=1.0
ICI_WEIGHT=1.0
URGENT_BACKLOG_WEIGHT=0.5
URGENT_TTL_MS=20.0
BACKLOG_DEBT_WEIGHT=0.0
PACKET_RATE_WEIGHT=0.0
PACKET_COMPLETION_WEIGHT=0.0
PACKET_RATE_SCALE_MBPS=10.0
PACKET_COMPLETION_SCALE=36.0
AGED_BACKLOG_WEIGHT=0.0

EXTRA_SIM_ENV=()

usage() {
    cat <<EOF
Run Stage-B sparse-entity graph online PPO fine-tuning.

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

Online PPO options:
  --out-dir <path>
  --iterations <n>
  --episode-horizon <n>
  --rollout-steps <n>
  --topology-seed-mode <m>    auto | fixed | sequential | list_cycle
  --seed-list <list>
  --socket-path <path>
  --connect-timeout-s <sec>
  --sim-wait-timeout <sec>
  --sim-log-mode <m>          file | inherit
  --online-persistent <0|1>
  --ue-prg-topk <n>
  --ue-prg-diversity-extra <n>
  --candidate-mode <m>        all_demand_ues | topk_diversity
  --max-prg-candidates <n>
  --slot-repair-max-per-cell <n>
  --slot-repair-min-backlog-bytes <v>
  --tb-err-ema-alpha <v>
  --hidden-dim <n>
  --message-layers <n>
  --dropout <v>
  --lr <v>
  --actor-lr <v>
  --critic-lr <v>
  --weight-decay <v>
  --gamma <v>
  --gae-lambda <v>
  --clip-eps <v>
  --value-coef <v>
  --entropy-coef <v>
  --target-kl <v>
  --update-epochs <n>
  --minibatch-size <n>
  --normalize-state <0|1>
  --normalize-reward <0|1>
  --normalize-adv <0|1>
  --max-grad-norm <v>
  --value-loss <m>            mse | huber
  --value-huber-beta <v>
  --seed <n>
  --device <auto|cpu|cuda>
  --print-every-iters <n>
  --rollout-log-every-steps <n>
  --update-log-every-minibatches <n>
  --init-checkpoint <path>
  --imitation-episodes <n>
  --imitation-max-steps <n>
  --imitation-lr <v>
  --teacher-mode <m>          heuristic_graph | pfq_passthrough
  --teacher-aux-coef <v>
  --teacher-aux-decay-iters <n>
  --reward-mode <raw_env|stageb_v1|stageb_blankaware>
  --best-metric <reward|goodput>
  --early-stop-patience <n>
  --early-stop-min-delta <v>
  --early-stop-warmup-iters <n>
  --goodput-weight <v>
  --tb-err-weight <v>
  --expiry-weight <v>
  --conflict-weight <v>
  --ici-weight <v>
  --urgent-backlog-weight <v>
  --urgent-ttl-ms <v>
  --backlog-debt-weight <v>
  --packet-rate-weight <v>
  --packet-completion-weight <v>
  --packet-rate-scale-mbps <v>
  --packet-completion-scale <v>
  --aged-backlog-weight <v>
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
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --episode-horizon) EPISODE_HORIZON="$2"; shift 2 ;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
        --topology-seed-mode) TOPOLOGY_SEED_MODE="$2"; shift 2 ;;
        --seed-list) SEED_LIST="$2"; shift 2 ;;
        --socket-path) SOCKET_PATH="$2"; shift 2 ;;
        --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2 ;;
        --sim-wait-timeout) SIM_WAIT_TIMEOUT="$2"; shift 2 ;;
        --sim-log-mode) SIM_LOG_MODE="$2"; shift 2 ;;
        --online-persistent) ONLINE_PERSISTENT="$2"; shift 2 ;;
        --ue-prg-topk) UE_PRG_TOPK="$2"; shift 2 ;;
        --ue-prg-diversity-extra) UE_PRG_DIVERSITY_EXTRA="$2"; shift 2 ;;
        --candidate-mode) CANDIDATE_MODE="$2"; shift 2 ;;
        --max-prg-candidates) MAX_PRG_CANDIDATES="$2"; shift 2 ;;
        --slot-repair-max-per-cell) SLOT_REPAIR_MAX_PER_CELL="$2"; shift 2 ;;
        --slot-repair-min-backlog-bytes) SLOT_REPAIR_MIN_BACKLOG_BYTES="$2"; shift 2 ;;
        --tb-err-ema-alpha) TB_ERR_EMA_ALPHA="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --message-layers) MESSAGE_LAYERS="$2"; shift 2 ;;
        --dropout) DROPOUT="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --actor-lr) ACTOR_LR="$2"; shift 2 ;;
        --critic-lr) CRITIC_LR="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --gamma) GAMMA="$2"; shift 2 ;;
        --gae-lambda) GAE_LAMBDA="$2"; shift 2 ;;
        --clip-eps) CLIP_EPS="$2"; shift 2 ;;
        --value-coef) VALUE_COEF="$2"; shift 2 ;;
        --entropy-coef) ENTROPY_COEF="$2"; shift 2 ;;
        --target-kl) TARGET_KL="$2"; shift 2 ;;
        --update-epochs) UPDATE_EPOCHS="$2"; shift 2 ;;
        --minibatch-size) MINIBATCH_SIZE="$2"; shift 2 ;;
        --normalize-state) NORMALIZE_STATE="$2"; shift 2 ;;
        --normalize-reward) NORMALIZE_REWARD="$2"; shift 2 ;;
        --normalize-adv) NORMALIZE_ADV="$2"; shift 2 ;;
        --max-grad-norm) MAX_GRAD_NORM="$2"; shift 2 ;;
        --value-loss) VALUE_LOSS="$2"; shift 2 ;;
        --value-huber-beta) VALUE_HUBER_BETA="$2"; shift 2 ;;
        --seed) TRAIN_SEED="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --print-every-iters) PRINT_EVERY_ITERS="$2"; shift 2 ;;
        --rollout-log-every-steps) ROLLOUT_LOG_EVERY_STEPS="$2"; shift 2 ;;
        --update-log-every-minibatches) UPDATE_LOG_EVERY_MINIBATCHES="$2"; shift 2 ;;
        --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
        --imitation-episodes) IMITATION_EPISODES="$2"; shift 2 ;;
        --imitation-max-steps) IMITATION_MAX_STEPS="$2"; shift 2 ;;
        --imitation-lr) IMITATION_LR="$2"; shift 2 ;;
        --teacher-mode) TEACHER_MODE="$2"; shift 2 ;;
        --teacher-aux-coef) TEACHER_AUX_COEF="$2"; shift 2 ;;
        --teacher-aux-decay-iters) TEACHER_AUX_DECAY_ITERS="$2"; shift 2 ;;
        --reward-mode) REWARD_MODE="$2"; shift 2 ;;
        --best-metric) BEST_METRIC="$2"; shift 2 ;;
        --early-stop-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --early-stop-min-delta) EARLY_STOP_MIN_DELTA="$2"; shift 2 ;;
        --early-stop-warmup-iters) EARLY_STOP_WARMUP_ITERS="$2"; shift 2 ;;
        --goodput-weight) GOODPUT_WEIGHT="$2"; shift 2 ;;
        --tb-err-weight) TB_ERR_WEIGHT="$2"; shift 2 ;;
        --expiry-weight) EXPIRY_WEIGHT="$2"; shift 2 ;;
        --conflict-weight) CONFLICT_WEIGHT="$2"; shift 2 ;;
        --ici-weight) ICI_WEIGHT="$2"; shift 2 ;;
        --urgent-backlog-weight) URGENT_BACKLOG_WEIGHT="$2"; shift 2 ;;
        --urgent-ttl-ms) URGENT_TTL_MS="$2"; shift 2 ;;
        --backlog-debt-weight) BACKLOG_DEBT_WEIGHT="$2"; shift 2 ;;
        --packet-rate-weight) PACKET_RATE_WEIGHT="$2"; shift 2 ;;
        --packet-completion-weight) PACKET_COMPLETION_WEIGHT="$2"; shift 2 ;;
        --packet-rate-scale-mbps) PACKET_RATE_SCALE_MBPS="$2"; shift 2 ;;
        --packet-completion-scale) PACKET_COMPLETION_SCALE="$2"; shift 2 ;;
        --aged-backlog-weight) AGED_BACKLOG_WEIGHT="$2"; shift 2 ;;
        --sim-env) EXTRA_SIM_ENV+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -x "${RUN_SCRIPT}" ]]; then
    echo "Missing run script: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${PPO_SCRIPT}" ]]; then
    echo "Missing online PPO script: ${PPO_SCRIPT}" >&2
    exit 1
fi

EXEC_MODE="$(echo "${EXEC_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" != "both" ]]; then
    echo "--exec-mode must be both for online bridge PPO." >&2
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
if [[ "${CANDIDATE_MODE}" == "all_demand_ues" && "${MAX_PRG_CANDIDATES}" == "0" ]]; then
    MAX_PRG_CANDIDATES="${UE_PER_CELL}"
fi

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
fi
if [[ "${OUT_DIR}" != /* ]]; then
    OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi
if [[ "${SOCKET_PATH}" != /* ]]; then
    SOCKET_PATH="${ROOT_DIR}/${SOCKET_PATH}"
fi
if [[ -n "${INIT_CHECKPOINT}" && "${INIT_CHECKPOINT}" != /* ]]; then
    INIT_CHECKPOINT="${ROOT_DIR}/${INIT_CHECKPOINT}"
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
    "CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1"
)
if [[ "${TEACHER_MODE}" == "pfq_passthrough" || "${IMITATION_EPISODES}" != "0" ]]; then
    SIM_ENV+=("CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1")
fi
SIM_REWARD_MODE="goodput_only"
case "${REWARD_MODE}" in
    stageb_blankaware)
        SIM_REWARD_MODE="goodput_reliability_blankaware"
        ;;
    raw_env)
        SIM_REWARD_MODE="goodput_only"
        ;;
    stageb_v1|*)
        SIM_REWARD_MODE="goodput_only"
        ;;
esac
SIM_ENV+=("CUMAC_ONLINE_REWARD_MODE=${SIM_REWARD_MODE}")
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
echo "[Stage-B HGraph] ue_per_cell=${UE_PER_CELL} total_ue_count=${TOTAL_UE_COUNT}"
echo "[Stage-B HGraph] topology_seed_mode=${TOPOLOGY_SEED_MODE} topology_seed=${TOPOLOGY_SEED} seed_list=${SEED_LIST:-none}"
echo "[Stage-B HGraph] episode_horizon=${EPISODE_HORIZON} rollout_steps=${ROLLOUT_STEPS} iterations=${ITERATIONS}"
echo "[Stage-B HGraph] reward_mode=${REWARD_MODE} init_checkpoint=${INIT_CHECKPOINT:-none} precoding=${PRECODING}"
echo "[Stage-B HGraph] imitation_episodes=${IMITATION_EPISODES} teacher_mode=${TEACHER_MODE} teacher_aux_coef=${TEACHER_AUX_COEF} teacher_aux_decay_iters=${TEACHER_AUX_DECAY_ITERS}"
echo "[Stage-B HGraph] candidate_mode=${CANDIDATE_MODE} ue_prg_topk=${UE_PRG_TOPK} diversity_extra=${UE_PRG_DIVERSITY_EXTRA} max_prg_candidates=${MAX_PRG_CANDIDATES}"
echo "[Stage-B HGraph] slot_repair_max_per_cell=${SLOT_REPAIR_MAX_PER_CELL} slot_repair_min_backlog_bytes=${SLOT_REPAIR_MIN_BACKLOG_BYTES} backlog_debt_weight=${BACKLOG_DEBT_WEIGHT}"
echo "[Stage-B HGraph] packet_rate_weight=${PACKET_RATE_WEIGHT} packet_completion_weight=${PACKET_COMPLETION_WEIGHT} aged_backlog_weight=${AGED_BACKLOG_WEIGHT}"

CMD=(
    "${PYTHON_BIN}" "${PPO_SCRIPT}"
    --sim-bin "${BIN}"
    --sim-args "${SIM_ARGS}"
    --sim-cwd "${SIM_CWD}"
    --socket-path "${SOCKET_PATH}"
    --connect-timeout-s "${CONNECT_TIMEOUT_S}"
    --sim-wait-timeout "${SIM_WAIT_TIMEOUT}"
    --sim-log-mode "${SIM_LOG_MODE}"
    --online-persistent "${ONLINE_PERSISTENT}"
    --episode-horizon "${EPISODE_HORIZON}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --iterations "${ITERATIONS}"
    --topology-seed "${TOPOLOGY_SEED}"
    --topology-seed-mode "${TOPOLOGY_SEED_MODE}"
    --seed-list "${SEED_LIST}"
    --out-dir "${OUT_DIR}"
    --ue-prg-topk "${UE_PRG_TOPK}"
    --ue-prg-diversity-extra "${UE_PRG_DIVERSITY_EXTRA}"
    --candidate-mode "${CANDIDATE_MODE}"
    --max-prg-candidates "${MAX_PRG_CANDIDATES}"
    --slot-repair-max-per-cell "${SLOT_REPAIR_MAX_PER_CELL}"
    --slot-repair-min-backlog-bytes "${SLOT_REPAIR_MIN_BACKLOG_BYTES}"
    --tb-err-ema-alpha "${TB_ERR_EMA_ALPHA}"
    --hidden-dim "${HIDDEN_DIM}"
    --message-layers "${MESSAGE_LAYERS}"
    --dropout "${DROPOUT}"
    --lr "${LR}"
    --actor-lr "${ACTOR_LR}"
    --critic-lr "${CRITIC_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --gamma "${GAMMA}"
    --gae-lambda "${GAE_LAMBDA}"
    --clip-eps "${CLIP_EPS}"
    --value-coef "${VALUE_COEF}"
    --entropy-coef "${ENTROPY_COEF}"
    --target-kl "${TARGET_KL}"
    --update-epochs "${UPDATE_EPOCHS}"
    --minibatch-size "${MINIBATCH_SIZE}"
    --normalize-state "${NORMALIZE_STATE}"
    --normalize-reward "${NORMALIZE_REWARD}"
    --normalize-adv "${NORMALIZE_ADV}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --value-loss "${VALUE_LOSS}"
    --value-huber-beta "${VALUE_HUBER_BETA}"
    --seed "${TRAIN_SEED}"
    --device "${DEVICE}"
    --print-every-iters "${PRINT_EVERY_ITERS}"
    --rollout-log-every-steps "${ROLLOUT_LOG_EVERY_STEPS}"
    --update-log-every-minibatches "${UPDATE_LOG_EVERY_MINIBATCHES}"
    --init-checkpoint "${INIT_CHECKPOINT}"
    --imitation-episodes "${IMITATION_EPISODES}"
    --imitation-max-steps "${IMITATION_MAX_STEPS}"
    --imitation-lr "${IMITATION_LR}"
    --teacher-mode "${TEACHER_MODE}"
    --teacher-aux-coef "${TEACHER_AUX_COEF}"
    --teacher-aux-decay-iters "${TEACHER_AUX_DECAY_ITERS}"
    --reward-mode "${REWARD_MODE}"
    --best-metric "${BEST_METRIC}"
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}"
    --early-stop-warmup-iters "${EARLY_STOP_WARMUP_ITERS}"
    --goodput-weight "${GOODPUT_WEIGHT}"
    --tb-err-weight "${TB_ERR_WEIGHT}"
    --expiry-weight "${EXPIRY_WEIGHT}"
    --conflict-weight "${CONFLICT_WEIGHT}"
    --ici-weight "${ICI_WEIGHT}"
    --urgent-backlog-weight "${URGENT_BACKLOG_WEIGHT}"
    --urgent-ttl-ms "${URGENT_TTL_MS}"
    --backlog-debt-weight "${BACKLOG_DEBT_WEIGHT}"
    --packet-rate-weight "${PACKET_RATE_WEIGHT}"
    --packet-completion-weight "${PACKET_COMPLETION_WEIGHT}"
    --packet-rate-scale-mbps "${PACKET_RATE_SCALE_MBPS}"
    --packet-completion-scale "${PACKET_COMPLETION_SCALE}"
    --aged-backlog-weight "${AGED_BACKLOG_WEIGHT}"
)
for kv in "${SIM_ENV[@]}"; do
    CMD+=(--sim-env "${kv}")
done

echo "[Stage-B HGraph] running online PPO fine-tuning"
"${CMD[@]}"
