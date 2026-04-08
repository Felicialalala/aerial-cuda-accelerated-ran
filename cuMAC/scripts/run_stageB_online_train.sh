#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
TRAIN_SCRIPT="${ROOT_DIR}/training/gnnrl/ppo_online_train.py"
EVAL_SCRIPT="${ROOT_DIR}/training/gnnrl/eval_candidate_checkpoints.py"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4"
GPU_ID=0
TTI_COUNT=2000
DL_UL="dl"
FADING_MODE=0
TOPOLOGY_SCENARIO="7cell"
UE_PER_CELL=8
TOTAL_UE_COUNT=""
TOPOLOGY_SEED=0
UE_PLACEMENT="uniform"
UE_RADIUS_SPLITS="0.33,0.66"
UE_STRATA_COUNTS=""
UE_VORONOI_CLIP=1
BS_TX_PATTERN="omni"
TRAFFIC_PERCENT=100
PACKET_SIZE_BYTES=5000
TRAFFIC_ARRIVAL_RATE=0.2
PACKET_TTL_TTI=0
PACKET_TTL_MS=0
CDL_PROFILES="NA"
CDL_DELAY_SPREADS_NS="0"
ALLOW_PROFILE_D=0
PRBS_PER_GROUP=16
CUSTOM_UE_PRG=0
BASELINE_SCHEDULER="pf"
COMPACT_TTI_LOG=1
PROGRESS_TTI_INTERVAL=0
KPI_TTI_LOG_INTERVAL=100
COMPARE_TTI_INTERVAL=0
REPLAY_DUMP=0
REPLAY_DIR=""
TAG=""
RESTORE_PARAMS=0
EXEC_MODE="both"

OUT_DIR="${ROOT_DIR}/training/gnnrl/checkpoints/m3_online_ppo"
INIT_POLICY_CHECKPOINT=""
ITERS=500
ROLLOUT_STEPS=1024
PPO_EPOCHS=6
MINIBATCH_SIZE=128
GAMMA=0.99
GAE_LAMBDA=0.95
CLIP_EPS=0.2
ENTROPY_COEF=0.01
VALUE_COEF=0.5
ACTOR_LR=1e-4
CRITIC_LR=3e-4
TARGET_KL=0.05
NORMALIZE_STATE=1
NORMALIZE_REWARD=1
NORMALIZE_ADV=1
MAX_GRAD_NORM=1.0
VALUE_LOSS="huber"
VALUE_HUBER_BETA=10.0
PLOT_AFTER_TRAIN=1
PLOT_SMOOTH_WINDOW=5
CURVE_EVERY_EPISODES=0
HIDDEN_DIM=128
NUM_CELL_MSG_LAYERS=2
ACTION_MODE="joint"
SEED=42
SEED_LIST=""
TOPOLOGY_SEED_MODE="auto"
DEVICE="auto"
SOCKET_PATH="/tmp/cumac_stageb_online.sock"
CONNECT_TIMEOUT_S=20.0
SIM_WAIT_TIMEOUT=10.0
SIM_LOG_MODE="file"
ONLINE_PERSISTENT=1
EPISODE_HORIZON=1024
EPISODE_BOUNDARY_MODE="auto"
REWARD_MODE="goodput_only"
CANDIDATE_SAVE_EVERY_ITERS=0
CANDIDATE_SAVE_START_ITER=1
AUTO_MAIN_EVAL=0
EVAL_BUILD_METHOD="skip"
EVAL_DECODE_MODE="sample"
EVAL_SAMPLE_SEEDS="42"
EVAL_CANDIDATE_LIMIT=0
EVAL_PROMOTE_BEST=1
EVAL_TAG_PREFIX=""
UE_PER_CELL_EXPLICIT=0
EPISODE_HORIZON_EXPLICIT=0

EXTRA_SIM_ENV=()

usage() {
    cat <<EOF
Run Stage-B online PPO training with the same scenario-style CLI used by run_stageB_main_experiment.sh.

Usage:
  $(basename "$0") [options]

Stage-B scenario options:
  --build-dir <path>          Build directory (default: ${BUILD_DIR})
  --build-method <m>          phase4 | cmake | skip (default: ${BUILD_METHOD})
  --gpu <id>                  GPU device id (default: ${GPU_ID})
  --tti <count>               Number of TTIs compiled into the Stage-B binary (default: ${TTI_COUNT})
  --mode <dl|ul>              Downlink or uplink (default: ${DL_UL})
  --fading-mode <0|1|2|3|4>   Stage-B fading mode (default: ${FADING_MODE})
  --topology-scenario <m>     7cell | 3cell (default: ${TOPOLOGY_SCENARIO})
  --ue-per-cell <n>           Active UE count per cell (default: ${UE_PER_CELL})
  --total-ue-count <n>        Total active UE count across coordinated cells; must divide cell count
  --topology-seed <n>         Base/fixed topology seed (default: ${TOPOLOGY_SEED})
  --ue-placement <m>          uniform | stratified (default: ${UE_PLACEMENT})
  --ue-radius-splits <a,b>    Stratified radius split ratios (default: ${UE_RADIUS_SPLITS})
  --ue-strata-counts <a,b,c>  Stratified UE counts per cell
  --ue-voronoi-clip <0|1>     Voronoi clipping flag (default: ${UE_VORONOI_CLIP})
  --bs-tx-pattern <m>         omni | sector (default: ${BS_TX_PATTERN})
  --traffic-percent <p>       UE traffic percentage (default: ${TRAFFIC_PERCENT})
  --packet-size-bytes <b>     Packet size in bytes (default: ${PACKET_SIZE_BYTES})
  --traffic-arrival-rate <r>  Traffic arrival rate in pkt/TTI (default: ${TRAFFIC_ARRIVAL_RATE})
  --packet-ttl-tti <n>        Packet TTL in TTI, 0 disables expiry (default: ${PACKET_TTL_TTI})
  --packet-ttl-ms <v>         Packet TTL in ms, 0 disables expiry; ignored when ttl-tti > 0 (default: ${PACKET_TTL_MS})
  --prbs-per-group <n>        Number of RBs in one PRG/RBG; PRG count is auto-derived for 272 total PRBs (default: ${PRBS_PER_GROUP})
  --cdl-profiles <list>       Comma-separated CDL profiles, single scenario uses the first item
  --cdl-delay-spreads <list>  Comma-separated delay spreads, single scenario uses the first item
  --allow-profile-d <0|1>     Forwarded to Stage-B build-only validation (default: ${ALLOW_PROFILE_D})
  --custom-ue-prg <0|1>       Forwarded for CLI consistency; online bridge still overrides scheduler action
  --baseline-scheduler <m>    pf | pfq | rr, forwarded to binary CLI for consistency (default: ${BASELINE_SCHEDULER})
  --compact-tti-log <0|1>     Compact TTI logging flag (default: ${COMPACT_TTI_LOG})
  --progress-tti <n>          Progress print interval in TTI (default: ${PROGRESS_TTI_INTERVAL})
  --kpi-tti-log <n>           KPI print interval in TTI (default: ${KPI_TTI_LOG_INTERVAL})
  --compare-tti <n>           Compare interval, usually 0 for online bridge (default: ${COMPARE_TTI_INTERVAL})
  --replay-dump <0|1>         Enable RL replay dump during online run (default: ${REPLAY_DUMP})
  --replay-dir <path>         Replay output dir when replay dump is enabled
  --exec-mode <mode>          Must be both for online bridge (default: ${EXEC_MODE})
  --tag <name>                Optional build tag forwarded to Stage-B
  --restore-params <0|1>      Restore parameters.h on exit (default: ${RESTORE_PARAMS})

Online PPO options:
  --out-dir <path>            PPO output directory (default: ${OUT_DIR})
  --init-policy-checkpoint <path>
  --iters <n>                 PPO iterations (default: ${ITERS})
  --rollout-steps <n>         Rollout steps per iter (default: ${ROLLOUT_STEPS})
  --ppo-epochs <n>            PPO epochs per iter (default: ${PPO_EPOCHS})
  --minibatch-size <n>        Minibatch size (default: ${MINIBATCH_SIZE})
  --gamma <v>                 Discount factor (default: ${GAMMA})
  --gae-lambda <v>            GAE lambda (default: ${GAE_LAMBDA})
  --clip-eps <v>              PPO clip epsilon (default: ${CLIP_EPS})
  --entropy-coef <v>          Entropy coefficient (default: ${ENTROPY_COEF})
  --value-coef <v>            Value coefficient (default: ${VALUE_COEF})
  --actor-lr <v>              Actor learning rate (default: ${ACTOR_LR})
  --critic-lr <v>             Critic learning rate (default: ${CRITIC_LR})
  --target-kl <v>             Target KL (default: ${TARGET_KL})
  --normalize-state <0|1>     State normalization (default: ${NORMALIZE_STATE})
  --normalize-reward <0|1>    Reward normalization (default: ${NORMALIZE_REWARD})
  --normalize-adv <0|1>       Advantage normalization (default: ${NORMALIZE_ADV})
  --max-grad-norm <v>         Gradient clipping (default: ${MAX_GRAD_NORM})
  --value-loss <m>            mse | huber (default: ${VALUE_LOSS})
  --value-huber-beta <v>      Huber beta (default: ${VALUE_HUBER_BETA})
  --plot-after-train <0|1>    Export training curves (default: ${PLOT_AFTER_TRAIN})
  --plot-smooth-window <n>    Plot moving-average window (default: ${PLOT_SMOOTH_WINDOW})
  --curve-every-episodes <n>  Write latest curves and a snapshot every N completed episodes (default: ${CURVE_EVERY_EPISODES})
  --hidden-dim <n>            Policy hidden dim (default: ${HIDDEN_DIM})
  --num-cell-msg-layers <n>   Number of cell message layers (default: ${NUM_CELL_MSG_LAYERS})
  --action-mode <m>           joint | prg_only_type0 (default: ${ACTION_MODE})
  --seed <n>                  Trainer random seed for PPO sampling/optimization (default: ${SEED})
  --seed-list <list>          Comma-separated topology seeds for explicit list-cycle mode
  --topology-seed-mode <m>    auto | fixed | sequential | list_cycle (default: ${TOPOLOGY_SEED_MODE})
  --device <m>                auto | cpu | cuda (default: ${DEVICE})
  --socket-path <path>        Online socket path prefix (default: ${SOCKET_PATH})
  --connect-timeout-s <sec>   Socket connect timeout (default: ${CONNECT_TIMEOUT_S})
  --sim-wait-timeout <sec>    Simulator shutdown wait timeout (default: ${SIM_WAIT_TIMEOUT})
  --sim-log-mode <m>          file | inherit (default: ${SIM_LOG_MODE})
  --online-persistent <0|1>   Persistent bridge mode (default: ${ONLINE_PERSISTENT})
  --episode-horizon <n>       Episode horizon for reset requests (default: ${EPISODE_HORIZON})
  --episode-boundary-mode <m> auto | bridge | trainer (default: ${EPISODE_BOUNDARY_MODE})
  --reward-mode <m>           legacy | goodput_only | goodput_soft_queue | goodput_reliability | goodput_reliability_reuseaware | goodput_reliability_blankaware (default: ${REWARD_MODE})
  --candidate-save-every-iters <n>
                              Save periodic candidate checkpoints every N iters using the pre-update rollout actor (default: ${CANDIDATE_SAVE_EVERY_ITERS})
  --candidate-save-start-iter <n>
                              First iter eligible for periodic candidate checkpoint save (default: ${CANDIDATE_SAVE_START_ITER})
  --auto-main-eval <0|1>      After training, export/evaluate candidates with main experiment and promote deployment best (default: ${AUTO_MAIN_EVAL})
  --eval-build-method <m>     skip | cmake for candidate main-experiment eval (default: ${EVAL_BUILD_METHOD})
  --eval-decode-mode <m>      sample | argmax for candidate eval (default: ${EVAL_DECODE_MODE})
  --eval-sample-seeds <list>  Comma-separated sample seeds for candidate eval in sample mode (default: ${EVAL_SAMPLE_SEEDS})
  --eval-candidate-limit <n>  Evaluate top-N training candidates; 0 means all discovered candidates (default: ${EVAL_CANDIDATE_LIMIT})
  --eval-promote-best <0|1>   Copy deployment-best checkpoint/onnx to *_best_eval artifacts (default: ${EVAL_PROMOTE_BEST})
  --eval-tag-prefix <text>    Optional tag prefix for candidate eval runs
  --sim-env <KEY=VALUE>       Extra simulator env var, may be repeated
  -h, --help                  Show this help
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
        --ue-per-cell) UE_PER_CELL="$2"; UE_PER_CELL_EXPLICIT=1; shift 2 ;;
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
        --replay-dump) REPLAY_DUMP="$2"; shift 2 ;;
        --replay-dir) REPLAY_DIR="$2"; shift 2 ;;
        --exec-mode) EXEC_MODE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --restore-params) RESTORE_PARAMS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --init-policy-checkpoint) INIT_POLICY_CHECKPOINT="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
        --ppo-epochs) PPO_EPOCHS="$2"; shift 2 ;;
        --minibatch-size) MINIBATCH_SIZE="$2"; shift 2 ;;
        --gamma) GAMMA="$2"; shift 2 ;;
        --gae-lambda) GAE_LAMBDA="$2"; shift 2 ;;
        --clip-eps) CLIP_EPS="$2"; shift 2 ;;
        --entropy-coef) ENTROPY_COEF="$2"; shift 2 ;;
        --value-coef) VALUE_COEF="$2"; shift 2 ;;
        --actor-lr) ACTOR_LR="$2"; shift 2 ;;
        --critic-lr) CRITIC_LR="$2"; shift 2 ;;
        --target-kl) TARGET_KL="$2"; shift 2 ;;
        --normalize-state) NORMALIZE_STATE="$2"; shift 2 ;;
        --normalize-reward) NORMALIZE_REWARD="$2"; shift 2 ;;
        --normalize-adv) NORMALIZE_ADV="$2"; shift 2 ;;
        --max-grad-norm) MAX_GRAD_NORM="$2"; shift 2 ;;
        --value-loss) VALUE_LOSS="$2"; shift 2 ;;
        --value-huber-beta) VALUE_HUBER_BETA="$2"; shift 2 ;;
        --plot-after-train) PLOT_AFTER_TRAIN="$2"; shift 2 ;;
        --plot-smooth-window) PLOT_SMOOTH_WINDOW="$2"; shift 2 ;;
        --curve-every-episodes) CURVE_EVERY_EPISODES="$2"; shift 2 ;;
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --num-cell-msg-layers) NUM_CELL_MSG_LAYERS="$2"; shift 2 ;;
        --action-mode) ACTION_MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --seed-list) SEED_LIST="$2"; shift 2 ;;
        --topology-seed-mode) TOPOLOGY_SEED_MODE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --socket-path) SOCKET_PATH="$2"; shift 2 ;;
        --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2 ;;
        --sim-wait-timeout) SIM_WAIT_TIMEOUT="$2"; shift 2 ;;
        --sim-log-mode) SIM_LOG_MODE="$2"; shift 2 ;;
        --online-persistent) ONLINE_PERSISTENT="$2"; shift 2 ;;
        --episode-horizon) EPISODE_HORIZON="$2"; EPISODE_HORIZON_EXPLICIT=1; shift 2 ;;
        --episode-boundary-mode) EPISODE_BOUNDARY_MODE="$2"; shift 2 ;;
        --reward-mode) REWARD_MODE="$2"; shift 2 ;;
        --candidate-save-every-iters) CANDIDATE_SAVE_EVERY_ITERS="$2"; shift 2 ;;
        --candidate-save-start-iter) CANDIDATE_SAVE_START_ITER="$2"; shift 2 ;;
        --auto-main-eval) AUTO_MAIN_EVAL="$2"; shift 2 ;;
        --eval-build-method) EVAL_BUILD_METHOD="$2"; shift 2 ;;
        --eval-decode-mode) EVAL_DECODE_MODE="$2"; shift 2 ;;
        --eval-sample-seeds) EVAL_SAMPLE_SEEDS="$2"; shift 2 ;;
        --eval-candidate-limit) EVAL_CANDIDATE_LIMIT="$2"; shift 2 ;;
        --eval-promote-best) EVAL_PROMOTE_BEST="$2"; shift 2 ;;
        --eval-tag-prefix) EVAL_TAG_PREFIX="$2"; shift 2 ;;
        --sim-env) EXTRA_SIM_ENV+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ! -x "${RUN_SCRIPT}" ]]; then
    echo "Missing run script: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Missing training script: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

EXEC_MODE="$(echo "${EXEC_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${EXEC_MODE}" != "both" ]]; then
    echo "--exec-mode must be both for online bridge training." >&2
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
if ! [[ "${UE_PER_CELL}" =~ ^[0-9]+$ ]] || [[ "${UE_PER_CELL}" -lt 1 ]]; then
    echo "--ue-per-cell must be a positive integer" >&2
    exit 1
fi
if [[ -n "${TOTAL_UE_COUNT}" ]]; then
    if ! [[ "${TOTAL_UE_COUNT}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_UE_COUNT}" -lt 1 ]]; then
        echo "--total-ue-count must be a positive integer" >&2
        exit 1
    fi
    if (( TOTAL_UE_COUNT % TOPOLOGY_NUM_CELLS != 0 )); then
        echo "--total-ue-count must be divisible by the coordinated cell count (${TOPOLOGY_NUM_CELLS})" >&2
        exit 1
    fi
    DERIVED_UE_PER_CELL=$((TOTAL_UE_COUNT / TOPOLOGY_NUM_CELLS))
    if [[ "${UE_PER_CELL_EXPLICIT}" == "1" && "${UE_PER_CELL}" -ne "${DERIVED_UE_PER_CELL}" ]]; then
        echo "--ue-per-cell (${UE_PER_CELL}) conflicts with --total-ue-count (${TOTAL_UE_COUNT}) for topology ${TOPOLOGY_SCENARIO}" >&2
        exit 1
    fi
    UE_PER_CELL="${DERIVED_UE_PER_CELL}"
fi
TOTAL_UE_COUNT=$((UE_PER_CELL * TOPOLOGY_NUM_CELLS))

BASELINE_SCHEDULER="$(echo "${BASELINE_SCHEDULER}" | tr '[:upper:]' '[:lower:]')"
if [[ "${BASELINE_SCHEDULER}" != "pf" && "${BASELINE_SCHEDULER}" != "pfq" && "${BASELINE_SCHEDULER}" != "rr" ]]; then
    echo "--baseline-scheduler must be pf, pfq, or rr" >&2
    exit 1
fi

REWARD_MODE="$(echo "${REWARD_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${REWARD_MODE}" != "legacy" && "${REWARD_MODE}" != "goodput_only" && "${REWARD_MODE}" != "goodput_soft_queue" && "${REWARD_MODE}" != "goodput_reliability" && "${REWARD_MODE}" != "goodput_reliability_reuseaware" && "${REWARD_MODE}" != "goodput_reliability_blankaware" ]]; then
    echo "--reward-mode must be legacy, goodput_only, goodput_soft_queue, goodput_reliability, goodput_reliability_reuseaware, or goodput_reliability_blankaware" >&2
    exit 1
fi
SIM_LOG_MODE="$(echo "${SIM_LOG_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${SIM_LOG_MODE}" != "file" && "${SIM_LOG_MODE}" != "inherit" ]]; then
    echo "--sim-log-mode must be file or inherit" >&2
    exit 1
fi
if ! [[ "${PRBS_PER_GROUP}" =~ ^[0-9]+$ ]] || [[ "${PRBS_PER_GROUP}" -lt 1 ]]; then
    echo "--prbs-per-group must be a positive integer" >&2
    exit 1
fi
if (( 272 % PRBS_PER_GROUP != 0 )); then
    echo "--prbs-per-group must divide 272 total PRBs for the current Stage-B carrier configuration" >&2
    exit 1
fi
PRG_COUNT=$((272 / PRBS_PER_GROUP))

if ! [[ "${ITERS}" =~ ^[0-9]+$ ]] || [[ "${ITERS}" -lt 1 ]]; then
    echo "--iters must be a positive integer" >&2
    exit 1
fi
if ! [[ "${ROLLOUT_STEPS}" =~ ^[0-9]+$ ]] || [[ "${ROLLOUT_STEPS}" -lt 1 ]]; then
    echo "--rollout-steps must be a positive integer" >&2
    exit 1
fi
if [[ "${EPISODE_HORIZON_EXPLICIT}" != "1" ]]; then
    EPISODE_HORIZON="${ROLLOUT_STEPS}"
fi
if ! [[ "${EPISODE_HORIZON}" =~ ^[0-9]+$ ]] || [[ "${EPISODE_HORIZON}" -lt 1 ]]; then
    echo "--episode-horizon must be a positive integer" >&2
    exit 1
fi
if [[ "${EPISODE_HORIZON}" -ne "${ROLLOUT_STEPS}" ]]; then
    echo "--episode-horizon (${EPISODE_HORIZON}) must equal --rollout-steps (${ROLLOUT_STEPS}) in the aligned online PPO setup" >&2
    exit 1
fi

ACTION_MODE="$(echo "${ACTION_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${ACTION_MODE}" != "joint" && "${ACTION_MODE}" != "prg_only_type0" ]]; then
    echo "--action-mode must be joint or prg_only_type0" >&2
    exit 1
fi

TOPOLOGY_SEED_MODE="$(echo "${TOPOLOGY_SEED_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${TOPOLOGY_SEED_MODE}" != "auto" && "${TOPOLOGY_SEED_MODE}" != "fixed" && "${TOPOLOGY_SEED_MODE}" != "sequential" && "${TOPOLOGY_SEED_MODE}" != "list_cycle" ]]; then
    echo "--topology-seed-mode must be auto, fixed, sequential, or list_cycle" >&2
    exit 1
fi

BS_TX_PATTERN="$(echo "${BS_TX_PATTERN}" | tr '[:upper:]' '[:lower:]')"
if [[ "${BS_TX_PATTERN}" != "omni" && "${BS_TX_PATTERN}" != "sector" ]]; then
    echo "--bs-tx-pattern must be omni or sector" >&2
    exit 1
fi

if [[ "${BUILD_DIR}" != /* ]]; then
    BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
fi
if [[ "${OUT_DIR}" != /* ]]; then
    OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi
if [[ -n "${INIT_POLICY_CHECKPOINT}" && "${INIT_POLICY_CHECKPOINT}" != /* ]]; then
    INIT_POLICY_CHECKPOINT="${ROOT_DIR}/${INIT_POLICY_CHECKPOINT}"
fi
if [[ -n "${REPLAY_DIR}" && "${REPLAY_DIR}" != /* ]]; then
    REPLAY_DIR="${ROOT_DIR}/${REPLAY_DIR}"
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
    --gnnrl-action-mode "${ACTION_MODE}"
    --compact-tti-log "${COMPACT_TTI_LOG}"
    --progress-tti "${PROGRESS_TTI_INTERVAL}"
    --kpi-tti-log "${KPI_TTI_LOG_INTERVAL}"
    --compare-tti "${COMPARE_TTI_INTERVAL}"
    --replay-dump "${REPLAY_DUMP}"
    --replay-dir "${REPLAY_DIR}"
    --online-bridge 1
    --online-socket "${SOCKET_PATH}"
    --exec-mode both
    --restore-params "${RESTORE_PARAMS}"
)
if [[ -n "${TAG}" ]]; then
    BUILD_ARGS+=(--tag "${TAG}")
fi

echo "[Stage-B Online] preparing binary with run_stageB_main_experiment.sh --build-only 1"
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

profile="${CDL_PROFILES%%,*}"
profile="$(echo "${profile}" | tr '[:lower:]' '[:upper:]' | xargs)"
delay="${CDL_DELAY_SPREADS_NS%%,*}"
delay="$(echo "${delay}" | xargs)"
if [[ "${FADING_MODE}" == "0" || "${FADING_MODE}" == "1" || "${FADING_MODE}" == "2" ]]; then
    profile="NA"
    delay="0"
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
    "CUMAC_GNNRL_ACTION_MODE=${ACTION_MODE}"
    "CUMAC_RL_REPLAY_DUMP=${REPLAY_DUMP}"
    "CUMAC_RL_REPLAY_DIR=${REPLAY_DIR}"
    "CUMAC_EXEC_MODE=both"
    "CUMAC_ONLINE_REWARD_MODE=${REWARD_MODE}"
)
if [[ "${profile}" != "NA" ]]; then
    SIM_ENV+=(
        "CUMAC_CDL_PROFILE=${profile}"
        "CUMAC_CDL_DELAY_SPREAD_NS=${delay}"
    )
fi
for kv in "${EXTRA_SIM_ENV[@]}"; do
    SIM_ENV+=("${kv}")
done

SIM_ARGS="-d ${DL_IND} -b ${BASELINE_IND} -f ${FADING_MODE} -x ${CUSTOM_UE_PRG} -g ${TRAFFIC_PERCENT} -r ${PACKET_SIZE_BYTES}"

RESOLVED_TOPOLOGY_SEED_MODE="${TOPOLOGY_SEED_MODE}"
if [[ "${RESOLVED_TOPOLOGY_SEED_MODE}" == "auto" ]]; then
    if [[ -n "${SEED_LIST}" ]]; then
        RESOLVED_TOPOLOGY_SEED_MODE="list_cycle"
    else
        RESOLVED_TOPOLOGY_SEED_MODE="sequential"
    fi
fi

echo "[Stage-B Online] sim_bin=${BIN}"
echo "[Stage-B Online] sim_cwd=${SIM_CWD}"
echo "[Stage-B Online] sim_args=${SIM_ARGS}"
echo "[Stage-B Online] action_mode=${ACTION_MODE}"
echo "[Stage-B Online] trainer_seed=${SEED}"
echo "[Stage-B Online] reward_mode=${REWARD_MODE}"
echo "[Stage-B Online] sim_log_mode=${SIM_LOG_MODE}"
echo "[Stage-B Online] prbs_per_group=${PRBS_PER_GROUP} prg_count=${PRG_COUNT}"
echo "[Stage-B Online] ue_per_cell=${UE_PER_CELL} total_ue_count=${TOTAL_UE_COUNT}"
echo "[Stage-B Online] rollout_steps=${ROLLOUT_STEPS} episode_horizon=${EPISODE_HORIZON} iters=${ITERS} total_train_tti=$((ITERS * ROLLOUT_STEPS))"
echo "[Stage-B Online] packet_ttl_tti=${PACKET_TTL_TTI} packet_ttl_ms=${PACKET_TTL_MS}"
echo "[Stage-B Online] candidate_save_every_iters=${CANDIDATE_SAVE_EVERY_ITERS} candidate_save_start_iter=${CANDIDATE_SAVE_START_ITER} auto_main_eval=${AUTO_MAIN_EVAL}"
if [[ -n "${SEED_LIST}" ]]; then
    echo "[Stage-B Online] topology_seed_schedule=${RESOLVED_TOPOLOGY_SEED_MODE} seed_list=${SEED_LIST}"
else
    echo "[Stage-B Online] topology_seed_schedule=${RESOLVED_TOPOLOGY_SEED_MODE} start_or_fixed_seed=${TOPOLOGY_SEED}"
fi
echo "[Stage-B Online] episode_boundary_mode=${EPISODE_BOUNDARY_MODE} curve_every_episodes=${CURVE_EVERY_EPISODES}"
echo "[Stage-B Online] scenario_frozen=topology_scenario:${TOPOLOGY_SCENARIO} tti:${TTI_COUNT} ue_per_cell:${UE_PER_CELL} total_ue_count:${TOTAL_UE_COUNT} topology_seed:${TOPOLOGY_SEED} ue_placement:${UE_PLACEMENT} voronoi_clip:${UE_VORONOI_CLIP} bs_tx_pattern:${BS_TX_PATTERN} prbs_per_group:${PRBS_PER_GROUP} prg_count:${PRG_COUNT} packet_size_bytes:${PACKET_SIZE_BYTES} traffic_arrival_rate:${TRAFFIC_ARRIVAL_RATE} packet_ttl_tti:${PACKET_TTL_TTI} packet_ttl_ms:${PACKET_TTL_MS} exec_mode:both"

TRAIN_CMD=(
    python3 "${TRAIN_SCRIPT}"
    --sim-bin "${BIN}"
    --sim-args "${SIM_ARGS}"
    --sim-cwd "${SIM_CWD}"
    --socket-path "${SOCKET_PATH}"
    --connect-timeout-s "${CONNECT_TIMEOUT_S}"
    --sim-wait-timeout "${SIM_WAIT_TIMEOUT}"
    --sim-log-mode "${SIM_LOG_MODE}"
    --online-persistent "${ONLINE_PERSISTENT}"
    --episode-horizon "${EPISODE_HORIZON}"
    --iters "${ITERS}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --candidate-save-every-iters "${CANDIDATE_SAVE_EVERY_ITERS}"
    --candidate-save-start-iter "${CANDIDATE_SAVE_START_ITER}"
    --ppo-epochs "${PPO_EPOCHS}"
    --minibatch-size "${MINIBATCH_SIZE}"
    --gamma "${GAMMA}"
    --gae-lambda "${GAE_LAMBDA}"
    --clip-eps "${CLIP_EPS}"
    --entropy-coef "${ENTROPY_COEF}"
    --value-coef "${VALUE_COEF}"
    --actor-lr "${ACTOR_LR}"
    --critic-lr "${CRITIC_LR}"
    --target-kl "${TARGET_KL}"
    --normalize-state "${NORMALIZE_STATE}"
    --normalize-reward "${NORMALIZE_REWARD}"
    --normalize-adv "${NORMALIZE_ADV}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --value-loss "${VALUE_LOSS}"
    --value-huber-beta "${VALUE_HUBER_BETA}"
    --plot-after-train "${PLOT_AFTER_TRAIN}"
    --plot-smooth-window "${PLOT_SMOOTH_WINDOW}"
    --curve-every-episodes "${CURVE_EVERY_EPISODES}"
    --hidden-dim "${HIDDEN_DIM}"
    --num-cell-msg-layers "${NUM_CELL_MSG_LAYERS}"
    --action-mode "${ACTION_MODE}"
    --out-dir "${OUT_DIR}"
    --seed "${SEED}"
    --reward-mode "${REWARD_MODE}"
    --topology-seed-mode "${TOPOLOGY_SEED_MODE}"
    --episode-boundary-mode "${EPISODE_BOUNDARY_MODE}"
    --device "${DEVICE}"
)
if [[ -n "${INIT_POLICY_CHECKPOINT}" ]]; then
    TRAIN_CMD+=(--init-policy-checkpoint "${INIT_POLICY_CHECKPOINT}")
fi
if [[ -n "${SEED_LIST}" ]]; then
    TRAIN_CMD+=(--seed-list "${SEED_LIST}")
fi
if [[ ${#SIM_ENV[@]} -gt 0 ]]; then
    TRAIN_CMD+=(--sim-env "${SIM_ENV[@]}")
fi

"${TRAIN_CMD[@]}"

if [[ "${AUTO_MAIN_EVAL}" == "1" ]]; then
    if [[ -z "${EVAL_TAG_PREFIX}" ]]; then
        EVAL_TAG_PREFIX="$(basename "${OUT_DIR}")"
    fi
    echo "[Stage-B Online] auto main eval enabled: decode_mode=${EVAL_DECODE_MODE} sample_seeds=${EVAL_SAMPLE_SEEDS} candidate_limit=${EVAL_CANDIDATE_LIMIT}"
    EVAL_CMD=(
        python3 "${EVAL_SCRIPT}"
        --train-out-dir "${OUT_DIR}"
        --build-method "${EVAL_BUILD_METHOD}"
        --topology-scenario "${TOPOLOGY_SCENARIO}"
        --total-ue-count "${TOTAL_UE_COUNT}"
        --prbs-per-group "${PRBS_PER_GROUP}"
        --baseline-scheduler "${BASELINE_SCHEDULER}"
        --fading-mode "${FADING_MODE}"
        --cdl-profiles "${CDL_PROFILES}"
        --cdl-delay-spreads "${CDL_DELAY_SPREADS_NS}"
        --tti "${TTI_COUNT}"
        --packet-size-bytes "${PACKET_SIZE_BYTES}"
        --traffic-arrival-rate "${TRAFFIC_ARRIVAL_RATE}"
        --packet-ttl-tti "${PACKET_TTL_TTI}"
        --packet-ttl-ms "${PACKET_TTL_MS}"
        --topology-seed "${TOPOLOGY_SEED}"
        --progress-tti "${PROGRESS_TTI_INTERVAL}"
        --kpi-tti-log "${KPI_TTI_LOG_INTERVAL}"
        --compare-tti "${COMPARE_TTI_INTERVAL}"
        --compact-output "${COMPACT_TTI_LOG}"
        --exec-mode "${EXEC_MODE}"
        --gnnrl-action-mode "${ACTION_MODE}"
        --gnnrl-model-decode-mode "${EVAL_DECODE_MODE}"
        --gnnrl-model-sample-seeds "${EVAL_SAMPLE_SEEDS}"
        --gnnrl-model-no-ue-bias "0"
        --gnnrl-model-no-prg-bias "0"
        --gnnrl-model-min-sched-ratio "0"
        --gnnrl-model-min-prg-ratio "0"
        --candidate-limit "${EVAL_CANDIDATE_LIMIT}"
        --promote-best "${EVAL_PROMOTE_BEST}"
        --tag-prefix "${EVAL_TAG_PREFIX}"
    )
    "${EVAL_CMD[@]}"
fi
