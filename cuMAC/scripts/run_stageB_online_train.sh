#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
TRAIN_SCRIPT="${ROOT_DIR}/training/gnnrl/ppo_online_train.py"

ARCH="$(uname -m)"
BUILD_DIR="${ROOT_DIR}/build.${ARCH}"
BUILD_METHOD="phase4"
GPU_ID=0
TTI_COUNT=2000
DL_UL="dl"
FADING_MODE=0
TOPOLOGY_SCENARIO="7cell"
UE_PER_CELL=8
TOPOLOGY_SEED=0
UE_PLACEMENT="uniform"
UE_RADIUS_SPLITS="0.33,0.66"
UE_STRATA_COUNTS=""
UE_VORONOI_CLIP=1
BS_TX_PATTERN="omni"
TRAFFIC_PERCENT=100
PACKET_SIZE_BYTES=5000
TRAFFIC_ARRIVAL_RATE=0.2
CDL_PROFILES="NA"
CDL_DELAY_SPREADS_NS="0"
ALLOW_PROFILE_D=0
CUSTOM_UE_PRG=0
BASELINE_SCHEDULER="pf"
COMPACT_TTI_LOG=1
PROGRESS_TTI_INTERVAL=100
KPI_TTI_LOG_INTERVAL=100
COMPARE_TTI_INTERVAL=0
REPLAY_DUMP=0
REPLAY_DIR=""
TAG=""
RESTORE_PARAMS=0
EXEC_MODE="both"

OUT_DIR="${ROOT_DIR}/training/gnnrl/checkpoints/m3_online_ppo"
INIT_POLICY_CHECKPOINT=""
ITERS=20
ROLLOUT_STEPS=256
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
HIDDEN_DIM=128
NUM_CELL_MSG_LAYERS=2
ACTION_MODE="prg_only_type0"
SEED=42
DEVICE="auto"
SOCKET_PATH="/tmp/cumac_stageb_online.sock"
CONNECT_TIMEOUT_S=20.0
SIM_WAIT_TIMEOUT=10.0
ONLINE_PERSISTENT=1
EPISODE_HORIZON=400

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
  --topology-seed <n>         Fixed topology seed (default: ${TOPOLOGY_SEED})
  --ue-placement <m>          uniform | stratified (default: ${UE_PLACEMENT})
  --ue-radius-splits <a,b>    Stratified radius split ratios (default: ${UE_RADIUS_SPLITS})
  --ue-strata-counts <a,b,c>  Stratified UE counts per cell
  --ue-voronoi-clip <0|1>     Voronoi clipping flag (default: ${UE_VORONOI_CLIP})
  --bs-tx-pattern <m>         omni | sector (default: ${BS_TX_PATTERN})
  --traffic-percent <p>       UE traffic percentage (default: ${TRAFFIC_PERCENT})
  --packet-size-bytes <b>     Packet size in bytes (default: ${PACKET_SIZE_BYTES})
  --traffic-arrival-rate <r>  Traffic arrival rate in pkt/TTI (default: ${TRAFFIC_ARRIVAL_RATE})
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
  --hidden-dim <n>            Policy hidden dim (default: ${HIDDEN_DIM})
  --num-cell-msg-layers <n>   Number of cell message layers (default: ${NUM_CELL_MSG_LAYERS})
  --action-mode <m>           joint | prg_only_type0 (default: ${ACTION_MODE})
  --seed <n>                  Random seed (default: ${SEED})
  --device <m>                auto | cpu | cuda (default: ${DEVICE})
  --socket-path <path>        Online socket path prefix (default: ${SOCKET_PATH})
  --connect-timeout-s <sec>   Socket connect timeout (default: ${CONNECT_TIMEOUT_S})
  --sim-wait-timeout <sec>    Simulator shutdown wait timeout (default: ${SIM_WAIT_TIMEOUT})
  --online-persistent <0|1>   Persistent bridge mode (default: ${ONLINE_PERSISTENT})
  --episode-horizon <n>       Episode horizon for reset requests (default: ${EPISODE_HORIZON})
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
        --ue-per-cell) UE_PER_CELL="$2"; shift 2 ;;
        --topology-seed) TOPOLOGY_SEED="$2"; shift 2 ;;
        --ue-placement) UE_PLACEMENT="$2"; shift 2 ;;
        --ue-radius-splits) UE_RADIUS_SPLITS="$2"; shift 2 ;;
        --ue-strata-counts) UE_STRATA_COUNTS="$2"; shift 2 ;;
        --ue-voronoi-clip) UE_VORONOI_CLIP="$2"; shift 2 ;;
        --bs-tx-pattern) BS_TX_PATTERN="$2"; shift 2 ;;
        --traffic-percent) TRAFFIC_PERCENT="$2"; shift 2 ;;
        --packet-size-bytes|--traffic-rate) PACKET_SIZE_BYTES="$2"; shift 2 ;;
        --traffic-arrival-rate) TRAFFIC_ARRIVAL_RATE="$2"; shift 2 ;;
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
        --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
        --num-cell-msg-layers) NUM_CELL_MSG_LAYERS="$2"; shift 2 ;;
        --action-mode) ACTION_MODE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --socket-path) SOCKET_PATH="$2"; shift 2 ;;
        --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2 ;;
        --sim-wait-timeout) SIM_WAIT_TIMEOUT="$2"; shift 2 ;;
        --online-persistent) ONLINE_PERSISTENT="$2"; shift 2 ;;
        --episode-horizon) EPISODE_HORIZON="$2"; shift 2 ;;
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

BASELINE_SCHEDULER="$(echo "${BASELINE_SCHEDULER}" | tr '[:upper:]' '[:lower:]')"
if [[ "${BASELINE_SCHEDULER}" != "pf" && "${BASELINE_SCHEDULER}" != "pfq" && "${BASELINE_SCHEDULER}" != "rr" ]]; then
    echo "--baseline-scheduler must be pf, pfq, or rr" >&2
    exit 1
fi

ACTION_MODE="$(echo "${ACTION_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${ACTION_MODE}" != "joint" && "${ACTION_MODE}" != "prg_only_type0" ]]; then
    echo "--action-mode must be joint or prg_only_type0" >&2
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
    "CUMAC_GNNRL_ACTION_MODE=${ACTION_MODE}"
    "CUMAC_RL_REPLAY_DUMP=${REPLAY_DUMP}"
    "CUMAC_RL_REPLAY_DIR=${REPLAY_DIR}"
    "CUMAC_EXEC_MODE=both"
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

echo "[Stage-B Online] sim_bin=${BIN}"
echo "[Stage-B Online] sim_cwd=${SIM_CWD}"
echo "[Stage-B Online] sim_args=${SIM_ARGS}"
echo "[Stage-B Online] action_mode=${ACTION_MODE}"
echo "[Stage-B Online] scenario_frozen=topology_scenario:${TOPOLOGY_SCENARIO} tti:${TTI_COUNT} topology_seed:${TOPOLOGY_SEED} ue_placement:${UE_PLACEMENT} voronoi_clip:${UE_VORONOI_CLIP} bs_tx_pattern:${BS_TX_PATTERN} packet_size_bytes:${PACKET_SIZE_BYTES} traffic_arrival_rate:${TRAFFIC_ARRIVAL_RATE} exec_mode:both"

TRAIN_CMD=(
    python3 "${TRAIN_SCRIPT}"
    --sim-bin "${BIN}"
    --sim-args "${SIM_ARGS}"
    --sim-cwd "${SIM_CWD}"
    --socket-path "${SOCKET_PATH}"
    --connect-timeout-s "${CONNECT_TIMEOUT_S}"
    --sim-wait-timeout "${SIM_WAIT_TIMEOUT}"
    --online-persistent "${ONLINE_PERSISTENT}"
    --episode-horizon "${EPISODE_HORIZON}"
    --iters "${ITERS}"
    --rollout-steps "${ROLLOUT_STEPS}"
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
    --hidden-dim "${HIDDEN_DIM}"
    --num-cell-msg-layers "${NUM_CELL_MSG_LAYERS}"
    --action-mode "${ACTION_MODE}"
    --out-dir "${OUT_DIR}"
    --seed "${SEED}"
    --device "${DEVICE}"
)
if [[ -n "${INIT_POLICY_CHECKPOINT}" ]]; then
    TRAIN_CMD+=(--init-policy-checkpoint "${INIT_POLICY_CHECKPOINT}")
fi
if [[ ${#SIM_ENV[@]} -gt 0 ]]; then
    TRAIN_CMD+=(--sim-env "${SIM_ENV[@]}")
fi

"${TRAIN_CMD[@]}"
