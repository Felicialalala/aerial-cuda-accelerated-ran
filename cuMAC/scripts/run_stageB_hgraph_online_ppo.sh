#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
PPO_SCRIPT="${ROOT_DIR}/training/stageb_hgraph/online_ppo.py"
PPO_MODULE="training.stageb_hgraph.online_ppo"

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

OUT_DIR="${ROOT_DIR}/training/stageb_hgraph/runs/online_ppo"
PPO_PRESET=""
ITERATIONS=50
EPISODE_HORIZON=400
ROLLOUT_STEPS=400
ROLLOUT_WARMUP_STEPS=0
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
ACTION_HEAD_MODE="slot"
INCLUDE_PFQ_SCORE_ANCHOR=1
INCLUDE_CANDIDATE_RISK_FEATURES=0
CANDIDATE_BLANK_LOGIT_BIAS=0.0
CANDIDATE_SUCCESS_LOGIT_SCALE=1.0
CANDIDATE_BLANK_SUCCESS_SCALE=1.0
CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
CANDIDATE_SUCCESS_AUX_COEF=0.0
CANDIDATE_SUCCESS_CALIB_COEF=0.0
CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.0
CANDIDATE_SUCCESS_TARGET_EPS=0.05
CANDIDATE_SEQ_STATE_DIM=0
CANDIDATE_SEQ_STATE_COEF=1.0
CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET=-1
CANDIDATE_SEQ_TRAIN_ONLY=0
CANDIDATE_PAIRWISE_RANK_COEF=0.0
CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.70
CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.30
CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.15
CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.0
CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
CANDIDATE_RESCUE_AUX_COEF=0.0
CANDIDATE_RESCUE_POS_WEIGHT=2.0
CANDIDATE_RESCUE_SERVICE_TARGET=0.80
CANDIDATE_RESCUE_TTL_GUARD_MS=60.0
CANDIDATE_RESCUE_MAX_TARGET=1.0
CANDIDATE_DECODE_GOODPUT_COEF=0.0
CANDIDATE_DECODE_GOODPUT_TARGET_BYTES=3000.0
CANDIDATE_DECODE_GOODPUT_MAX_TARGET=1.0
CANDIDATE_DECODE_GOODPUT_POS_WEIGHT=2.0
CANDIDATE_ARGMAX_DECODE_COEF=0.0
CANDIDATE_ARGMAX_DECODE_POS_WEIGHT=2.0
CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="stable"
CANDIDATE_BLANK_TARGET_COEF=0.0
CANDIDATE_BLANK_TARGET_THRESHOLD=0.45
CANDIDATE_BLANK_TARGET_MARGIN=0.5
CANDIDATE_BLANK_TARGET_DEADBAND=0.05
CANDIDATE_BLANK_TARGET_POS_WEIGHT=1.0
CANDIDATE_BLANK_TARGET_FILL_WEIGHT=1.0
CANDIDATE_BLANK_TARGET_POSITIVE_ONLY=0
CANDIDATE_BLANK_TARGET_SERVICE_GUARD=0
CANDIDATE_BLANK_TARGET_SERVICE_MIN_SUCCESS=0.70
CANDIDATE_BLANK_TARGET_SERVICE_MIN_DEBT=0.25
CANDIDATE_BLANK_TARGET_SERVICE_MIN_HOL_MS=1.0
CANDIDATE_BLANK_TARGET_SERVICE_MAX_SCHEDULED_RATIO=0.65
CANDIDATE_BLANK_TARGET_SERVICE_FILL_WEIGHT=0.0
CANDIDATE_BLANK_TARGET_SERVICE_FILL_MARGIN=0.15
CANDIDATE_BLANK_TARGET_SERVICE_FILL_MIN_SUCCESS=0.995
SAFE_FILL_MAX_PER_CELL=0
SAFE_FILL_MIN_QUALITY_DB=-2.0
SAFE_FILL_MAX_QUALITY_GAP_DB=8.0
SAFE_FILL_MAX_PRG_RISK=0.70
SAFE_FILL_MIN_BACKLOG_BYTES=1.0
SAFE_FILL_MAX_UE_TB_ERR=0.45
CANDIDATE_BLANK_BUDGET_MAX_PER_CELL=-1
CANDIDATE_BLANK_BUDGET_MIN_DECISION_LOGIT=-1000000000.0
CANDIDATE_DECODE_DEMAND_SLACK_BYTES=3000.0
CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=-1.0
CANDIDATE_BUDGETED_SELECT_MODE="off"
CANDIDATE_BUDGETED_HARD_CAP=1
CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
CANDIDATE_BUDGETED_OVERCAP_PENALTY=8.0
CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.0
CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.0
CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
CANDIDATE_PACKET_TAIL_FEATURE_ENABLE=0
CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS=2.0
CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS=10.0
CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.0
CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.5
CANDIDATE_BUDGETED_COMPLETION_BONUS=0.0
CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT=0.25
CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE=1.5
CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE=0
CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE=0.55
CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES=6
CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE=1
CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS=4.0
CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS=2.0
CANDIDATE_BUDGETED_TAIL_QUOTA_MAX_SCORE=4.0
CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX=0.70
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
TARGET_KL_MIN_UPDATES=4
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
WRITE_DIAGNOSTICS=1
INIT_CHECKPOINT=""
IMITATION_EPISODES=0
IMITATION_MAX_STEPS=0
IMITATION_LR=0
TEACHER_MODE="pfq_passthrough"
TEACHER_AUX_COEF=0.20
TEACHER_AUX_MIN_COEF=0.0
TEACHER_AUX_DECAY_ITERS=80
TEACHER_SOFT_TARGET=1
TEACHER_SOFT_TEMPERATURE=0.85
TEACHER_SOFT_HIT_BONUS=2.0
TEACHER_SOFT_BLANK_BONUS=0.35
TEACHER_SOFT_BLANK_SAFE_PENALTY=1.25
TEACHER_SOFT_BLANK_NO_SAFE_BONUS=1.0
REWARD_MODE="stageb_v1"
BEST_METRIC="goodput"
EARLY_STOP_PATIENCE=60
EARLY_STOP_MIN_DELTA=1.0
EARLY_STOP_WARMUP_ITERS=50
BEST_MIN_ITER=0
CANDIDATE_CHECKPOINT_INTERVAL=0
CANDIDATE_CHECKPOINT_REQUIRE_GATE=1
BEST_SERVICE_GAP_PENALTY_MBPS=100.0
BEST_UNSERVED_UE_TARGET=4.3
BEST_UNSERVED_PENALTY_MBPS=3.0
BEST_TB_ERR_TARGET=0.115
BEST_TB_ERR_PENALTY_MBPS=500.0
BEST_EXPIRY_PENALTY_MBPS=1200.0
BEST_PACKET_RATE_BONUS_MBPS=12.0
BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=0.0
BEST_PACKET_DELAY_PENALTY_MBPS=60.0
BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS=0.0
BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS=0.0
BEST_UE_P10_RATE_PROXY_TARGET_MBPS=0.0
BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=0.0
BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=0.0
BEST_UE_P10_PACKET_RATE_TARGET_MBPS=0.0
BEST_BACKLOG_P90_PENALTY_MBPS=0.0
BEST_HARD_TB_ERR_MAX=-1.0
BEST_HARD_EXPIRY_RATE_MAX=-1.0
BEST_HARD_SERVICE_P10_MIN=-1.0
BEST_HARD_UNSERVED_UE_MAX=-1.0
BEST_HARD_BACKLOG_P90_MAX=-1.0
BEST_HARD_PACKET_RATE_MIN_MBPS=-1.0
BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=-1.0
BEST_HARD_PACKET_DELAY_MAX_MS=-1.0
BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS=-1.0
BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS=-1.0
BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=-1.0
BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=-1.0
BEST_MULTI_SEED_WINDOW_ITERS=0
BEST_MULTI_SEED_MIN_COVERED_SEEDS=0
GOODPUT_WEIGHT=1.0
TB_ERR_WEIGHT=8.0
EXPIRY_WEIGHT=6.0
CONFLICT_WEIGHT=1.0
ICI_WEIGHT=1.0
URGENT_BACKLOG_WEIGHT=0.5
URGENT_TTL_MS=20.0
BACKLOG_DEBT_WEIGHT=0.0
PACKET_RATE_WEIGHT=0.0
PACKET_PER_PACKET_RATE_WEIGHT=0.0
PACKET_COMPLETION_WEIGHT=0.0
PACKET_DELAY_WEIGHT=0.0
PACKET_RATE_SCALE_MBPS=10.0
PACKET_PER_PACKET_RATE_SCALE_MBPS=24.0
PACKET_COMPLETION_SCALE=36.0
PACKET_DELAY_SCALE_MS=10.0
UE_MACRO_RATE_PROXY_WEIGHT=0.0
UE_P10_RATE_PROXY_GAP_WEIGHT=0.0
UE_RATE_PROXY_SCALE_MBPS=8.0
UE_P10_RATE_PROXY_TARGET_MBPS=0.0
UE_MACRO_PACKET_RATE_WEIGHT=0.0
UE_P10_PACKET_RATE_GAP_WEIGHT=0.0
UE_PACKET_RATE_SCALE_MBPS=8.0
UE_P10_PACKET_RATE_TARGET_MBPS=0.0
AGED_BACKLOG_WEIGHT=0.0
UNSERVED_UE_WEIGHT=0.0
SERVICE_GAP_WEIGHT=0.0
SERVICE_RATE_TARGET=0.0
UTIL_FLOOR_WEIGHT=0.0
UTIL_FLOOR_TARGET=0.0
UTIL_FLOOR_TB_ERR_GUARD=1.0
UTIL_FLOOR_TB_ERR_SOFTNESS=0.0
MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
MISSED_SAFE_OPPORTUNITY_SCALE=51.0
PRG_EFFICIENCY_WEIGHT=-1.0
HARMFUL_PACKING_WEIGHT=-1.0
TB_ERR_TARGET=-1.0
TB_ERR_OVER_TARGET_WEIGHT=0.0
TB_ERR_GOODPUT_GATE_WEIGHT=0.0
TAIL_DELTA_WEIGHT=0.0
TAIL_PRESSURE_WEIGHT=0.0
TAIL_DELTA_CLIP=1.0
TAIL_POTENTIAL_DISCOUNT=1.0
TAIL_POTENTIAL_TOPK=8
TAIL_SERVICE_TARGET=0.75
TAIL_TTL_GUARD_MS=20.0
EXPIRY_DELTA_WEIGHT=0.0
EXPIRY_PRESSURE_WEIGHT=0.0
EXPIRY_DELTA_CLIP=1.0
EXPIRY_POTENTIAL_DISCOUNT=1.0
EXPIRY_POTENTIAL_TOPK=8
EXPIRY_TTL_GUARD_MS=40.0
EXPIRY_REWARD_WINDOW_STEPS=0
EXPIRY_REWARD_OFFERED_BYTES_PER_TTI=0.0
RESCUE_CREDIT_WEIGHT=0.0
RESCUE_MISS_WEIGHT=0.0
RESCUE_TOPK=8
RESCUE_SERVICE_TARGET=0.80
RESCUE_TTL_GUARD_MS=60.0
RESCUE_MAX_SCORE=1.5
RESCUE_GOODPUT_BYTES_SCALE=3000.0

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

Online PPO options:
  --ppo-preset <name>          none | physical_success_blankaware | physical_success_blankaware_v2 | goodput_delay_ue_rate_v1 | goodput_delay_ue_rate_v2_no_teacher | goodput_delay_rate_simple_risk_v1 | goodput_delay_rate_simple_risk_v2_fast | goodput_delay_rate_simple_risk_v3_fast | goodput_delay_rate_simple_risk_v3_tailbias_fast | goodput_delay_rate_simple_risk_v3_repair_fast | goodput_delay_rate_simple_risk_v4_marginal_fast | goodput_delay_rate_simple_risk_v5_relative_margin_fast | goodput_delay_rate_simple_risk_v6_ttl_headroom_fast | goodput_delay_rate_simple_risk_v7_risk_gate_fast | goodput_delay_rate_simple_risk_v7_budgeted_sample_fast | goodput_delay_rate_simple_risk_v7_budgeted_greedy_fast | goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast | goodput_delay_rate_simple_risk_v8_seq_budgeted_greedy_fast | goodput_delay_rate_simple_risk_v9_reliability_gate_sample_fast | goodput_delay_rate_simple_risk_v9_reliability_gate_greedy_fast | goodput_delay_rate_simple_risk_v9b_reliability_gate_sample_fast | goodput_delay_rate_simple_risk_v9b_reliability_gate_greedy_fast | goodput_delay_rate_simple_risk_v9c_prg_efficiency_sample_fast | goodput_delay_rate_simple_risk_v9c_prg_efficiency_greedy_fast | goodput_delay_rate_simple_risk_v9d_stable_sample_fast | goodput_delay_rate_simple_risk_v9d_stable_greedy_fast | goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast | goodput_delay_rate_simple_risk_v9e_gated_fill_greedy_fast | goodput_delay_rate_simple_risk_v10_rate2_sample_fast | goodput_delay_rate_simple_risk_v10_rate2_greedy_fast | goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast | goodput_delay_rate_simple_risk_v11_rate_tail_greedy_fast | goodput_delay_rate_simple_risk_v12_multiseed_rate_sample_fast | goodput_delay_rate_simple_risk_v12_multiseed_rate_greedy_fast | goodput_delay_rate_simple_risk_v12b_steady_rate_sample_fast | goodput_delay_rate_simple_risk_v12b_steady_rate_greedy_fast | goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast | goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_greedy_fast | goodput_delay_rate_simple_risk_v13_ue_rate_proxy_sample_fast | goodput_delay_rate_simple_risk_v13_ue_rate_proxy_greedy_fast | goodput_delay_rate_simple_risk_v14_true_packet_rate_sample_fast | goodput_delay_rate_simple_risk_v14_true_packet_rate_greedy_fast | goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_sample_fast | goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_greedy_fast | goodput_delay_rate_simple_risk_v15_packet_tail_feature_sample_fast | goodput_delay_rate_simple_risk_v15_packet_tail_feature_greedy_fast | goodput_delay_rate_simple_risk_v16_packet_tail_strict_sample_fast | goodput_delay_rate_simple_risk_v16_packet_tail_strict_greedy_fast | goodput_delay_rate_simple_risk_v17_packet_tail_quota_greedy_fast | goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_greedy_fast | goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_greedy_fast | goodput_tailguard_v3_no_teacher | goodput_tailcredit_v4_no_teacher | goodput_tailcredit_expiry_v5_no_teacher | goodput_tailcredit_expiry_rescue_v6_no_teacher
  --out-dir <path>
  --iterations <n>
  --episode-horizon <n>
  --rollout-steps <n>
  --rollout-warmup-steps <n>
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
  --action-head-mode <m>      slot | candidate
  --include-pfq-score-anchor <0|1>
  --include-candidate-risk-features <0|1>
  --candidate-blank-logit-bias <v>
  --candidate-success-logit-scale <v>
  --candidate-blank-success-scale <v>
  --candidate-pfq-anchor-logit-coef <v>
  --candidate-success-aux-coef <v>
  --candidate-success-calib-coef <v>
  --candidate-success-max-calib-coef <v>
  --candidate-success-target-eps <v>
  --candidate-seq-state-dim <n>
  --candidate-seq-state-coef <v>
  --candidate-seq-fail-risk-feature-offset <n>
  --candidate-seq-train-only <0|1>
  --candidate-pairwise-rank-coef <v>
  --candidate-pairwise-rank-margin <v>
  --candidate-pairwise-rank-min-gap <v>
  --candidate-pairwise-rank-success-weight <v>
  --candidate-pairwise-rank-service-weight <v>
  --candidate-pairwise-rank-risk-weight <v>
  --candidate-pairwise-rank-fail-risk-weight <v>
  --candidate-pairwise-rank-rescue-weight <v>
  --candidate-rescue-aux-coef <v>
  --candidate-rescue-pos-weight <v>
  --candidate-rescue-service-target <v>
  --candidate-rescue-ttl-guard-ms <v>
  --candidate-rescue-max-target <v>
  --candidate-decode-goodput-coef <v>
  --candidate-decode-goodput-target-bytes <v>
  --candidate-decode-goodput-max-target <v>
  --candidate-decode-goodput-pos-weight <v>
  --candidate-argmax-decode-coef <v>
  --candidate-argmax-decode-pos-weight <v>
  --candidate-argmax-decode-headroom-rank-mode <m>
  --candidate-blank-target-coef <v>
  --candidate-blank-target-threshold <v>
  --candidate-blank-target-margin <v>
  --candidate-blank-target-deadband <v>
  --candidate-blank-target-pos-weight <v>
  --candidate-blank-target-fill-weight <v>
  --candidate-blank-target-positive-only <0|1>
  --candidate-blank-target-service-guard <0|1>
  --candidate-blank-target-service-min-success <v>
  --candidate-blank-target-service-min-debt <v>
  --candidate-blank-target-service-min-hol-ms <v>
  --candidate-blank-target-service-max-scheduled-ratio <v>
  --candidate-blank-target-service-fill-weight <v>
  --candidate-blank-target-service-fill-margin <v>
  --candidate-blank-target-service-fill-min-success <v>
  --safe-fill-max-per-cell <n>
  --safe-fill-min-quality-db <v>
  --safe-fill-max-quality-gap-db <v>
  --safe-fill-max-prg-risk <v>
  --safe-fill-min-backlog-bytes <v>
  --safe-fill-max-ue-tb-err <v>
  --candidate-blank-budget-max-per-cell <n>
  --candidate-blank-budget-min-decision-logit <v>
  --candidate-decode-demand-slack-bytes <v>
  --candidate-marginal-headroom-slack-bytes <v>
  --candidate-budgeted-select-mode <off|greedy|sample>
  --candidate-budgeted-hard-cap <0|1>
  --candidate-budgeted-usage-penalty <v>
  --candidate-budgeted-overcap-penalty <v>
  --candidate-budgeted-first-service-bonus <v>
  --candidate-budgeted-tail-bias-coef <v>
  --candidate-budgeted-tail-bias-service-target <v>
  --candidate-budgeted-tail-bias-ttl-guard-ms <v>
  --candidate-budgeted-tail-bias-max <v>
  --candidate-packet-tail-feature-enable <0|1>
  --candidate-packet-tail-rate-target-mbps <v>
  --candidate-packet-tail-delay-scale-ms <v>
  --candidate-budgeted-rescue-bias-coef <v>
  --candidate-budgeted-rescue-bias-service-target <v>
  --candidate-budgeted-rescue-bias-ttl-guard-ms <v>
  --candidate-budgeted-rescue-bias-max <v>
  --candidate-budgeted-completion-bonus <v>
  --candidate-budgeted-completion-progress-weight <v>
  --candidate-budgeted-completion-max-score <v>
  --candidate-budgeted-tail-quota-enable <0|1>
  --candidate-budgeted-tail-quota-min-score <v>
  --candidate-budgeted-tail-quota-topk-ues <n>
  --candidate-budgeted-tail-quota-prgs-per-ue <n>
  --candidate-budgeted-tail-quota-bonus <v>
  --candidate-budgeted-tail-quota-order-bonus <v>
  --candidate-budgeted-tail-quota-max-score <v>
  --candidate-budgeted-tail-quota-risk-max <v>
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
  --target-kl-min-updates <n>
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
  --write-diagnostics <0|1>
  --init-checkpoint <path>
  --imitation-episodes <n>
  --imitation-max-steps <n>
  --imitation-lr <v>
  --teacher-mode <m>          none | heuristic_graph | pfq_passthrough
  --teacher-aux-coef <v>
  --teacher-aux-min-coef <v>
  --teacher-aux-decay-iters <n>
  --teacher-soft-target <0|1>
  --teacher-soft-temperature <v>
  --teacher-soft-hit-bonus <v>
  --teacher-soft-blank-bonus <v>
  --teacher-soft-blank-safe-penalty <v>
  --teacher-soft-blank-no-safe-bonus <v>
  --reward-mode <raw_env|stageb_v1|stageb_blankaware|stageb_tailguard_simple|stageb_tailcredit_v4|stageb_tailcredit_expiry_v5|stageb_tailcredit_expiry_rescue_v6>
  --best-metric <reward|goodput|service_guarded_goodput|latency_guarded_goodput|packet_latency_guarded_goodput|reliability_guarded_goodput|rate_guarded_goodput|rate_tail_guarded_goodput|multi_seed_rate_guarded_goodput>
  --early-stop-patience <n>
  --early-stop-min-delta <v>
  --early-stop-warmup-iters <n>
  --best-min-iter <n>
  --candidate-checkpoint-interval <n>
  --candidate-checkpoint-require-gate <0|1>
  --best-service-gap-penalty-mbps <v>
  --best-unserved-ue-target <v>
  --best-unserved-penalty-mbps <v>
  --best-tb-err-target <v>
  --best-tb-err-penalty-mbps <v>
  --best-expiry-penalty-mbps <v>
  --best-packet-rate-bonus-mbps <v>
  --best-packet-per-packet-rate-bonus-mbps <v>
  --best-packet-delay-penalty-mbps <v>
  --best-ue-macro-rate-proxy-bonus-mbps <v>
  --best-ue-p10-rate-proxy-gap-penalty-mbps <v>
  --best-ue-p10-rate-proxy-target-mbps <v>
  --best-ue-macro-packet-rate-bonus-mbps <v>
  --best-ue-p10-packet-rate-gap-penalty-mbps <v>
  --best-ue-p10-packet-rate-target-mbps <v>
  --best-backlog-p90-penalty-mbps <v>
  --best-hard-tb-err-max <v>
  --best-hard-expiry-rate-max <v>
  --best-hard-service-p10-min <v>
  --best-hard-unserved-ue-max <v>
  --best-hard-backlog-p90-max <v>
  --best-hard-packet-rate-min-mbps <v>
  --best-hard-packet-per-packet-rate-min-mbps <v>
  --best-hard-packet-delay-max-ms <v>
  --best-hard-ue-macro-rate-proxy-min-mbps <v>
  --best-hard-ue-p10-rate-proxy-min-mbps <v>
  --best-hard-ue-macro-packet-rate-min-mbps <v>
  --best-hard-ue-p10-packet-rate-min-mbps <v>
  --best-multi-seed-window-iters <n>
  --best-multi-seed-min-covered-seeds <n>
  --goodput-weight <v>
  --tb-err-weight <v>
  --expiry-weight <v>
  --conflict-weight <v>
  --ici-weight <v>
  --urgent-backlog-weight <v>
  --urgent-ttl-ms <v>
  --backlog-debt-weight <v>
  --packet-rate-weight <v>
  --packet-per-packet-rate-weight <v>
  --packet-completion-weight <v>
  --packet-delay-weight <v>
  --packet-rate-scale-mbps <v>
  --packet-per-packet-rate-scale-mbps <v>
  --packet-completion-scale <v>
  --packet-delay-scale-ms <v>
  --ue-macro-rate-proxy-weight <v>
  --ue-p10-rate-proxy-gap-weight <v>
  --ue-rate-proxy-scale-mbps <v>
  --ue-p10-rate-proxy-target-mbps <v>
  --ue-macro-packet-rate-weight <v>
  --ue-p10-packet-rate-gap-weight <v>
  --ue-packet-rate-scale-mbps <v>
  --ue-p10-packet-rate-target-mbps <v>
  --aged-backlog-weight <v>
  --unserved-ue-weight <v>
  --service-gap-weight <v>
  --service-rate-target <v>
  --util-floor-weight <v>
  --util-floor-target <v>
  --util-floor-tb-err-guard <v>
  --util-floor-tb-err-softness <v>
  --missed-safe-opportunity-weight <v>
  --missed-safe-opportunity-scale <v>
  --prg-efficiency-weight <v>
  --harmful-packing-weight <v>
  --tail-delta-weight <v>
  --tail-pressure-weight <v>
  --tail-delta-clip <v>
  --tail-potential-discount <v>
  --tail-potential-topk <n>
  --tail-service-target <v>
  --tail-ttl-guard-ms <v>
  --expiry-delta-weight <v>
  --expiry-pressure-weight <v>
  --expiry-delta-clip <v>
  --expiry-potential-discount <v>
  --expiry-potential-topk <n>
  --expiry-ttl-guard-ms <v>
  --expiry-reward-window-steps <n>
  --expiry-reward-offered-bytes-per-tti <v>
  --rescue-credit-weight <v>
  --rescue-miss-weight <v>
  --rescue-topk <n>
  --rescue-service-target <v>
  --rescue-ttl-guard-ms <v>
  --rescue-max-score <v>
  --rescue-goodput-bytes-scale <v>
  --sim-env <KEY=VALUE>       Extra simulator env var, may be repeated
  -h, --help
EOF
}

apply_ppo_preset() {
    local preset="$1"
    case "${preset}" in
        ""|none)
            ;;
        physical_success_blankaware)
            # 7-cell candidate path: learn physical-success-aware residuals
            # over PFQ while optimizing for efficient reuse rather than raw fill.
            ACTION_HEAD_MODE="candidate"
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
            CANDIDATE_SUCCESS_AUX_COEF=0.05
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
            CANDIDATE_DECODE_GOODPUT_COEF=0.02
            CANDIDATE_ARGMAX_DECODE_COEF=0.05
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            REWARD_MODE="stageb_blankaware"
            ICI_WEIGHT=1.0
            CONFLICT_WEIGHT=1.0
            PACKET_RATE_WEIGHT=0.20
            PACKET_COMPLETION_WEIGHT=0.05
            UTIL_FLOOR_WEIGHT=1.5
            UTIL_FLOOR_TARGET=0.95
            UTIL_FLOOR_TB_ERR_GUARD=0.155
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.03
            ;;
        physical_success_blankaware_v2)
            # Follow-up after v1: keep the efficient-reuse reward, but make
            # physical success and PHY-ranked decode harder gates.
            ACTION_HEAD_MODE="candidate"
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            REWARD_MODE="stageb_blankaware"
            ICI_WEIGHT=1.0
            CONFLICT_WEIGHT=1.0
            PACKET_RATE_WEIGHT=0.20
            PACKET_COMPLETION_WEIGHT=0.05
            UTIL_FLOOR_WEIGHT=1.0
            UTIL_FLOOR_TARGET=0.95
            UTIL_FLOOR_TB_ERR_GUARD=0.145
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.02
            ;;
        goodput_delay_ue_rate_v1)
            # Main 7-cell candidate path for high goodput, low delay, and UE-rate
            # protection. PRG utilization is diagnostic here; productive blanking
            # is allowed when it improves goodput per active PRG and avoids BLER.
            ACTION_HEAD_MODE="candidate"
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=1.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_SERVICE_GAP_PENALTY_MBPS=220.0
            BEST_UNSERVED_UE_TARGET=4.0
            BEST_UNSERVED_PENALTY_MBPS=5.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=800.0
            BEST_EXPIRY_PENALTY_MBPS=1500.0
            BEST_PACKET_RATE_BONUS_MBPS=16.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00008
            GOODPUT_WEIGHT=0.08
            TB_ERR_WEIGHT=32.0
            EXPIRY_WEIGHT=60.0
            ICI_WEIGHT=1.25
            CONFLICT_WEIGHT=1.25
            PACKET_RATE_WEIGHT=0.40
            PACKET_COMPLETION_WEIGHT=0.10
            PACKET_DELAY_WEIGHT=1.20
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.60
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            UTIL_FLOOR_TB_ERR_GUARD=0.145
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.02
            ;;
        goodput_delay_ue_rate_v2_no_teacher)
            # Strict no-teacher follow-up: keep physical-success/decode auxiliaries,
            # remove PFQ anchor input/bias, and make expiry/debt/service-p10 much
            # more visible in the scalar reward.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_SERVICE_GAP_PENALTY_MBPS=360.0
            BEST_UNSERVED_UE_TARGET=3.0
            BEST_UNSERVED_PENALTY_MBPS=8.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=900.0
            BEST_EXPIRY_PENALTY_MBPS=2400.0
            BEST_PACKET_RATE_BONUS_MBPS=16.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00012
            GOODPUT_WEIGHT=0.10
            TB_ERR_WEIGHT=36.0
            EXPIRY_WEIGHT=120.0
            ICI_WEIGHT=1.15
            CONFLICT_WEIGHT=1.15
            URGENT_BACKLOG_WEIGHT=0.75
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.45
            PACKET_COMPLETION_WEIGHT=0.12
            PACKET_DELAY_WEIGHT=1.40
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            UTIL_FLOOR_TB_ERR_GUARD=0.145
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.02
            ;;
        goodput_delay_rate_simple_risk_v1)
            # Stochastic-policy PPO for the 3cell coop-center postEq path.
            # Keep reward readable: goodput first, with light packet-rate,
            # completion, delay/debt, and reliability terms.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            INCLUDE_CANDIDATE_RISK_FEATURES=1
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.08
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.03
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.70
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.25
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.18
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
            CANDIDATE_DECODE_GOODPUT_COEF=0.02
            CANDIDATE_ARGMAX_DECODE_COEF=0.0
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=3000.0
            CANDIDATE_BUDGETED_SELECT_MODE="off"
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_SERVICE_GAP_PENALTY_MBPS=120.0
            BEST_UNSERVED_UE_TARGET=13.0
            BEST_UNSERVED_PENALTY_MBPS=3.0
            BEST_TB_ERR_TARGET=0.122
            BEST_TB_ERR_PENALTY_MBPS=500.0
            BEST_EXPIRY_PENALTY_MBPS=1200.0
            BEST_PACKET_RATE_BONUS_MBPS=12.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00005
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=10.0
            EXPIRY_WEIGHT=12.0
            ICI_WEIGHT=0.70
            CONFLICT_WEIGHT=0.70
            URGENT_BACKLOG_WEIGHT=0.40
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.18
            PACKET_COMPLETION_WEIGHT=0.04
            PACKET_DELAY_WEIGHT=0.80
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.45
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            ;;
        goodput_delay_rate_simple_risk_v2_fast)
            # Same clear objective as v1: goodput first, with delay/rate and
            # reliability guards. Tighten the gates that v1 left effectively off.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            INCLUDE_CANDIDATE_RISK_FEATURES=1
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.08
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.03
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.70
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.30
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.20
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
            CANDIDATE_DECODE_GOODPUT_COEF=0.02
            CANDIDATE_ARGMAX_DECODE_COEF=0.0
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=3000.0
            CANDIDATE_BUDGETED_SELECT_MODE="off"
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=40
            BEST_SERVICE_GAP_PENALTY_MBPS=260.0
            BEST_UNSERVED_UE_TARGET=2.70
            BEST_UNSERVED_PENALTY_MBPS=12.0
            BEST_TB_ERR_TARGET=0.125
            BEST_TB_ERR_PENALTY_MBPS=500.0
            BEST_EXPIRY_PENALTY_MBPS=1600.0
            BEST_PACKET_RATE_BONUS_MBPS=12.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00008
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=10.0
            EXPIRY_WEIGHT=12.0
            ICI_WEIGHT=0.70
            CONFLICT_WEIGHT=0.70
            URGENT_BACKLOG_WEIGHT=0.40
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.20
            PACKET_COMPLETION_WEIGHT=0.04
            PACKET_DELAY_WEIGHT=0.90
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.70
            UTIL_FLOOR_WEIGHT=1.50
            UTIL_FLOOR_TARGET=0.955
            UTIL_FLOOR_TB_ERR_GUARD=0.130
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.020
            ROLLOUT_LOG_EVERY_STEPS=0
            UPDATE_LOG_EVERY_MINIBATCHES=0
            WRITE_DIAGNOSTICS=0
            ;;
        goodput_delay_rate_simple_risk_v3_fast|goodput_delay_rate_simple_risk_v3_tailbias_fast)
            # v3 keeps the same simple objective but fixes two v2 issues:
            #  - candidate risk now includes weak-layer postEq SINR/spread features;
            #  - delay/service/util guards are scaled enough to affect PPO gradients.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            INCLUDE_CANDIDATE_RISK_FEATURES=1
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.08
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.65
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.45
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.30
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
            CANDIDATE_DECODE_GOODPUT_COEF=0.04
            CANDIDATE_ARGMAX_DECODE_COEF=0.0
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=3000.0
            CANDIDATE_BUDGETED_SELECT_MODE="off"
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=40
            BEST_SERVICE_GAP_PENALTY_MBPS=380.0
            BEST_UNSERVED_UE_TARGET=2.10
            BEST_UNSERVED_PENALTY_MBPS=22.0
            BEST_TB_ERR_TARGET=0.125
            BEST_TB_ERR_PENALTY_MBPS=550.0
            BEST_EXPIRY_PENALTY_MBPS=1900.0
            BEST_PACKET_RATE_BONUS_MBPS=14.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=10.0
            EXPIRY_WEIGHT=14.0
            ICI_WEIGHT=0.70
            CONFLICT_WEIGHT=0.70
            URGENT_BACKLOG_WEIGHT=0.45
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.28
            PACKET_COMPLETION_WEIGHT=0.06
            PACKET_DELAY_WEIGHT=1.10
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=45.00
            UTIL_FLOOR_TARGET=0.965
            UTIL_FLOOR_TB_ERR_GUARD=0.135
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.025
            ROLLOUT_LOG_EVERY_STEPS=0
            UPDATE_LOG_EVERY_MINIBATCHES=0
            WRITE_DIAGNOSTICS=0
            KPI_TTI_LOG_INTERVAL=0
            if [[ "${preset}" == "goodput_delay_rate_simple_risk_v3_tailbias_fast" ]]; then
                # Keep the reward simple; apply a lightweight direct logit
                # bias toward UEs with short-term service debt without the
                # sequential budgeted selector cost.
                CANDIDATE_BUDGETED_SELECT_MODE="off"
                CANDIDATE_BUDGETED_HARD_CAP=1
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
                CANDIDATE_BUDGETED_OVERCAP_PENALTY=8.0
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.0
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.55
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.5
                SAFE_FILL_MAX_PER_CELL=1
                SAFE_FILL_MIN_QUALITY_DB=-2.0
                SAFE_FILL_MAX_QUALITY_GAP_DB=8.0
                SAFE_FILL_MAX_PRG_RISK=0.70
                SAFE_FILL_MIN_BACKLOG_BYTES=3000.0
                SAFE_FILL_MAX_UE_TB_ERR=0.45
            fi
            ;;
        goodput_delay_rate_simple_risk_v3_repair_fast)
            # v3 plus minimal coverage repair. Keep the objective simple, but
            # avoid wasting high-confidence windows on blank PRGs or repeatedly
            # skipping demand UEs.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            INCLUDE_CANDIDATE_RISK_FEATURES=1
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.08
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.65
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.45
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.30
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
            CANDIDATE_DECODE_GOODPUT_COEF=0.04
            CANDIDATE_ARGMAX_DECODE_COEF=0.0
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=3000.0
            CANDIDATE_BUDGETED_SELECT_MODE="off"
            SLOT_REPAIR_MAX_PER_CELL=1
            SLOT_REPAIR_MIN_BACKLOG_BYTES=3000.0
            SAFE_FILL_MAX_PER_CELL=1
            SAFE_FILL_MIN_QUALITY_DB=-2.0
            SAFE_FILL_MAX_QUALITY_GAP_DB=8.0
            SAFE_FILL_MAX_PRG_RISK=0.70
            SAFE_FILL_MIN_BACKLOG_BYTES=3000.0
            SAFE_FILL_MAX_UE_TB_ERR=0.45
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=40
            BEST_SERVICE_GAP_PENALTY_MBPS=380.0
            BEST_UNSERVED_UE_TARGET=2.10
            BEST_UNSERVED_PENALTY_MBPS=22.0
            BEST_TB_ERR_TARGET=0.125
            BEST_TB_ERR_PENALTY_MBPS=550.0
            BEST_EXPIRY_PENALTY_MBPS=1900.0
            BEST_PACKET_RATE_BONUS_MBPS=14.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=10.0
            EXPIRY_WEIGHT=14.0
            ICI_WEIGHT=0.70
            CONFLICT_WEIGHT=0.70
            URGENT_BACKLOG_WEIGHT=0.45
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.28
            PACKET_COMPLETION_WEIGHT=0.06
            PACKET_DELAY_WEIGHT=1.10
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=45.00
            UTIL_FLOOR_TARGET=0.965
            UTIL_FLOOR_TB_ERR_GUARD=0.135
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.025
            ROLLOUT_LOG_EVERY_STEPS=0
            UPDATE_LOG_EVERY_MINIBATCHES=0
            WRITE_DIAGNOSTICS=0
            KPI_TTI_LOG_INTERVAL=0
            ;;
        goodput_delay_rate_simple_risk_v4_marginal_fast|goodput_delay_rate_simple_risk_v5_relative_margin_fast|goodput_delay_rate_simple_risk_v6_ttl_headroom_fast|goodput_delay_rate_simple_risk_v7_risk_gate_fast|goodput_delay_rate_simple_risk_v7_budgeted_sample_fast|goodput_delay_rate_simple_risk_v7_budgeted_greedy_fast|goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast|goodput_delay_rate_simple_risk_v8_seq_budgeted_greedy_fast|goodput_delay_rate_simple_risk_v9_reliability_gate_sample_fast|goodput_delay_rate_simple_risk_v9_reliability_gate_greedy_fast|goodput_delay_rate_simple_risk_v9b_reliability_gate_sample_fast|goodput_delay_rate_simple_risk_v9b_reliability_gate_greedy_fast|goodput_delay_rate_simple_risk_v9c_prg_efficiency_sample_fast|goodput_delay_rate_simple_risk_v9c_prg_efficiency_greedy_fast|goodput_delay_rate_simple_risk_v9d_stable_sample_fast|goodput_delay_rate_simple_risk_v9d_stable_greedy_fast|goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast|goodput_delay_rate_simple_risk_v9e_gated_fill_greedy_fast|goodput_delay_rate_simple_risk_v10_rate2_sample_fast|goodput_delay_rate_simple_risk_v10_rate2_greedy_fast|goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast|goodput_delay_rate_simple_risk_v11_rate_tail_greedy_fast|goodput_delay_rate_simple_risk_v12_multiseed_rate_sample_fast|goodput_delay_rate_simple_risk_v12_multiseed_rate_greedy_fast|goodput_delay_rate_simple_risk_v12b_steady_rate_sample_fast|goodput_delay_rate_simple_risk_v12b_steady_rate_greedy_fast|goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast|goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_greedy_fast|goodput_delay_rate_simple_risk_v13_ue_rate_proxy_sample_fast|goodput_delay_rate_simple_risk_v13_ue_rate_proxy_greedy_fast|goodput_delay_rate_simple_risk_v14_true_packet_rate_sample_fast|goodput_delay_rate_simple_risk_v14_true_packet_rate_greedy_fast|goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_sample_fast|goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_greedy_fast|goodput_delay_rate_simple_risk_v15_packet_tail_feature_sample_fast|goodput_delay_rate_simple_risk_v15_packet_tail_feature_greedy_fast|goodput_delay_rate_simple_risk_v16_packet_tail_strict_sample_fast|goodput_delay_rate_simple_risk_v16_packet_tail_strict_greedy_fast|goodput_delay_rate_simple_risk_v17_packet_tail_quota_greedy_fast|goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_greedy_fast|goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_greedy_fast)
            # v4 keeps the objective compact: goodput first, BLER/expiry guard,
            # and packet-rate/delay terms. It avoids a hard duplicate-UE penalty:
            # repeated PRGs are allowed for high-backlog UEs, but excess over a
            # soft marginal-demand cap is considered only after first/soft service.
            # v5 keeps the same reward and uses richer same-PRG relative margin
            # edge features for candidate ranking. v6 keeps the reward unchanged
            # and adds direct TTL/packet-age/headroom context to candidate edges.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            INCLUDE_CANDIDATE_RISK_FEATURES=1
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.08
            CANDIDATE_SUCCESS_CALIB_COEF=0.02
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.01
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.65
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.45
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.30
            CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.0
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.0
            if [[ "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v5_relative_margin_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v6_ttl_headroom_fast" ]]; then
                CANDIDATE_PAIRWISE_RANK_MARGIN=0.18
                CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.07
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.70
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.38
                EXPIRY_REWARD_WINDOW_STEPS=1000
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v7_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v8_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v10* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v11* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12* ]]; then
                CANDIDATE_PAIRWISE_RANK_COEF=0.05
                CANDIDATE_PAIRWISE_RANK_MARGIN=0.24
                CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.75
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.35
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.48
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.35
                EXPIRY_REWARD_WINDOW_STEPS=1000
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_* ]]; then
                CANDIDATE_PAIRWISE_RANK_COEF=0.05
                CANDIDATE_PAIRWISE_RANK_MARGIN=0.24
                CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.75
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.35
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.48
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.35
                EXPIRY_REWARD_WINDOW_STEPS=1000
            fi
            CANDIDATE_DECODE_GOODPUT_COEF=0.04
            CANDIDATE_ARGMAX_DECODE_COEF=0.0
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="stable"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
            CANDIDATE_BUDGETED_SELECT_MODE="off"
            if [[ "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v7_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v7_budgeted_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v8_seq_budgeted_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9_reliability_gate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9b_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9b_reliability_gate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9c_prg_efficiency_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9c_prg_efficiency_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9d_stable_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9d_stable_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9e_gated_fill_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v10_rate2_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v10_rate2_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v11_rate_tail_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12_multiseed_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12_multiseed_rate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12b_steady_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12b_steady_rate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_greedy_fast" ]]; then
                if [[ "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v7_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9b_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9c_prg_efficiency_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9d_stable_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v10_rate2_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12_multiseed_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12b_steady_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast" ]]; then
                    CANDIDATE_BUDGETED_SELECT_MODE="sample"
                else
                    CANDIDATE_BUDGETED_SELECT_MODE="greedy"
                fi
                CANDIDATE_ARGMAX_DECODE_COEF=0.025
                CANDIDATE_BUDGETED_HARD_CAP=1
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.30
                CANDIDATE_BUDGETED_OVERCAP_PENALTY=8.0
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.02
                if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v8_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v10* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v11* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12* ]]; then
                    CANDIDATE_SEQ_STATE_DIM=8
                    CANDIDATE_SEQ_STATE_COEF=1.0
                    CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET=5
                    LR=5e-5
                fi
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_* ]]; then
                if [[ "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v13_ue_rate_proxy_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v14_true_packet_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v15_packet_tail_feature_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v16_packet_tail_strict_sample_fast" ]]; then
                    CANDIDATE_BUDGETED_SELECT_MODE="sample"
                else
                    CANDIDATE_BUDGETED_SELECT_MODE="greedy"
                fi
                CANDIDATE_ARGMAX_DECODE_COEF=0.025
                CANDIDATE_BUDGETED_HARD_CAP=1
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.30
                CANDIDATE_BUDGETED_OVERCAP_PENALTY=8.0
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.02
                CANDIDATE_SEQ_STATE_DIM=8
                CANDIDATE_SEQ_STATE_COEF=1.0
                CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET=5
                LR=5e-5
            fi
            SLOT_REPAIR_MAX_PER_CELL=2
            SLOT_REPAIR_MIN_BACKLOG_BYTES=3000.0
            SAFE_FILL_MAX_PER_CELL=1
            SAFE_FILL_MIN_QUALITY_DB=-2.0
            SAFE_FILL_MAX_QUALITY_GAP_DB=8.0
            SAFE_FILL_MAX_PRG_RISK=0.70
            SAFE_FILL_MIN_BACKLOG_BYTES=3000.0
            SAFE_FILL_MAX_UE_TB_ERR=0.45
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_blankaware"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=40
            BEST_SERVICE_GAP_PENALTY_MBPS=320.0
            BEST_UNSERVED_UE_TARGET=2.40
            BEST_UNSERVED_PENALTY_MBPS=18.0
            BEST_TB_ERR_TARGET=0.125
            BEST_TB_ERR_PENALTY_MBPS=550.0
            BEST_EXPIRY_PENALTY_MBPS=1900.0
            BEST_PACKET_RATE_BONUS_MBPS=14.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v7_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v8_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v10* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v11* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12* ]]; then
                BEST_TB_ERR_TARGET=0.123
                BEST_TB_ERR_PENALTY_MBPS=850.0
                BEST_EXPIRY_PENALTY_MBPS=2600.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00018
                BEST_HARD_TB_ERR_MAX=0.130
                BEST_HARD_EXPIRY_RATE_MAX=0.026
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_* ]]; then
                BEST_TB_ERR_TARGET=0.123
                BEST_TB_ERR_PENALTY_MBPS=850.0
                BEST_EXPIRY_PENALTY_MBPS=2600.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00018
                BEST_HARD_TB_ERR_MAX=0.130
                BEST_HARD_EXPIRY_RATE_MAX=0.026
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
            fi
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=10.0
            EXPIRY_WEIGHT=14.0
            ICI_WEIGHT=0.70
            CONFLICT_WEIGHT=0.70
            URGENT_BACKLOG_WEIGHT=0.45
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.28
            PACKET_COMPLETION_WEIGHT=0.06
            PACKET_DELAY_WEIGHT=1.10
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.72
            UTIL_FLOOR_WEIGHT=4.00
            UTIL_FLOOR_TARGET=0.940
            UTIL_FLOOR_TB_ERR_GUARD=0.135
            UTIL_FLOOR_TB_ERR_SOFTNESS=0.025
            MISSED_SAFE_OPPORTUNITY_WEIGHT=2.00
            MISSED_SAFE_OPPORTUNITY_SCALE=51.0
            if [[ "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v7_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v7_budgeted_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v8_seq_budgeted_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v8_seq_budgeted_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9_reliability_gate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9b_reliability_gate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9b_reliability_gate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9c_prg_efficiency_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9c_prg_efficiency_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9d_stable_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9d_stable_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9e_gated_fill_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v9e_gated_fill_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v10_rate2_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v10_rate2_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v11_rate_tail_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v11_rate_tail_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12_multiseed_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12_multiseed_rate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12b_steady_rate_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12b_steady_rate_greedy_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_sample_fast" || "${PPO_PRESET}" == "goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_greedy_fast" ]]; then
                # The budgeted selector's proven win comes from leaving unsafe/no-headroom
                # PRGs blank; do not train it back toward high utilization.
                UTIL_FLOOR_WEIGHT=0.0
                UTIL_FLOOR_TARGET=0.0
                UTIL_FLOOR_TB_ERR_GUARD=1.0
                UTIL_FLOOR_TB_ERR_SOFTNESS=0.0
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.50
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14_* ]]; then
                UTIL_FLOOR_WEIGHT=0.0
                UTIL_FLOOR_TARGET=0.0
                UTIL_FLOOR_TB_ERR_GUARD=1.0
                UTIL_FLOOR_TB_ERR_SOFTNESS=0.0
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.50
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9* ]]; then
                # v9 changes the objective from packet-rate/delay-biased score
                # to reliability-gated decoded goodput for the deployment gate.
                BEST_METRIC="reliability_guarded_goodput"
                BEST_MIN_ITER=5
                BEST_TB_ERR_TARGET=0.116
                BEST_TB_ERR_PENALTY_MBPS=2500.0
                BEST_EXPIRY_PENALTY_MBPS=2600.0
                BEST_PACKET_RATE_BONUS_MBPS=0.0
                BEST_PACKET_DELAY_PENALTY_MBPS=0.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00010
                BEST_HARD_TB_ERR_MAX=0.118
                BEST_HARD_EXPIRY_RATE_MAX=0.010
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
                TB_ERR_WEIGHT=56.0
                EXPIRY_WEIGHT=18.0
                PACKET_RATE_WEIGHT=0.10
                PACKET_COMPLETION_WEIGHT=0.02
                PACKET_DELAY_WEIGHT=0.35
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.20
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.82
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.28
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.58
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.55
                CANDIDATE_ARGMAX_DECODE_COEF=0.05
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.40
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.01
                SAFE_FILL_MAX_PRG_RISK=0.55
                SAFE_FILL_MAX_UE_TB_ERR=0.35
                TB_ERR_TARGET=0.116
                TB_ERR_OVER_TARGET_WEIGHT=20.0
                TB_ERR_GOODPUT_GATE_WEIGHT=12.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9b_* ]]; then
                BEST_TB_ERR_TARGET=0.115
                BEST_TB_ERR_PENALTY_MBPS=3200.0
                BEST_HARD_TB_ERR_MAX=0.117
                TB_ERR_WEIGHT=64.0
                PACKET_RATE_WEIGHT=0.06
                PACKET_COMPLETION_WEIGHT=0.01
                PACKET_DELAY_WEIGHT=0.25
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.88
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.66
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.70
                SAFE_FILL_MAX_PRG_RISK=0.48
                SAFE_FILL_MAX_UE_TB_ERR=0.30
                TB_ERR_TARGET=0.115
                TB_ERR_OVER_TARGET_WEIGHT=28.0
                TB_ERR_GOODPUT_GATE_WEIGHT=20.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9c_* ]]; then
                # v9c keeps v9b's reliability target but aligns training decode
                # with the deployment sweep that beat PFQ: leave marginal/no-headroom
                # PRGs blank instead of backfilling them via safe-fill.
                BEST_TB_ERR_TARGET=0.115
                BEST_TB_ERR_PENALTY_MBPS=3200.0
                BEST_HARD_TB_ERR_MAX=0.117
                TB_ERR_WEIGHT=64.0
                PACKET_RATE_WEIGHT=0.06
                PACKET_COMPLETION_WEIGHT=0.01
                PACKET_DELAY_WEIGHT=0.25
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.88
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.66
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.70
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.48
                SAFE_FILL_MAX_UE_TB_ERR=0.30
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.45
                HARMFUL_PACKING_WEIGHT=24.0
                TB_ERR_TARGET=0.115
                TB_ERR_OVER_TARGET_WEIGHT=28.0
                TB_ERR_GOODPUT_GATE_WEIGHT=20.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9d_* ]]; then
                # v9d keeps v9c's deployment semantics but makes fine-tuning
                # conservative enough for warm-started sequential/budgeted PPO.
                # It targets the seed45-style failure: over-blank/low-service
                # without falling back to unsafe post-decode safe-fill.
                LR=5e-6
                ACTOR_LR=5e-6
                CRITIC_LR=1e-4
                CLIP_EPS=0.05
                ENTROPY_COEF=0.002
                TARGET_KL=0.01
                TARGET_KL_MIN_UPDATES=1
                UPDATE_EPOCHS=1
                MINIBATCH_SIZE=128
                BEST_METRIC="reliability_guarded_goodput"
                BEST_MIN_ITER=3
                BEST_TB_ERR_TARGET=0.115
                BEST_TB_ERR_PENALTY_MBPS=3000.0
                BEST_EXPIRY_PENALTY_MBPS=2800.0
                BEST_PACKET_RATE_BONUS_MBPS=0.0
                BEST_PACKET_DELAY_PENALTY_MBPS=0.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00012
                BEST_HARD_TB_ERR_MAX=0.117
                BEST_HARD_EXPIRY_RATE_MAX=0.010
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
                TB_ERR_WEIGHT=60.0
                EXPIRY_WEIGHT=20.0
                PACKET_RATE_WEIGHT=0.10
                PACKET_COMPLETION_WEIGHT=0.02
                PACKET_DELAY_WEIGHT=0.20
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.86
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.38
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.62
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.66
                CANDIDATE_ARGMAX_DECODE_COEF=0.05
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.30
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.03
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.08
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.2
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.06
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.2
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.48
                SAFE_FILL_MAX_UE_TB_ERR=0.30
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.35
                HARMFUL_PACKING_WEIGHT=20.0
                TB_ERR_TARGET=0.115
                TB_ERR_OVER_TARGET_WEIGHT=24.0
                TB_ERR_GOODPUT_GATE_WEIGHT=18.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v9e_* ]]; then
                # v9e targets the seed45 failure mode observed after v9d:
                # reliability was healthy, but the policy over-blanked. Keep the
                # stable PPO step and deployment semantics, then add a BLER-gated
                # fill floor that is active only while reliability is comfortably
                # inside the hard gate.
                LR=2e-6
                ACTOR_LR=2e-6
                CRITIC_LR=1e-4
                CLIP_EPS=0.05
                ENTROPY_COEF=0.0025
                TARGET_KL=0.01
                TARGET_KL_MIN_UPDATES=1
                UPDATE_EPOCHS=1
                MINIBATCH_SIZE=128
                BEST_METRIC="reliability_guarded_goodput"
                BEST_MIN_ITER=4
                BEST_TB_ERR_TARGET=0.115
                BEST_TB_ERR_PENALTY_MBPS=3000.0
                BEST_EXPIRY_PENALTY_MBPS=2800.0
                BEST_PACKET_RATE_BONUS_MBPS=0.0
                BEST_PACKET_DELAY_PENALTY_MBPS=0.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00012
                BEST_HARD_TB_ERR_MAX=0.117
                BEST_HARD_EXPIRY_RATE_MAX=0.010
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
                TB_ERR_WEIGHT=58.0
                EXPIRY_WEIGHT=20.0
                PACKET_RATE_WEIGHT=0.14
                PACKET_COMPLETION_WEIGHT=0.03
                PACKET_DELAY_WEIGHT=0.20
                SERVICE_GAP_WEIGHT=0.0
                SERVICE_RATE_TARGET=0.72
                UTIL_FLOOR_WEIGHT=12.0
                UTIL_FLOOR_TARGET=0.82
                UTIL_FLOOR_TB_ERR_GUARD=0.110
                UTIL_FLOOR_TB_ERR_SOFTNESS=0.008
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.05
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.84
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.44
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.58
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.62
                CANDIDATE_ARGMAX_DECODE_COEF=0.0
                CANDIDATE_DECODE_GOODPUT_COEF=0.03
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.20
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.05
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.10
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.2
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.08
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.2
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.48
                SAFE_FILL_MAX_UE_TB_ERR=0.30
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.30
                HARMFUL_PACKING_WEIGHT=18.0
                TB_ERR_TARGET=0.115
                TB_ERR_OVER_TARGET_WEIGHT=24.0
                TB_ERR_GOODPUT_GATE_WEIGHT=18.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v10_rate2_* ]]; then
                # v10 keeps the v9c/v9d deployment safety shape and promotes rate
                # as the second training axis: aggregate packet service rate for
                # capacity, per-completed-packet rate for latency-aware service.
                LR=3e-6
                ACTOR_LR=3e-6
                CRITIC_LR=1e-4
                CLIP_EPS=0.05
                ENTROPY_COEF=0.0025
                TARGET_KL=0.01
                TARGET_KL_MIN_UPDATES=1
                UPDATE_EPOCHS=1
                MINIBATCH_SIZE=128
                BEST_METRIC="rate_guarded_goodput"
                BEST_MIN_ITER=4
                BEST_TB_ERR_TARGET=0.115
                BEST_TB_ERR_PENALTY_MBPS=3000.0
                BEST_EXPIRY_PENALTY_MBPS=2800.0
                BEST_PACKET_RATE_BONUS_MBPS=8.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=4.0
                BEST_PACKET_DELAY_PENALTY_MBPS=20.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00012
                BEST_HARD_TB_ERR_MAX=0.117
                BEST_HARD_EXPIRY_RATE_MAX=0.010
                BEST_HARD_SERVICE_P10_MIN=0.70
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1250000.0
                TB_ERR_WEIGHT=60.0
                EXPIRY_WEIGHT=20.0
                PACKET_RATE_WEIGHT=0.20
                PACKET_PER_PACKET_RATE_WEIGHT=0.10
                PACKET_COMPLETION_WEIGHT=0.03
                PACKET_DELAY_WEIGHT=0.25
                PACKET_RATE_SCALE_MBPS=10.0
                PACKET_PER_PACKET_RATE_SCALE_MBPS=24.0
                SERVICE_GAP_WEIGHT=0.0
                SERVICE_RATE_TARGET=0.72
                UTIL_FLOOR_WEIGHT=0.0
                UTIL_FLOOR_TARGET=0.0
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.86
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.48
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.62
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.66
                CANDIDATE_ARGMAX_DECODE_COEF=0.05
                CANDIDATE_DECODE_GOODPUT_COEF=0.03
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.24
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.06
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.12
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.2
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.08
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.2
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.48
                SAFE_FILL_MAX_UE_TB_ERR=0.30
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.35
                HARMFUL_PACKING_WEIGHT=20.0
                TB_ERR_TARGET=0.115
                TB_ERR_OVER_TARGET_WEIGHT=24.0
                TB_ERR_GOODPUT_GATE_WEIGHT=18.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v11_rate_tail_* ]]; then
                # v11 targets the cross-seed failure exposed by v10: goodput/TB can
                # look healthy while packet service rate collapses and delay grows.
                # Keep the safe budgeted deployment shape, then make rate/tail
                # quality a hard checkpoint gate rather than a small bonus only.
                LR=2e-6
                ACTOR_LR=2e-6
                CRITIC_LR=1e-4
                CLIP_EPS=0.05
                ENTROPY_COEF=0.002
                TARGET_KL=0.008
                TARGET_KL_MIN_UPDATES=1
                UPDATE_EPOCHS=1
                MINIBATCH_SIZE=128
                BEST_METRIC="rate_tail_guarded_goodput"
                BEST_MIN_ITER=4
                BEST_TB_ERR_TARGET=0.113
                BEST_TB_ERR_PENALTY_MBPS=3600.0
                BEST_EXPIRY_PENALTY_MBPS=3200.0
                BEST_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=10.0
                BEST_PACKET_DELAY_PENALTY_MBPS=80.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
                BEST_HARD_TB_ERR_MAX=0.116
                BEST_HARD_EXPIRY_RATE_MAX=0.005
                BEST_HARD_SERVICE_P10_MIN=0.80
                BEST_HARD_UNSERVED_UE_MAX=2.40
                BEST_HARD_BACKLOG_P90_MAX=750000.0
                BEST_HARD_PACKET_RATE_MIN_MBPS=22.5
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=31.5
                BEST_HARD_PACKET_DELAY_MAX_MS=1.25
                TB_ERR_WEIGHT=64.0
                EXPIRY_WEIGHT=24.0
                PACKET_RATE_WEIGHT=0.28
                PACKET_PER_PACKET_RATE_WEIGHT=0.22
                PACKET_COMPLETION_WEIGHT=0.04
                PACKET_DELAY_WEIGHT=0.55
                PACKET_RATE_SCALE_MBPS=10.0
                PACKET_PER_PACKET_RATE_SCALE_MBPS=24.0
                SERVICE_GAP_WEIGHT=0.0
                SERVICE_RATE_TARGET=0.80
                UTIL_FLOOR_WEIGHT=0.0
                UTIL_FLOOR_TARGET=0.0
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.82
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.62
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.62
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.66
                CANDIDATE_ARGMAX_DECODE_COEF=0.04
                CANDIDATE_DECODE_GOODPUT_COEF=0.025
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.18
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.10
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.18
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.82
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=30.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.12
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.84
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=80.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.5
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.46
                SAFE_FILL_MAX_UE_TB_ERR=0.28
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.30
                HARMFUL_PACKING_WEIGHT=18.0
                TB_ERR_TARGET=0.113
                TB_ERR_OVER_TARGET_WEIGHT=30.0
                TB_ERR_GOODPUT_GATE_WEIGHT=22.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12_multiseed_rate_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12b_steady_rate_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_ue_rate_proxy_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14_true_packet_rate_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_packet_tail_feature_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_packet_tail_strict_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_packet_tail_quota_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_* ]]; then
                # v12 promotes the rate objective from a single-seed gate to a
                # rolling multi-seed gate.  It is meant for list_cycle training
                # over the known failure seeds (41/42/43 first), so checkpoint
                # selection uses the worst seed in the recent window rather than
                # a single attractive seed41 rollout.
                LR=1.5e-6
                ACTOR_LR=1.5e-6
                CRITIC_LR=8e-5
                CLIP_EPS=0.04
                ENTROPY_COEF=0.002
                TARGET_KL=0.006
                TARGET_KL_MIN_UPDATES=1
                UPDATE_EPOCHS=1
                MINIBATCH_SIZE=128
                BEST_METRIC="multi_seed_rate_guarded_goodput"
                BEST_MIN_ITER=6
                BEST_MULTI_SEED_WINDOW_ITERS=6
                BEST_MULTI_SEED_MIN_COVERED_SEEDS=3
                BEST_TB_ERR_TARGET=0.114
                BEST_TB_ERR_PENALTY_MBPS=4200.0
                BEST_EXPIRY_PENALTY_MBPS=3600.0
                BEST_PACKET_RATE_BONUS_MBPS=14.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=14.0
                BEST_PACKET_DELAY_PENALTY_MBPS=120.0
                BEST_BACKLOG_P90_PENALTY_MBPS=0.00016
                BEST_HARD_TB_ERR_MAX=0.118
                BEST_HARD_EXPIRY_RATE_MAX=0.004
                BEST_HARD_SERVICE_P10_MIN=0.74
                BEST_HARD_UNSERVED_UE_MAX=3.20
                BEST_HARD_BACKLOG_P90_MAX=1100000.0
                BEST_HARD_PACKET_RATE_MIN_MBPS=6.0
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=27.0
                BEST_HARD_PACKET_DELAY_MAX_MS=8.0
                TB_ERR_WEIGHT=66.0
                EXPIRY_WEIGHT=26.0
                PACKET_RATE_WEIGHT=0.34
                PACKET_PER_PACKET_RATE_WEIGHT=0.30
                PACKET_COMPLETION_WEIGHT=0.04
                PACKET_DELAY_WEIGHT=0.75
                PACKET_RATE_SCALE_MBPS=10.0
                PACKET_PER_PACKET_RATE_SCALE_MBPS=24.0
                SERVICE_GAP_WEIGHT=0.0
                SERVICE_RATE_TARGET=0.80
                UTIL_FLOOR_WEIGHT=0.0
                UTIL_FLOOR_TARGET=0.0
                CANDIDATE_SUCCESS_CALIB_COEF=0.03
                CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.03
                CANDIDATE_PAIRWISE_RANK_COEF=0.06
                CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.80
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.72
                CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.62
                CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT=0.68
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.18
                CANDIDATE_RESCUE_AUX_COEF=0.015
                CANDIDATE_RESCUE_SERVICE_TARGET=0.86
                CANDIDATE_RESCUE_TTL_GUARD_MS=100.0
                CANDIDATE_ARGMAX_DECODE_COEF=0.035
                CANDIDATE_DECODE_GOODPUT_COEF=0.025
                CANDIDATE_DECODE_DEMAND_SLACK_BYTES=0.0
                CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES=3000.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.16
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.12
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.24
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.86
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=45.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.8
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.18
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.88
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=120.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.8
                SAFE_FILL_MAX_PER_CELL=0
                SAFE_FILL_MAX_PRG_RISK=0.46
                SAFE_FILL_MAX_UE_TB_ERR=0.28
                MISSED_SAFE_OPPORTUNITY_WEIGHT=0.0
                PRG_EFFICIENCY_WEIGHT=0.28
                HARMFUL_PACKING_WEIGHT=18.0
                TB_ERR_TARGET=0.114
                TB_ERR_OVER_TARGET_WEIGHT=32.0
                TB_ERR_GOODPUT_GATE_WEIGHT=24.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12b_steady_rate_* ]]; then
                # v12b makes the training objective match the deployment metric:
                # drive the current actor through the cold-start interval, then
                # collect PPO and checkpoint metrics from the steady-state window.
                EPISODE_HORIZON=1512
                ROLLOUT_WARMUP_STEPS=1000
                ROLLOUT_STEPS=512
                BEST_HARD_TB_ERR_MAX=0.118
                BEST_HARD_PACKET_RATE_MIN_MBPS=7.5
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=27.0
                BEST_HARD_PACKET_DELAY_MAX_MS=7.0
                BEST_PACKET_RATE_BONUS_MBPS=18.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=10.0
                BEST_PACKET_DELAY_PENALTY_MBPS=160.0
                PACKET_RATE_WEIGHT=0.42
                PACKET_PER_PACKET_RATE_WEIGHT=0.22
                PACKET_DELAY_WEIGHT=0.95
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.28
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=80.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.24
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=140.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v12c_steady_tail_rescue_* ]]; then
                # v12c keeps v12b's deployment-aligned warmup, but adds direct
                # steady-state tail/rescue shaping so seed42/43 service-time
                # collapse is not left to packet-rate aggregates alone.
                EPISODE_HORIZON=1512
                ROLLOUT_WARMUP_STEPS=1000
                ROLLOUT_STEPS=512
                BEST_HARD_TB_ERR_MAX=0.120
                BEST_HARD_PACKET_RATE_MIN_MBPS=7.5
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=26.8
                BEST_HARD_PACKET_DELAY_MAX_MS=7.5
                BEST_PACKET_RATE_BONUS_MBPS=22.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_DELAY_PENALTY_MBPS=190.0
                PACKET_RATE_WEIGHT=0.50
                PACKET_PER_PACKET_RATE_WEIGHT=0.26
                PACKET_DELAY_WEIGHT=1.10
                TAIL_DELTA_WEIGHT=0.70
                TAIL_PRESSURE_WEIGHT=0.12
                TAIL_DELTA_CLIP=0.75
                TAIL_POTENTIAL_DISCOUNT=1.0
                TAIL_TTL_GUARD_MS=120.0
                EXPIRY_DELTA_WEIGHT=0.55
                EXPIRY_PRESSURE_WEIGHT=0.18
                EXPIRY_DELTA_CLIP=0.75
                EXPIRY_POTENTIAL_DISCOUNT=1.0
                EXPIRY_TTL_GUARD_MS=160.0
                RESCUE_CREDIT_WEIGHT=0.45
                RESCUE_MISS_WEIGHT=0.20
                RESCUE_SERVICE_TARGET=0.90
                RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.35
                CANDIDATE_RESCUE_AUX_COEF=0.035
                CANDIDATE_RESCUE_POS_WEIGHT=3.0
                CANDIDATE_RESCUE_SERVICE_TARGET=0.90
                CANDIDATE_RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.85
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=2.2
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.65
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=180.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=2.2
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v13_ue_rate_proxy_* ]]; then
                # v13 promotes rate into a second mainline at UE granularity.
                # The current bridge does not expose true per-UE packet system
                # time, so this uses demand-active recent100 goodput Mbps as a
                # clearly named proxy while keeping the v12b steady-state warmup.
                EPISODE_HORIZON=1512
                ROLLOUT_WARMUP_STEPS=1000
                ROLLOUT_STEPS=512
                BEST_HARD_TB_ERR_MAX=0.120
                BEST_HARD_PACKET_RATE_MIN_MBPS=7.2
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=26.8
                BEST_HARD_PACKET_DELAY_MAX_MS=8.0
                BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS=10.0
                BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS=90.0
                BEST_UE_P10_RATE_PROXY_TARGET_MBPS=16.0
                BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS=34.0
                BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS=8.0
                BEST_PACKET_RATE_BONUS_MBPS=20.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_DELAY_PENALTY_MBPS=185.0
                PACKET_RATE_WEIGHT=0.48
                PACKET_PER_PACKET_RATE_WEIGHT=0.24
                PACKET_DELAY_WEIGHT=1.05
                UE_MACRO_RATE_PROXY_WEIGHT=0.20
                UE_P10_RATE_PROXY_GAP_WEIGHT=0.70
                UE_RATE_PROXY_SCALE_MBPS=8.0
                UE_P10_RATE_PROXY_TARGET_MBPS=16.0
                TAIL_DELTA_WEIGHT=0.55
                TAIL_PRESSURE_WEIGHT=0.10
                TAIL_DELTA_CLIP=0.75
                TAIL_POTENTIAL_DISCOUNT=1.0
                TAIL_TTL_GUARD_MS=120.0
                EXPIRY_DELTA_WEIGHT=0.45
                EXPIRY_PRESSURE_WEIGHT=0.16
                EXPIRY_DELTA_CLIP=0.75
                EXPIRY_POTENTIAL_DISCOUNT=1.0
                EXPIRY_TTL_GUARD_MS=160.0
                RESCUE_CREDIT_WEIGHT=0.35
                RESCUE_MISS_WEIGHT=0.18
                RESCUE_SERVICE_TARGET=0.90
                RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.30
                CANDIDATE_RESCUE_AUX_COEF=0.030
                CANDIDATE_RESCUE_POS_WEIGHT=3.0
                CANDIDATE_RESCUE_SERVICE_TARGET=0.90
                CANDIDATE_RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.65
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=2.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.50
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=180.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=2.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14_true_packet_rate_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_packet_tail_feature_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_packet_tail_strict_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_packet_tail_quota_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_* ]]; then
                # v14 keeps the v13 steady-state/candidate-budget scaffold, but
                # uses native per-UE packet system-time deltas instead of the
                # goodput-rate proxy for the UE-rate objective and gate.
                EPISODE_HORIZON=1512
                ROLLOUT_WARMUP_STEPS=1000
                ROLLOUT_STEPS=512
                BEST_HARD_TB_ERR_MAX=0.120
                BEST_HARD_PACKET_RATE_MIN_MBPS=7.2
                BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS=26.8
                BEST_HARD_PACKET_DELAY_MAX_MS=8.0
                BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS=0.0
                BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS=0.0
                BEST_UE_P10_RATE_PROXY_TARGET_MBPS=0.0
                BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS=-1.0
                BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS=-1.0
                BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=16.0
                BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=120.0
                BEST_UE_P10_PACKET_RATE_TARGET_MBPS=4.0
                BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=2.5
                BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=0.8
                BEST_PACKET_RATE_BONUS_MBPS=18.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_DELAY_PENALTY_MBPS=185.0
                PACKET_RATE_WEIGHT=0.44
                PACKET_PER_PACKET_RATE_WEIGHT=0.22
                PACKET_DELAY_WEIGHT=1.05
                UE_MACRO_RATE_PROXY_WEIGHT=0.0
                UE_P10_RATE_PROXY_GAP_WEIGHT=0.0
                UE_MACRO_PACKET_RATE_WEIGHT=0.35
                UE_P10_PACKET_RATE_GAP_WEIGHT=0.90
                UE_PACKET_RATE_SCALE_MBPS=4.0
                UE_P10_PACKET_RATE_TARGET_MBPS=4.0
                TAIL_DELTA_WEIGHT=0.55
                TAIL_PRESSURE_WEIGHT=0.10
                TAIL_DELTA_CLIP=0.75
                TAIL_POTENTIAL_DISCOUNT=1.0
                TAIL_TTL_GUARD_MS=120.0
                EXPIRY_DELTA_WEIGHT=0.45
                EXPIRY_PRESSURE_WEIGHT=0.16
                EXPIRY_DELTA_CLIP=0.75
                EXPIRY_POTENTIAL_DISCOUNT=1.0
                EXPIRY_TTL_GUARD_MS=160.0
                RESCUE_CREDIT_WEIGHT=0.35
                RESCUE_MISS_WEIGHT=0.18
                RESCUE_SERVICE_TARGET=0.90
                RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.30
                CANDIDATE_RESCUE_AUX_COEF=0.030
                CANDIDATE_RESCUE_POS_WEIGHT=3.0
                CANDIDATE_RESCUE_SERVICE_TARGET=0.90
                CANDIDATE_RESCUE_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.65
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=160.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=2.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.50
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.92
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=180.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=2.0
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v14b_true_packet_tail_guard_* ]]; then
                # v14b reacts to the seed43 pilot: total goodput/rate improved,
                # but true UE p10 packet-rate collapsed after one PPO update.
                # Make true packet-tail rate a stronger training and checkpoint
                # constraint, while de-emphasizing the macro-rate bonus.
                BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=4.0
                BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=220.0
                BEST_UE_P10_PACKET_RATE_TARGET_MBPS=2.0
                BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=10.0
                BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=1.0
                BEST_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=8.0
                BEST_PACKET_DELAY_PENALTY_MBPS=220.0
                PACKET_RATE_WEIGHT=0.30
                PACKET_PER_PACKET_RATE_WEIGHT=0.18
                PACKET_DELAY_WEIGHT=1.25
                UE_MACRO_PACKET_RATE_WEIGHT=0.05
                UE_P10_PACKET_RATE_GAP_WEIGHT=4.00
                UE_PACKET_RATE_SCALE_MBPS=2.0
                UE_P10_PACKET_RATE_TARGET_MBPS=2.0
                TAIL_DELTA_WEIGHT=0.75
                TAIL_PRESSURE_WEIGHT=0.16
                RESCUE_CREDIT_WEIGHT=0.55
                RESCUE_MISS_WEIGHT=0.25
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.82
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.42
                CANDIDATE_RESCUE_AUX_COEF=0.045
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=1.05
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.94
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=180.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=2.4
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.80
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.94
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=200.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=2.4
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v15_packet_tail_feature_* ]]; then
                # v15 moves true packet-rate from reward-only shaping into the
                # candidate selector through training-internal UE tail features.
                BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=4.0
                BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=260.0
                BEST_UE_P10_PACKET_RATE_TARGET_MBPS=2.0
                BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=10.0
                BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=1.0
                BEST_PACKET_RATE_BONUS_MBPS=12.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=8.0
                BEST_PACKET_DELAY_PENALTY_MBPS=240.0
                PACKET_RATE_WEIGHT=0.30
                PACKET_PER_PACKET_RATE_WEIGHT=0.18
                PACKET_DELAY_WEIGHT=1.30
                UE_MACRO_PACKET_RATE_WEIGHT=0.05
                UE_P10_PACKET_RATE_GAP_WEIGHT=4.50
                UE_PACKET_RATE_SCALE_MBPS=2.0
                UE_P10_PACKET_RATE_TARGET_MBPS=2.0
                CANDIDATE_PACKET_TAIL_FEATURE_ENABLE=1
                CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS=2.0
                CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS=10.0
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=1.25
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.94
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=180.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=2.6
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.95
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.94
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=200.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=2.6
                TAIL_DELTA_WEIGHT=0.80
                TAIL_PRESSURE_WEIGHT=0.18
                RESCUE_CREDIT_WEIGHT=0.60
                RESCUE_MISS_WEIGHT=0.28
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v16_packet_tail_strict_* ]]; then
                # v16 reacts to the v15 seed43 pilot: the true packet-tail
                # feature worked, but a 2 Mbps selector target was too soft.
                # Treat UE packet-rate as the second mainline by lifting the
                # selector target and spreading first service before packing.
                BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=2.0
                BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=360.0
                BEST_UE_P10_PACKET_RATE_TARGET_MBPS=8.0
                BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=12.0
                BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=2.0
                BEST_PACKET_RATE_BONUS_MBPS=10.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=8.0
                BEST_PACKET_DELAY_PENALTY_MBPS=300.0
                PACKET_RATE_WEIGHT=0.28
                PACKET_PER_PACKET_RATE_WEIGHT=0.16
                PACKET_DELAY_WEIGHT=1.45
                UE_MACRO_PACKET_RATE_WEIGHT=0.03
                UE_P10_PACKET_RATE_GAP_WEIGHT=6.00
                UE_PACKET_RATE_SCALE_MBPS=4.0
                UE_P10_PACKET_RATE_TARGET_MBPS=8.0
                CANDIDATE_PACKET_TAIL_FEATURE_ENABLE=1
                CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS=12.0
                CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS=6.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.28
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.45
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=2.80
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.96
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=220.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=4.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=2.00
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.96
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=240.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=4.0
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.90
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.55
                CANDIDATE_RESCUE_AUX_COEF=0.050
                TAIL_DELTA_WEIGHT=0.90
                TAIL_PRESSURE_WEIGHT=0.22
                RESCUE_CREDIT_WEIGHT=0.70
                RESCUE_MISS_WEIGHT=0.35
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v17_packet_tail_quota_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_* || "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_* ]]; then
                # v17 keeps v16's true packet-tail features but moves the
                # pressure from soft score bias to an explicit per-UE tail
                # quota pass before ordinary greedy packing.
                BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS=2.0
                BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS=340.0
                BEST_UE_P10_PACKET_RATE_TARGET_MBPS=6.0
                BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS=12.0
                BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS=1.0
                BEST_PACKET_RATE_BONUS_MBPS=10.0
                BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS=8.0
                BEST_PACKET_DELAY_PENALTY_MBPS=300.0
                PACKET_RATE_WEIGHT=0.28
                PACKET_PER_PACKET_RATE_WEIGHT=0.16
                PACKET_DELAY_WEIGHT=1.45
                UE_MACRO_PACKET_RATE_WEIGHT=0.03
                UE_P10_PACKET_RATE_GAP_WEIGHT=5.50
                UE_PACKET_RATE_SCALE_MBPS=4.0
                UE_P10_PACKET_RATE_TARGET_MBPS=6.0
                CANDIDATE_PACKET_TAIL_FEATURE_ENABLE=1
                CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS=12.0
                CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS=6.0
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.24
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.25
                CANDIDATE_BUDGETED_TAIL_BIAS_COEF=1.60
                CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.96
                CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=220.0
                CANDIDATE_BUDGETED_TAIL_BIAS_MAX=3.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=1.10
                CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.96
                CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=240.0
                CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=3.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE=1
                CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE=0.55
                CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES=8
                CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE=1
                CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS=6.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS=4.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_MAX_SCORE=4.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX=0.70
                CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.90
                CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.55
                CANDIDATE_RESCUE_AUX_COEF=0.050
                TAIL_DELTA_WEIGHT=0.90
                TAIL_PRESSURE_WEIGHT=0.22
                RESCUE_CREDIT_WEIGHT=0.70
                RESCUE_MISS_WEIGHT=0.35
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v18_packet_tail_quota_thick_* ]]; then
                # v18 responds to v17's seed43 failure mode: the worst UEs were
                # served every recent TTI but only received one thin PRG, so
                # packet-rate p10 still collapsed. Keep the same tail detector
                # and reserve fewer, thicker UE quotas before greedy packing.
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.16
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.10
                CANDIDATE_BUDGETED_COMPLETION_BONUS=0.80
                CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT=0.35
                CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE=2.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE=0.50
                CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES=5
                CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE=3
                CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS=7.5
                CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS=5.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX=0.78
            fi
            if [[ "${PPO_PRESET}" == goodput_delay_rate_simple_risk_v19_packet_tail_quota_coverage_* ]]; then
                # v19 keeps v18's explicit quota idea but spreads the same
                # approximate quota budget across more tail UEs. This targets
                # the cell-1 thin-service cluster where top5*3 helped p10 only
                # transiently, while top8*2 stayed above the UE p10 gate.
                CANDIDATE_BUDGETED_USAGE_PENALTY=0.18
                CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.10
                CANDIDATE_BUDGETED_COMPLETION_BONUS=0.50
                CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT=0.35
                CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE=2.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE=0.35
                CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES=8
                CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE=2
                CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS=7.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS=5.0
                CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX=0.78
            fi
            ROLLOUT_LOG_EVERY_STEPS=0
            UPDATE_LOG_EVERY_MINIBATCHES=0
            WRITE_DIAGNOSTICS=0
            KPI_TTI_LOG_INTERVAL=0
            ;;
        goodput_tailguard_v3_no_teacher)
            # Simple three-signal reward:
            #   goodput_norm - reliability(BLER+expiry) - max-pooled tail pressure.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.04
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.25
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.10
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_tailguard_simple"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_SERVICE_GAP_PENALTY_MBPS=420.0
            BEST_UNSERVED_UE_TARGET=3.0
            BEST_UNSERVED_PENALTY_MBPS=8.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=900.0
            BEST_EXPIRY_PENALTY_MBPS=2600.0
            BEST_PACKET_RATE_BONUS_MBPS=0.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=1.00
            EXPIRY_WEIGHT=6.00
            ICI_WEIGHT=0.0
            CONFLICT_WEIGHT=0.0
            URGENT_BACKLOG_WEIGHT=0.0
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.0
            PACKET_COMPLETION_WEIGHT=0.0
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=1.25
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            ;;
        goodput_tailcredit_v4_no_teacher)
            # Soft tail credit assignment:
            #   goodput_norm - reliability + delta(tail_potential) - residual tail pressure.
            # The tail signal is a small decode logit bias plus a reward potential delta;
            # it is not a hard scheduling constraint and keeps exploration available.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.05
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.55
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.60
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.15
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.03
            CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.20
            CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
            CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=20.0
            CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_tailcredit_v4"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_SERVICE_GAP_PENALTY_MBPS=420.0
            BEST_UNSERVED_UE_TARGET=3.0
            BEST_UNSERVED_PENALTY_MBPS=8.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=900.0
            BEST_EXPIRY_PENALTY_MBPS=2600.0
            BEST_PACKET_RATE_BONUS_MBPS=0.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00014
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=1.00
            EXPIRY_WEIGHT=6.00
            ICI_WEIGHT=0.0
            CONFLICT_WEIGHT=0.0
            URGENT_BACKLOG_WEIGHT=0.0
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.0
            PACKET_COMPLETION_WEIGHT=0.0
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            TAIL_DELTA_WEIGHT=1.00
            TAIL_PRESSURE_WEIGHT=0.15
            TAIL_DELTA_CLIP=0.75
            TAIL_POTENTIAL_DISCOUNT=1.0
            TAIL_POTENTIAL_TOPK=8
            TAIL_SERVICE_TARGET=0.75
            TAIL_TTL_GUARD_MS=20.0
            ;;
        goodput_tailcredit_expiry_v5_no_teacher)
            # v5 keeps v4's soft tail credit and adds a focused expiry
            # potential delta for TTL-near-drop packets. Best checkpoints are
            # gated until the policy has moved beyond the noisy opening iters.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.05
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.55
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.60
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.15
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.03
            CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.22
            CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
            CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=30.0
            CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_tailcredit_expiry_v5"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=12
            BEST_SERVICE_GAP_PENALTY_MBPS=450.0
            BEST_UNSERVED_UE_TARGET=3.0
            BEST_UNSERVED_PENALTY_MBPS=9.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=1050.0
            BEST_EXPIRY_PENALTY_MBPS=3400.0
            BEST_PACKET_RATE_BONUS_MBPS=0.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00016
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=1.00
            EXPIRY_WEIGHT=8.00
            ICI_WEIGHT=0.0
            CONFLICT_WEIGHT=0.0
            URGENT_BACKLOG_WEIGHT=0.0
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.0
            PACKET_COMPLETION_WEIGHT=0.0
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            TAIL_DELTA_WEIGHT=1.00
            TAIL_PRESSURE_WEIGHT=0.12
            TAIL_DELTA_CLIP=0.75
            TAIL_POTENTIAL_DISCOUNT=1.0
            TAIL_POTENTIAL_TOPK=8
            TAIL_SERVICE_TARGET=0.75
            TAIL_TTL_GUARD_MS=20.0
            EXPIRY_DELTA_WEIGHT=1.25
            EXPIRY_PRESSURE_WEIGHT=0.35
            EXPIRY_DELTA_CLIP=0.75
            EXPIRY_POTENTIAL_DISCOUNT=1.0
            EXPIRY_POTENTIAL_TOPK=8
            EXPIRY_TTL_GUARD_MS=40.0
            ;;
        goodput_tailcredit_expiry_rescue_v6_no_teacher)
            # v6 keeps the v4/v5 soft tail/expiry design, but adds UE-local
            # rescue credit so a single PRG action receives clearer signal for
            # serving near-expiry or long-starved UEs.
            ACTION_HEAD_MODE="candidate"
            INCLUDE_PFQ_SCORE_ANCHOR=0
            CANDIDATE_MODE="all_demand_ues"
            MAX_PRG_CANDIDATES=12
            CANDIDATE_PFQ_ANCHOR_LOGIT_COEF=0.0
            CANDIDATE_SUCCESS_AUX_COEF=0.10
            CANDIDATE_SUCCESS_CALIB_COEF=0.03
            CANDIDATE_SUCCESS_MAX_CALIB_COEF=0.02
            CANDIDATE_PAIRWISE_RANK_COEF=0.05
            CANDIDATE_PAIRWISE_RANK_MARGIN=0.20
            CANDIDATE_PAIRWISE_RANK_MIN_GAP=0.08
            CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT=0.50
            CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT=0.50
            CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT=0.12
            CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT=0.45
            CANDIDATE_RESCUE_AUX_COEF=0.08
            CANDIDATE_RESCUE_POS_WEIGHT=3.0
            CANDIDATE_RESCUE_SERVICE_TARGET=0.80
            CANDIDATE_RESCUE_TTL_GUARD_MS=60.0
            CANDIDATE_RESCUE_MAX_TARGET=1.0
            CANDIDATE_DECODE_GOODPUT_COEF=0.03
            CANDIDATE_ARGMAX_DECODE_COEF=0.08
            CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="phy"
            CANDIDATE_DECODE_DEMAND_SLACK_BYTES=9000.0
            CANDIDATE_BUDGETED_SELECT_MODE="greedy"
            CANDIDATE_BUDGETED_HARD_CAP=1
            CANDIDATE_BUDGETED_USAGE_PENALTY=0.35
            CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS=0.03
            CANDIDATE_BUDGETED_TAIL_BIAS_COEF=0.18
            CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET=0.75
            CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS=30.0
            CANDIDATE_BUDGETED_TAIL_BIAS_MAX=1.5
            CANDIDATE_BUDGETED_RESCUE_BIAS_COEF=0.32
            CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET=0.80
            CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS=60.0
            CANDIDATE_BUDGETED_RESCUE_BIAS_MAX=1.5
            IMITATION_EPISODES=0
            IMITATION_MAX_STEPS=0
            IMITATION_LR=0
            TEACHER_MODE="none"
            TEACHER_AUX_COEF=0.0
            TEACHER_AUX_MIN_COEF=0.0
            TEACHER_AUX_DECAY_ITERS=0
            REWARD_MODE="stageb_tailcredit_expiry_rescue_v6"
            BEST_METRIC="packet_latency_guarded_goodput"
            BEST_MIN_ITER=15
            BEST_SERVICE_GAP_PENALTY_MBPS=500.0
            BEST_UNSERVED_UE_TARGET=3.0
            BEST_UNSERVED_PENALTY_MBPS=10.0
            BEST_TB_ERR_TARGET=0.142
            BEST_TB_ERR_PENALTY_MBPS=1050.0
            BEST_EXPIRY_PENALTY_MBPS=3600.0
            BEST_PACKET_RATE_BONUS_MBPS=0.0
            BEST_BACKLOG_P90_PENALTY_MBPS=0.00018
            GOODPUT_WEIGHT=1.00
            TB_ERR_WEIGHT=1.00
            EXPIRY_WEIGHT=8.00
            ICI_WEIGHT=0.0
            CONFLICT_WEIGHT=0.0
            URGENT_BACKLOG_WEIGHT=0.0
            BACKLOG_DEBT_WEIGHT=0.0
            PACKET_RATE_WEIGHT=0.0
            PACKET_COMPLETION_WEIGHT=0.0
            AGED_BACKLOG_WEIGHT=0.0
            UNSERVED_UE_WEIGHT=0.0
            SERVICE_GAP_WEIGHT=0.0
            SERVICE_RATE_TARGET=0.75
            UTIL_FLOOR_WEIGHT=0.0
            UTIL_FLOOR_TARGET=0.0
            TAIL_DELTA_WEIGHT=1.00
            TAIL_PRESSURE_WEIGHT=0.10
            TAIL_DELTA_CLIP=0.75
            TAIL_POTENTIAL_DISCOUNT=1.0
            TAIL_POTENTIAL_TOPK=8
            TAIL_SERVICE_TARGET=0.75
            TAIL_TTL_GUARD_MS=20.0
            EXPIRY_DELTA_WEIGHT=1.25
            EXPIRY_PRESSURE_WEIGHT=0.30
            EXPIRY_DELTA_CLIP=0.75
            EXPIRY_POTENTIAL_DISCOUNT=1.0
            EXPIRY_POTENTIAL_TOPK=8
            EXPIRY_TTL_GUARD_MS=40.0
            RESCUE_CREDIT_WEIGHT=0.80
            RESCUE_MISS_WEIGHT=0.45
            RESCUE_TOPK=8
            RESCUE_SERVICE_TARGET=0.80
            RESCUE_TTL_GUARD_MS=60.0
            RESCUE_MAX_SCORE=1.5
            RESCUE_GOODPUT_BYTES_SCALE=3000.0
            ;;
        *)
            echo "Unknown --ppo-preset: ${preset}" >&2
            usage
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ppo-preset)
            PPO_PRESET="$2"
            apply_ppo_preset "${PPO_PRESET}"
            shift 2
            ;;
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
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --episode-horizon) EPISODE_HORIZON="$2"; shift 2 ;;
        --rollout-steps) ROLLOUT_STEPS="$2"; shift 2 ;;
        --rollout-warmup-steps) ROLLOUT_WARMUP_STEPS="$2"; shift 2 ;;
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
        --action-head-mode) ACTION_HEAD_MODE="$2"; shift 2 ;;
        --include-pfq-score-anchor) INCLUDE_PFQ_SCORE_ANCHOR="$2"; shift 2 ;;
        --include-candidate-risk-features) INCLUDE_CANDIDATE_RISK_FEATURES="$2"; shift 2 ;;
        --candidate-blank-logit-bias) CANDIDATE_BLANK_LOGIT_BIAS="$2"; shift 2 ;;
        --candidate-success-logit-scale) CANDIDATE_SUCCESS_LOGIT_SCALE="$2"; shift 2 ;;
        --candidate-blank-success-scale) CANDIDATE_BLANK_SUCCESS_SCALE="$2"; shift 2 ;;
        --candidate-pfq-anchor-logit-coef) CANDIDATE_PFQ_ANCHOR_LOGIT_COEF="$2"; shift 2 ;;
        --candidate-success-aux-coef) CANDIDATE_SUCCESS_AUX_COEF="$2"; shift 2 ;;
        --candidate-success-calib-coef) CANDIDATE_SUCCESS_CALIB_COEF="$2"; shift 2 ;;
        --candidate-success-max-calib-coef) CANDIDATE_SUCCESS_MAX_CALIB_COEF="$2"; shift 2 ;;
        --candidate-success-target-eps) CANDIDATE_SUCCESS_TARGET_EPS="$2"; shift 2 ;;
        --candidate-seq-state-dim) CANDIDATE_SEQ_STATE_DIM="$2"; shift 2 ;;
        --candidate-seq-state-coef) CANDIDATE_SEQ_STATE_COEF="$2"; shift 2 ;;
        --candidate-seq-fail-risk-feature-offset) CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET="$2"; shift 2 ;;
        --candidate-seq-train-only) CANDIDATE_SEQ_TRAIN_ONLY="$2"; shift 2 ;;
        --candidate-pairwise-rank-coef) CANDIDATE_PAIRWISE_RANK_COEF="$2"; shift 2 ;;
        --candidate-pairwise-rank-margin) CANDIDATE_PAIRWISE_RANK_MARGIN="$2"; shift 2 ;;
        --candidate-pairwise-rank-min-gap) CANDIDATE_PAIRWISE_RANK_MIN_GAP="$2"; shift 2 ;;
        --candidate-pairwise-rank-success-weight) CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT="$2"; shift 2 ;;
        --candidate-pairwise-rank-service-weight) CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT="$2"; shift 2 ;;
        --candidate-pairwise-rank-risk-weight) CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT="$2"; shift 2 ;;
        --candidate-pairwise-rank-fail-risk-weight) CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT="$2"; shift 2 ;;
        --candidate-pairwise-rank-rescue-weight) CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT="$2"; shift 2 ;;
        --candidate-rescue-aux-coef) CANDIDATE_RESCUE_AUX_COEF="$2"; shift 2 ;;
        --candidate-rescue-pos-weight) CANDIDATE_RESCUE_POS_WEIGHT="$2"; shift 2 ;;
        --candidate-rescue-service-target) CANDIDATE_RESCUE_SERVICE_TARGET="$2"; shift 2 ;;
        --candidate-rescue-ttl-guard-ms) CANDIDATE_RESCUE_TTL_GUARD_MS="$2"; shift 2 ;;
        --candidate-rescue-max-target) CANDIDATE_RESCUE_MAX_TARGET="$2"; shift 2 ;;
        --candidate-decode-goodput-coef) CANDIDATE_DECODE_GOODPUT_COEF="$2"; shift 2 ;;
        --candidate-decode-goodput-target-bytes) CANDIDATE_DECODE_GOODPUT_TARGET_BYTES="$2"; shift 2 ;;
        --candidate-decode-goodput-max-target) CANDIDATE_DECODE_GOODPUT_MAX_TARGET="$2"; shift 2 ;;
        --candidate-decode-goodput-pos-weight) CANDIDATE_DECODE_GOODPUT_POS_WEIGHT="$2"; shift 2 ;;
        --candidate-argmax-decode-coef) CANDIDATE_ARGMAX_DECODE_COEF="$2"; shift 2 ;;
        --candidate-argmax-decode-pos-weight) CANDIDATE_ARGMAX_DECODE_POS_WEIGHT="$2"; shift 2 ;;
        --candidate-argmax-decode-headroom-rank-mode) CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE="$2"; shift 2 ;;
        --candidate-blank-target-coef) CANDIDATE_BLANK_TARGET_COEF="$2"; shift 2 ;;
        --candidate-blank-target-threshold) CANDIDATE_BLANK_TARGET_THRESHOLD="$2"; shift 2 ;;
        --candidate-blank-target-margin) CANDIDATE_BLANK_TARGET_MARGIN="$2"; shift 2 ;;
        --candidate-blank-target-deadband) CANDIDATE_BLANK_TARGET_DEADBAND="$2"; shift 2 ;;
        --candidate-blank-target-pos-weight) CANDIDATE_BLANK_TARGET_POS_WEIGHT="$2"; shift 2 ;;
        --candidate-blank-target-fill-weight) CANDIDATE_BLANK_TARGET_FILL_WEIGHT="$2"; shift 2 ;;
        --candidate-blank-target-positive-only) CANDIDATE_BLANK_TARGET_POSITIVE_ONLY="$2"; shift 2 ;;
        --candidate-blank-target-service-guard) CANDIDATE_BLANK_TARGET_SERVICE_GUARD="$2"; shift 2 ;;
        --candidate-blank-target-service-min-success) CANDIDATE_BLANK_TARGET_SERVICE_MIN_SUCCESS="$2"; shift 2 ;;
        --candidate-blank-target-service-min-debt) CANDIDATE_BLANK_TARGET_SERVICE_MIN_DEBT="$2"; shift 2 ;;
        --candidate-blank-target-service-min-hol-ms) CANDIDATE_BLANK_TARGET_SERVICE_MIN_HOL_MS="$2"; shift 2 ;;
        --candidate-blank-target-service-max-scheduled-ratio) CANDIDATE_BLANK_TARGET_SERVICE_MAX_SCHEDULED_RATIO="$2"; shift 2 ;;
        --candidate-blank-target-service-fill-weight) CANDIDATE_BLANK_TARGET_SERVICE_FILL_WEIGHT="$2"; shift 2 ;;
        --candidate-blank-target-service-fill-margin) CANDIDATE_BLANK_TARGET_SERVICE_FILL_MARGIN="$2"; shift 2 ;;
        --candidate-blank-target-service-fill-min-success) CANDIDATE_BLANK_TARGET_SERVICE_FILL_MIN_SUCCESS="$2"; shift 2 ;;
        --safe-fill-max-per-cell) SAFE_FILL_MAX_PER_CELL="$2"; shift 2 ;;
        --safe-fill-min-quality-db) SAFE_FILL_MIN_QUALITY_DB="$2"; shift 2 ;;
        --safe-fill-max-quality-gap-db) SAFE_FILL_MAX_QUALITY_GAP_DB="$2"; shift 2 ;;
        --safe-fill-max-prg-risk) SAFE_FILL_MAX_PRG_RISK="$2"; shift 2 ;;
        --safe-fill-min-backlog-bytes) SAFE_FILL_MIN_BACKLOG_BYTES="$2"; shift 2 ;;
        --safe-fill-max-ue-tb-err) SAFE_FILL_MAX_UE_TB_ERR="$2"; shift 2 ;;
        --candidate-blank-budget-max-per-cell) CANDIDATE_BLANK_BUDGET_MAX_PER_CELL="$2"; shift 2 ;;
        --candidate-blank-budget-min-decision-logit) CANDIDATE_BLANK_BUDGET_MIN_DECISION_LOGIT="$2"; shift 2 ;;
        --candidate-decode-demand-slack-bytes) CANDIDATE_DECODE_DEMAND_SLACK_BYTES="$2"; shift 2 ;;
        --candidate-marginal-headroom-slack-bytes) CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES="$2"; shift 2 ;;
        --candidate-budgeted-select-mode) CANDIDATE_BUDGETED_SELECT_MODE="$2"; shift 2 ;;
        --candidate-budgeted-hard-cap) CANDIDATE_BUDGETED_HARD_CAP="$2"; shift 2 ;;
        --candidate-budgeted-usage-penalty) CANDIDATE_BUDGETED_USAGE_PENALTY="$2"; shift 2 ;;
        --candidate-budgeted-overcap-penalty) CANDIDATE_BUDGETED_OVERCAP_PENALTY="$2"; shift 2 ;;
        --candidate-budgeted-first-service-bonus) CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS="$2"; shift 2 ;;
        --candidate-budgeted-tail-bias-coef) CANDIDATE_BUDGETED_TAIL_BIAS_COEF="$2"; shift 2 ;;
        --candidate-budgeted-tail-bias-service-target) CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET="$2"; shift 2 ;;
        --candidate-budgeted-tail-bias-ttl-guard-ms) CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS="$2"; shift 2 ;;
        --candidate-budgeted-tail-bias-max) CANDIDATE_BUDGETED_TAIL_BIAS_MAX="$2"; shift 2 ;;
        --candidate-packet-tail-feature-enable) CANDIDATE_PACKET_TAIL_FEATURE_ENABLE="$2"; shift 2 ;;
        --candidate-packet-tail-rate-target-mbps) CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS="$2"; shift 2 ;;
        --candidate-packet-tail-delay-scale-ms) CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS="$2"; shift 2 ;;
        --candidate-budgeted-rescue-bias-coef) CANDIDATE_BUDGETED_RESCUE_BIAS_COEF="$2"; shift 2 ;;
        --candidate-budgeted-rescue-bias-service-target) CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET="$2"; shift 2 ;;
        --candidate-budgeted-rescue-bias-ttl-guard-ms) CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS="$2"; shift 2 ;;
        --candidate-budgeted-rescue-bias-max) CANDIDATE_BUDGETED_RESCUE_BIAS_MAX="$2"; shift 2 ;;
        --candidate-budgeted-completion-bonus) CANDIDATE_BUDGETED_COMPLETION_BONUS="$2"; shift 2 ;;
        --candidate-budgeted-completion-progress-weight) CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT="$2"; shift 2 ;;
        --candidate-budgeted-completion-max-score) CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-enable) CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-min-score) CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-topk-ues) CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-prgs-per-ue) CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-bonus) CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-order-bonus) CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-max-score) CANDIDATE_BUDGETED_TAIL_QUOTA_MAX_SCORE="$2"; shift 2 ;;
        --candidate-budgeted-tail-quota-risk-max) CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX="$2"; shift 2 ;;
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
        --target-kl-min-updates) TARGET_KL_MIN_UPDATES="$2"; shift 2 ;;
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
        --write-diagnostics) WRITE_DIAGNOSTICS="$2"; shift 2 ;;
        --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
        --imitation-episodes) IMITATION_EPISODES="$2"; shift 2 ;;
        --imitation-max-steps) IMITATION_MAX_STEPS="$2"; shift 2 ;;
        --imitation-lr) IMITATION_LR="$2"; shift 2 ;;
        --teacher-mode) TEACHER_MODE="$2"; shift 2 ;;
        --teacher-aux-coef) TEACHER_AUX_COEF="$2"; shift 2 ;;
        --teacher-aux-min-coef) TEACHER_AUX_MIN_COEF="$2"; shift 2 ;;
        --teacher-aux-decay-iters) TEACHER_AUX_DECAY_ITERS="$2"; shift 2 ;;
        --teacher-soft-target) TEACHER_SOFT_TARGET="$2"; shift 2 ;;
        --teacher-soft-temperature) TEACHER_SOFT_TEMPERATURE="$2"; shift 2 ;;
        --teacher-soft-hit-bonus) TEACHER_SOFT_HIT_BONUS="$2"; shift 2 ;;
        --teacher-soft-blank-bonus) TEACHER_SOFT_BLANK_BONUS="$2"; shift 2 ;;
        --teacher-soft-blank-safe-penalty) TEACHER_SOFT_BLANK_SAFE_PENALTY="$2"; shift 2 ;;
        --teacher-soft-blank-no-safe-bonus) TEACHER_SOFT_BLANK_NO_SAFE_BONUS="$2"; shift 2 ;;
        --reward-mode) REWARD_MODE="$2"; shift 2 ;;
        --best-metric) BEST_METRIC="$2"; shift 2 ;;
        --early-stop-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --early-stop-min-delta) EARLY_STOP_MIN_DELTA="$2"; shift 2 ;;
        --early-stop-warmup-iters) EARLY_STOP_WARMUP_ITERS="$2"; shift 2 ;;
        --best-min-iter) BEST_MIN_ITER="$2"; shift 2 ;;
        --candidate-checkpoint-interval) CANDIDATE_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --candidate-checkpoint-require-gate) CANDIDATE_CHECKPOINT_REQUIRE_GATE="$2"; shift 2 ;;
        --best-service-gap-penalty-mbps) BEST_SERVICE_GAP_PENALTY_MBPS="$2"; shift 2 ;;
        --best-unserved-ue-target) BEST_UNSERVED_UE_TARGET="$2"; shift 2 ;;
        --best-unserved-penalty-mbps) BEST_UNSERVED_PENALTY_MBPS="$2"; shift 2 ;;
        --best-tb-err-target) BEST_TB_ERR_TARGET="$2"; shift 2 ;;
        --best-tb-err-penalty-mbps) BEST_TB_ERR_PENALTY_MBPS="$2"; shift 2 ;;
        --best-expiry-penalty-mbps) BEST_EXPIRY_PENALTY_MBPS="$2"; shift 2 ;;
        --best-packet-rate-bonus-mbps) BEST_PACKET_RATE_BONUS_MBPS="$2"; shift 2 ;;
        --best-packet-per-packet-rate-bonus-mbps) BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS="$2"; shift 2 ;;
        --best-packet-delay-penalty-mbps) BEST_PACKET_DELAY_PENALTY_MBPS="$2"; shift 2 ;;
        --best-ue-macro-rate-proxy-bonus-mbps) BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS="$2"; shift 2 ;;
        --best-ue-p10-rate-proxy-gap-penalty-mbps) BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS="$2"; shift 2 ;;
        --best-ue-p10-rate-proxy-target-mbps) BEST_UE_P10_RATE_PROXY_TARGET_MBPS="$2"; shift 2 ;;
        --best-ue-macro-packet-rate-bonus-mbps) BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS="$2"; shift 2 ;;
        --best-ue-p10-packet-rate-gap-penalty-mbps) BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS="$2"; shift 2 ;;
        --best-ue-p10-packet-rate-target-mbps) BEST_UE_P10_PACKET_RATE_TARGET_MBPS="$2"; shift 2 ;;
        --best-backlog-p90-penalty-mbps) BEST_BACKLOG_P90_PENALTY_MBPS="$2"; shift 2 ;;
        --best-hard-tb-err-max) BEST_HARD_TB_ERR_MAX="$2"; shift 2 ;;
        --best-hard-expiry-rate-max) BEST_HARD_EXPIRY_RATE_MAX="$2"; shift 2 ;;
        --best-hard-service-p10-min) BEST_HARD_SERVICE_P10_MIN="$2"; shift 2 ;;
        --best-hard-unserved-ue-max) BEST_HARD_UNSERVED_UE_MAX="$2"; shift 2 ;;
        --best-hard-backlog-p90-max) BEST_HARD_BACKLOG_P90_MAX="$2"; shift 2 ;;
        --best-hard-packet-rate-min-mbps) BEST_HARD_PACKET_RATE_MIN_MBPS="$2"; shift 2 ;;
        --best-hard-packet-per-packet-rate-min-mbps) BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS="$2"; shift 2 ;;
        --best-hard-packet-delay-max-ms) BEST_HARD_PACKET_DELAY_MAX_MS="$2"; shift 2 ;;
        --best-hard-ue-macro-rate-proxy-min-mbps) BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS="$2"; shift 2 ;;
        --best-hard-ue-p10-rate-proxy-min-mbps) BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS="$2"; shift 2 ;;
        --best-hard-ue-macro-packet-rate-min-mbps) BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS="$2"; shift 2 ;;
        --best-hard-ue-p10-packet-rate-min-mbps) BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS="$2"; shift 2 ;;
        --best-multi-seed-window-iters) BEST_MULTI_SEED_WINDOW_ITERS="$2"; shift 2 ;;
        --best-multi-seed-min-covered-seeds) BEST_MULTI_SEED_MIN_COVERED_SEEDS="$2"; shift 2 ;;
        --goodput-weight) GOODPUT_WEIGHT="$2"; shift 2 ;;
        --tb-err-weight) TB_ERR_WEIGHT="$2"; shift 2 ;;
        --expiry-weight) EXPIRY_WEIGHT="$2"; shift 2 ;;
        --conflict-weight) CONFLICT_WEIGHT="$2"; shift 2 ;;
        --ici-weight) ICI_WEIGHT="$2"; shift 2 ;;
        --urgent-backlog-weight) URGENT_BACKLOG_WEIGHT="$2"; shift 2 ;;
        --urgent-ttl-ms) URGENT_TTL_MS="$2"; shift 2 ;;
        --backlog-debt-weight) BACKLOG_DEBT_WEIGHT="$2"; shift 2 ;;
        --packet-rate-weight) PACKET_RATE_WEIGHT="$2"; shift 2 ;;
        --packet-per-packet-rate-weight) PACKET_PER_PACKET_RATE_WEIGHT="$2"; shift 2 ;;
        --packet-completion-weight) PACKET_COMPLETION_WEIGHT="$2"; shift 2 ;;
        --packet-delay-weight) PACKET_DELAY_WEIGHT="$2"; shift 2 ;;
        --packet-rate-scale-mbps) PACKET_RATE_SCALE_MBPS="$2"; shift 2 ;;
        --packet-per-packet-rate-scale-mbps) PACKET_PER_PACKET_RATE_SCALE_MBPS="$2"; shift 2 ;;
        --packet-completion-scale) PACKET_COMPLETION_SCALE="$2"; shift 2 ;;
        --packet-delay-scale-ms) PACKET_DELAY_SCALE_MS="$2"; shift 2 ;;
        --ue-macro-rate-proxy-weight) UE_MACRO_RATE_PROXY_WEIGHT="$2"; shift 2 ;;
        --ue-p10-rate-proxy-gap-weight) UE_P10_RATE_PROXY_GAP_WEIGHT="$2"; shift 2 ;;
        --ue-rate-proxy-scale-mbps) UE_RATE_PROXY_SCALE_MBPS="$2"; shift 2 ;;
        --ue-p10-rate-proxy-target-mbps) UE_P10_RATE_PROXY_TARGET_MBPS="$2"; shift 2 ;;
        --ue-macro-packet-rate-weight) UE_MACRO_PACKET_RATE_WEIGHT="$2"; shift 2 ;;
        --ue-p10-packet-rate-gap-weight) UE_P10_PACKET_RATE_GAP_WEIGHT="$2"; shift 2 ;;
        --ue-packet-rate-scale-mbps) UE_PACKET_RATE_SCALE_MBPS="$2"; shift 2 ;;
        --ue-p10-packet-rate-target-mbps) UE_P10_PACKET_RATE_TARGET_MBPS="$2"; shift 2 ;;
        --aged-backlog-weight) AGED_BACKLOG_WEIGHT="$2"; shift 2 ;;
        --unserved-ue-weight) UNSERVED_UE_WEIGHT="$2"; shift 2 ;;
        --service-gap-weight) SERVICE_GAP_WEIGHT="$2"; shift 2 ;;
        --service-rate-target) SERVICE_RATE_TARGET="$2"; shift 2 ;;
        --util-floor-weight) UTIL_FLOOR_WEIGHT="$2"; shift 2 ;;
        --util-floor-target) UTIL_FLOOR_TARGET="$2"; shift 2 ;;
        --util-floor-tb-err-guard) UTIL_FLOOR_TB_ERR_GUARD="$2"; shift 2 ;;
        --util-floor-tb-err-softness) UTIL_FLOOR_TB_ERR_SOFTNESS="$2"; shift 2 ;;
        --missed-safe-opportunity-weight) MISSED_SAFE_OPPORTUNITY_WEIGHT="$2"; shift 2 ;;
        --missed-safe-opportunity-scale) MISSED_SAFE_OPPORTUNITY_SCALE="$2"; shift 2 ;;
        --prg-efficiency-weight) PRG_EFFICIENCY_WEIGHT="$2"; shift 2 ;;
        --harmful-packing-weight) HARMFUL_PACKING_WEIGHT="$2"; shift 2 ;;
        --tb-err-target) TB_ERR_TARGET="$2"; shift 2 ;;
        --tb-err-over-target-weight) TB_ERR_OVER_TARGET_WEIGHT="$2"; shift 2 ;;
        --tb-err-goodput-gate-weight) TB_ERR_GOODPUT_GATE_WEIGHT="$2"; shift 2 ;;
        --tail-delta-weight) TAIL_DELTA_WEIGHT="$2"; shift 2 ;;
        --tail-pressure-weight) TAIL_PRESSURE_WEIGHT="$2"; shift 2 ;;
        --tail-delta-clip) TAIL_DELTA_CLIP="$2"; shift 2 ;;
        --tail-potential-discount) TAIL_POTENTIAL_DISCOUNT="$2"; shift 2 ;;
        --tail-potential-topk) TAIL_POTENTIAL_TOPK="$2"; shift 2 ;;
        --tail-service-target) TAIL_SERVICE_TARGET="$2"; shift 2 ;;
        --tail-ttl-guard-ms) TAIL_TTL_GUARD_MS="$2"; shift 2 ;;
        --expiry-delta-weight) EXPIRY_DELTA_WEIGHT="$2"; shift 2 ;;
        --expiry-pressure-weight) EXPIRY_PRESSURE_WEIGHT="$2"; shift 2 ;;
        --expiry-delta-clip) EXPIRY_DELTA_CLIP="$2"; shift 2 ;;
        --expiry-potential-discount) EXPIRY_POTENTIAL_DISCOUNT="$2"; shift 2 ;;
        --expiry-potential-topk) EXPIRY_POTENTIAL_TOPK="$2"; shift 2 ;;
        --expiry-ttl-guard-ms) EXPIRY_TTL_GUARD_MS="$2"; shift 2 ;;
        --expiry-reward-window-steps) EXPIRY_REWARD_WINDOW_STEPS="$2"; shift 2 ;;
        --expiry-reward-offered-bytes-per-tti) EXPIRY_REWARD_OFFERED_BYTES_PER_TTI="$2"; shift 2 ;;
        --rescue-credit-weight) RESCUE_CREDIT_WEIGHT="$2"; shift 2 ;;
        --rescue-miss-weight) RESCUE_MISS_WEIGHT="$2"; shift 2 ;;
        --rescue-topk) RESCUE_TOPK="$2"; shift 2 ;;
        --rescue-service-target) RESCUE_SERVICE_TARGET="$2"; shift 2 ;;
        --rescue-ttl-guard-ms) RESCUE_TTL_GUARD_MS="$2"; shift 2 ;;
        --rescue-max-score) RESCUE_MAX_SCORE="$2"; shift 2 ;;
        --rescue-goodput-bytes-scale) RESCUE_GOODPUT_BYTES_SCALE="$2"; shift 2 ;;
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
if ! [[ "${ROLLOUT_WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "--rollout-warmup-steps must be a non-negative integer" >&2
    exit 1
fi
if [[ "${INCLUDE_PFQ_SCORE_ANCHOR}" != "0" && "${INCLUDE_PFQ_SCORE_ANCHOR}" != "1" ]]; then
    echo "--include-pfq-score-anchor must be 0 or 1" >&2
    exit 1
fi
if [[ "${INCLUDE_CANDIDATE_RISK_FEATURES}" != "0" && "${INCLUDE_CANDIDATE_RISK_FEATURES}" != "1" ]]; then
    echo "--include-candidate-risk-features must be 0 or 1" >&2
    exit 1
fi
if [[ "${WRITE_DIAGNOSTICS}" != "0" && "${WRITE_DIAGNOSTICS}" != "1" ]]; then
    echo "--write-diagnostics must be 0 or 1" >&2
    exit 1
fi
if [[ "${CANDIDATE_CHECKPOINT_REQUIRE_GATE}" != "0" && "${CANDIDATE_CHECKPOINT_REQUIRE_GATE}" != "1" ]]; then
    echo "--candidate-checkpoint-require-gate must be 0 or 1" >&2
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
if [[ "${EXPIRY_REWARD_WINDOW_STEPS}" != "0" && "${EXPIRY_REWARD_OFFERED_BYTES_PER_TTI}" == "0.0" ]]; then
    EXPIRY_REWARD_OFFERED_BYTES_PER_TTI="$(
        awk -v n="${TOTAL_UE_COUNT}" -v p="${PACKET_SIZE_BYTES}" -v r="${TRAFFIC_ARRIVAL_RATE}" \
            'BEGIN { printf "%.6f", n * p * r }'
    )"
fi
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
    "CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1"
)
if [[ "${TEACHER_MODE}" == "pfq_passthrough" || "${IMITATION_EPISODES}" != "0" ]]; then
    SIM_ENV+=("CUMAC_ONLINE_EXPORT_TEACHER_ACTION=1")
fi
SIM_REWARD_MODE="goodput_only"
case "${REWARD_MODE}" in
    stageb_blankaware|stageb_tailguard_simple|stageb_tailcredit_v4|stageb_tailcredit_expiry_v5|stageb_tailcredit_expiry_rescue_v6)
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
echo "[Stage-B HGraph] cell_radius_m=${CELL_RADIUS} site_spacing_m=$((2 * CELL_RADIUS)) ue_per_cell=${UE_PER_CELL} total_ue_count=${TOTAL_UE_COUNT}"
echo "[Stage-B HGraph] topology_seed_mode=${TOPOLOGY_SEED_MODE} topology_seed=${TOPOLOGY_SEED} seed_list=${SEED_LIST:-none}"
echo "[Stage-B HGraph] episode_horizon=${EPISODE_HORIZON} rollout_steps=${ROLLOUT_STEPS} rollout_warmup_steps=${ROLLOUT_WARMUP_STEPS} iterations=${ITERATIONS}"
echo "[Stage-B HGraph] print_every_iters=${PRINT_EVERY_ITERS} rollout_log_every_steps=${ROLLOUT_LOG_EVERY_STEPS} update_log_every_minibatches=${UPDATE_LOG_EVERY_MINIBATCHES} write_diagnostics=${WRITE_DIAGNOSTICS}"
echo "[Stage-B HGraph] ppo_preset=${PPO_PRESET:-none}"
echo "[Stage-B HGraph] reward_mode=${REWARD_MODE} init_checkpoint=${INIT_CHECKPOINT:-none} precoding=${PRECODING}"
echo "[Stage-B HGraph] best_metric=${BEST_METRIC} best_min_iter=${BEST_MIN_ITER} best_service_gap_penalty_mbps=${BEST_SERVICE_GAP_PENALTY_MBPS} best_unserved_ue_target=${BEST_UNSERVED_UE_TARGET} best_unserved_penalty_mbps=${BEST_UNSERVED_PENALTY_MBPS} best_tb_err_target=${BEST_TB_ERR_TARGET} best_packet_rate_bonus_mbps=${BEST_PACKET_RATE_BONUS_MBPS} best_packet_per_packet_rate_bonus_mbps=${BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS} best_packet_delay_penalty_mbps=${BEST_PACKET_DELAY_PENALTY_MBPS}"
echo "[Stage-B HGraph] candidate_checkpoint_interval=${CANDIDATE_CHECKPOINT_INTERVAL} candidate_checkpoint_require_gate=${CANDIDATE_CHECKPOINT_REQUIRE_GATE}"
echo "[Stage-B HGraph] best_hard_gates tb_err_max=${BEST_HARD_TB_ERR_MAX} expiry_rate_max=${BEST_HARD_EXPIRY_RATE_MAX} service_p10_min=${BEST_HARD_SERVICE_P10_MIN} unserved_ue_max=${BEST_HARD_UNSERVED_UE_MAX} backlog_p90_max=${BEST_HARD_BACKLOG_P90_MAX}"
echo "[Stage-B HGraph] best_rate_tail_gates packet_rate_min_mbps=${BEST_HARD_PACKET_RATE_MIN_MBPS} packet_per_packet_rate_min_mbps=${BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS} packet_delay_max_ms=${BEST_HARD_PACKET_DELAY_MAX_MS}"
echo "[Stage-B HGraph] best_ue_rate_proxy bonus_mbps=${BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS} p10_gap_penalty_mbps=${BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS} p10_target_mbps=${BEST_UE_P10_RATE_PROXY_TARGET_MBPS} macro_min_mbps=${BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS} p10_min_mbps=${BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS}"
echo "[Stage-B HGraph] best_ue_packet_rate bonus_mbps=${BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS} p10_gap_penalty_mbps=${BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS} p10_target_mbps=${BEST_UE_P10_PACKET_RATE_TARGET_MBPS} macro_min_mbps=${BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS} p10_min_mbps=${BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS}"
echo "[Stage-B HGraph] best_multi_seed window_iters=${BEST_MULTI_SEED_WINDOW_ITERS} min_covered_seeds=${BEST_MULTI_SEED_MIN_COVERED_SEEDS}"
echo "[Stage-B HGraph] imitation_episodes=${IMITATION_EPISODES} teacher_mode=${TEACHER_MODE} teacher_aux_coef=${TEACHER_AUX_COEF} teacher_aux_min_coef=${TEACHER_AUX_MIN_COEF} teacher_aux_decay_iters=${TEACHER_AUX_DECAY_ITERS} teacher_soft_target=${TEACHER_SOFT_TARGET}"
echo "[Stage-B HGraph] candidate_mode=${CANDIDATE_MODE} ue_prg_topk=${UE_PRG_TOPK} diversity_extra=${UE_PRG_DIVERSITY_EXTRA} max_prg_candidates=${MAX_PRG_CANDIDATES} action_head_mode=${ACTION_HEAD_MODE} include_pfq_score_anchor=${INCLUDE_PFQ_SCORE_ANCHOR} include_candidate_risk_features=${INCLUDE_CANDIDATE_RISK_FEATURES} candidate_blank_logit_bias=${CANDIDATE_BLANK_LOGIT_BIAS}"
echo "[Stage-B HGraph] candidate_success_logit_scale=${CANDIDATE_SUCCESS_LOGIT_SCALE} candidate_blank_success_scale=${CANDIDATE_BLANK_SUCCESS_SCALE} candidate_pfq_anchor_logit_coef=${CANDIDATE_PFQ_ANCHOR_LOGIT_COEF} candidate_success_aux_coef=${CANDIDATE_SUCCESS_AUX_COEF} candidate_success_calib_coef=${CANDIDATE_SUCCESS_CALIB_COEF}"
echo "[Stage-B HGraph] candidate_seq_state_dim=${CANDIDATE_SEQ_STATE_DIM} candidate_seq_state_coef=${CANDIDATE_SEQ_STATE_COEF} candidate_seq_fail_risk_feature_offset=${CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET} candidate_seq_train_only=${CANDIDATE_SEQ_TRAIN_ONLY}"
echo "[Stage-B HGraph] candidate_success_max_calib_coef=${CANDIDATE_SUCCESS_MAX_CALIB_COEF} candidate_success_target_eps=${CANDIDATE_SUCCESS_TARGET_EPS} pairwise_rank_coef=${CANDIDATE_PAIRWISE_RANK_COEF} pairwise_margin=${CANDIDATE_PAIRWISE_RANK_MARGIN} pairwise_min_gap=${CANDIDATE_PAIRWISE_RANK_MIN_GAP}"
echo "[Stage-B HGraph] pairwise_weights success=${CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT} service=${CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT} risk=${CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT} fail_risk=${CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT} rescue=${CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT} rescue_aux=${CANDIDATE_RESCUE_AUX_COEF} rescue_pos_weight=${CANDIDATE_RESCUE_POS_WEIGHT}"
echo "[Stage-B HGraph] candidate_decode_goodput_coef=${CANDIDATE_DECODE_GOODPUT_COEF} target_bytes=${CANDIDATE_DECODE_GOODPUT_TARGET_BYTES} max_target=${CANDIDATE_DECODE_GOODPUT_MAX_TARGET} pos_weight=${CANDIDATE_DECODE_GOODPUT_POS_WEIGHT}"
echo "[Stage-B HGraph] candidate_argmax_decode_coef=${CANDIDATE_ARGMAX_DECODE_COEF} pos_weight=${CANDIDATE_ARGMAX_DECODE_POS_WEIGHT} headroom_rank_mode=${CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE}"
echo "[Stage-B HGraph] candidate_blank_target_coef=${CANDIDATE_BLANK_TARGET_COEF} candidate_blank_target_threshold=${CANDIDATE_BLANK_TARGET_THRESHOLD} candidate_blank_target_margin=${CANDIDATE_BLANK_TARGET_MARGIN} candidate_blank_target_deadband=${CANDIDATE_BLANK_TARGET_DEADBAND} candidate_blank_target_pos_weight=${CANDIDATE_BLANK_TARGET_POS_WEIGHT} candidate_blank_target_fill_weight=${CANDIDATE_BLANK_TARGET_FILL_WEIGHT} candidate_blank_target_positive_only=${CANDIDATE_BLANK_TARGET_POSITIVE_ONLY}"
echo "[Stage-B HGraph] candidate_blank_target_service_guard=${CANDIDATE_BLANK_TARGET_SERVICE_GUARD} service_min_success=${CANDIDATE_BLANK_TARGET_SERVICE_MIN_SUCCESS} service_min_debt=${CANDIDATE_BLANK_TARGET_SERVICE_MIN_DEBT} service_min_hol_ms=${CANDIDATE_BLANK_TARGET_SERVICE_MIN_HOL_MS} service_max_scheduled_ratio=${CANDIDATE_BLANK_TARGET_SERVICE_MAX_SCHEDULED_RATIO}"
echo "[Stage-B HGraph] candidate_blank_target_service_fill_weight=${CANDIDATE_BLANK_TARGET_SERVICE_FILL_WEIGHT} service_fill_margin=${CANDIDATE_BLANK_TARGET_SERVICE_FILL_MARGIN} service_fill_min_success=${CANDIDATE_BLANK_TARGET_SERVICE_FILL_MIN_SUCCESS}"
echo "[Stage-B HGraph] safe_fill_max_per_cell=${SAFE_FILL_MAX_PER_CELL} safe_fill_min_quality_db=${SAFE_FILL_MIN_QUALITY_DB} safe_fill_max_quality_gap_db=${SAFE_FILL_MAX_QUALITY_GAP_DB} safe_fill_max_prg_risk=${SAFE_FILL_MAX_PRG_RISK}"
echo "[Stage-B HGraph] candidate_blank_budget_max_per_cell=${CANDIDATE_BLANK_BUDGET_MAX_PER_CELL} candidate_blank_budget_min_decision_logit=${CANDIDATE_BLANK_BUDGET_MIN_DECISION_LOGIT} candidate_decode_demand_slack_bytes=${CANDIDATE_DECODE_DEMAND_SLACK_BYTES} candidate_marginal_headroom_slack_bytes=${CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES}"
echo "[Stage-B HGraph] candidate_budgeted_select_mode=${CANDIDATE_BUDGETED_SELECT_MODE} hard_cap=${CANDIDATE_BUDGETED_HARD_CAP} usage_penalty=${CANDIDATE_BUDGETED_USAGE_PENALTY} overcap_penalty=${CANDIDATE_BUDGETED_OVERCAP_PENALTY} first_service_bonus=${CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS}"
echo "[Stage-B HGraph] candidate_tail_bias_coef=${CANDIDATE_BUDGETED_TAIL_BIAS_COEF} service_target=${CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET} ttl_guard_ms=${CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS} max=${CANDIDATE_BUDGETED_TAIL_BIAS_MAX}"
echo "[Stage-B HGraph] candidate_packet_tail_feature_enable=${CANDIDATE_PACKET_TAIL_FEATURE_ENABLE} rate_target_mbps=${CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS} delay_scale_ms=${CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS}"
echo "[Stage-B HGraph] candidate_rescue_bias_coef=${CANDIDATE_BUDGETED_RESCUE_BIAS_COEF} service_target=${CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET} ttl_guard_ms=${CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS} max=${CANDIDATE_BUDGETED_RESCUE_BIAS_MAX}"
echo "[Stage-B HGraph] candidate_completion_bonus=${CANDIDATE_BUDGETED_COMPLETION_BONUS} progress_weight=${CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT} max=${CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE}"
echo "[Stage-B HGraph] candidate_tail_quota_enable=${CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE} min_score=${CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE} topk_ues=${CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES} prgs_per_ue=${CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE} bonus=${CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS} order_bonus=${CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS} max_score=${CANDIDATE_BUDGETED_TAIL_QUOTA_MAX_SCORE} risk_max=${CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX}"
echo "[Stage-B HGraph] slot_repair_max_per_cell=${SLOT_REPAIR_MAX_PER_CELL} slot_repair_min_backlog_bytes=${SLOT_REPAIR_MIN_BACKLOG_BYTES} backlog_debt_weight=${BACKLOG_DEBT_WEIGHT}"
echo "[Stage-B HGraph] packet_rate_weight=${PACKET_RATE_WEIGHT} packet_per_packet_rate_weight=${PACKET_PER_PACKET_RATE_WEIGHT} packet_completion_weight=${PACKET_COMPLETION_WEIGHT} packet_delay_weight=${PACKET_DELAY_WEIGHT} packet_rate_scale_mbps=${PACKET_RATE_SCALE_MBPS} packet_per_packet_rate_scale_mbps=${PACKET_PER_PACKET_RATE_SCALE_MBPS} packet_delay_scale_ms=${PACKET_DELAY_SCALE_MS} aged_backlog_weight=${AGED_BACKLOG_WEIGHT}"
echo "[Stage-B HGraph] ue_rate_proxy macro_weight=${UE_MACRO_RATE_PROXY_WEIGHT} p10_gap_weight=${UE_P10_RATE_PROXY_GAP_WEIGHT} scale_mbps=${UE_RATE_PROXY_SCALE_MBPS} p10_target_mbps=${UE_P10_RATE_PROXY_TARGET_MBPS}"
echo "[Stage-B HGraph] ue_packet_rate macro_weight=${UE_MACRO_PACKET_RATE_WEIGHT} p10_gap_weight=${UE_P10_PACKET_RATE_GAP_WEIGHT} scale_mbps=${UE_PACKET_RATE_SCALE_MBPS} p10_target_mbps=${UE_P10_PACKET_RATE_TARGET_MBPS}"
echo "[Stage-B HGraph] unserved_ue_weight=${UNSERVED_UE_WEIGHT} service_gap_weight=${SERVICE_GAP_WEIGHT} service_rate_target=${SERVICE_RATE_TARGET}"
echo "[Stage-B HGraph] util_floor_weight=${UTIL_FLOOR_WEIGHT} util_floor_target=${UTIL_FLOOR_TARGET} util_floor_tb_err_guard=${UTIL_FLOOR_TB_ERR_GUARD} util_floor_tb_err_softness=${UTIL_FLOOR_TB_ERR_SOFTNESS}"
echo "[Stage-B HGraph] missed_safe_opportunity_weight=${MISSED_SAFE_OPPORTUNITY_WEIGHT} missed_safe_opportunity_scale=${MISSED_SAFE_OPPORTUNITY_SCALE}"
echo "[Stage-B HGraph] prg_efficiency_weight=${PRG_EFFICIENCY_WEIGHT} harmful_packing_weight=${HARMFUL_PACKING_WEIGHT}"
echo "[Stage-B HGraph] tb_err_target=${TB_ERR_TARGET} tb_err_over_target_weight=${TB_ERR_OVER_TARGET_WEIGHT} tb_err_goodput_gate_weight=${TB_ERR_GOODPUT_GATE_WEIGHT}"
echo "[Stage-B HGraph] tail_delta_weight=${TAIL_DELTA_WEIGHT} tail_pressure_weight=${TAIL_PRESSURE_WEIGHT} tail_delta_clip=${TAIL_DELTA_CLIP} tail_potential_discount=${TAIL_POTENTIAL_DISCOUNT} tail_topk=${TAIL_POTENTIAL_TOPK} tail_service_target=${TAIL_SERVICE_TARGET} tail_ttl_guard_ms=${TAIL_TTL_GUARD_MS}"
echo "[Stage-B HGraph] expiry_delta_weight=${EXPIRY_DELTA_WEIGHT} expiry_pressure_weight=${EXPIRY_PRESSURE_WEIGHT} expiry_delta_clip=${EXPIRY_DELTA_CLIP} expiry_potential_discount=${EXPIRY_POTENTIAL_DISCOUNT} expiry_topk=${EXPIRY_POTENTIAL_TOPK} expiry_ttl_guard_ms=${EXPIRY_TTL_GUARD_MS}"
echo "[Stage-B HGraph] expiry_reward_window_steps=${EXPIRY_REWARD_WINDOW_STEPS} expiry_reward_offered_bytes_per_tti=${EXPIRY_REWARD_OFFERED_BYTES_PER_TTI}"
echo "[Stage-B HGraph] rescue_credit_weight=${RESCUE_CREDIT_WEIGHT} rescue_miss_weight=${RESCUE_MISS_WEIGHT} rescue_topk=${RESCUE_TOPK} rescue_service_target=${RESCUE_SERVICE_TARGET} rescue_ttl_guard_ms=${RESCUE_TTL_GUARD_MS}"

CMD=(
    "${PYTHON_BIN}" -m "${PPO_MODULE}"
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
    --rollout-warmup-steps "${ROLLOUT_WARMUP_STEPS}"
    --iterations "${ITERATIONS}"
    --topology-seed "${TOPOLOGY_SEED}"
    --topology-seed-mode "${TOPOLOGY_SEED_MODE}"
    --seed-list "${SEED_LIST}"
    --out-dir "${OUT_DIR}"
    --ue-prg-topk "${UE_PRG_TOPK}"
    --ue-prg-diversity-extra "${UE_PRG_DIVERSITY_EXTRA}"
    --candidate-mode "${CANDIDATE_MODE}"
    --max-prg-candidates "${MAX_PRG_CANDIDATES}"
    --action-head-mode "${ACTION_HEAD_MODE}"
    --include-pfq-score-anchor "${INCLUDE_PFQ_SCORE_ANCHOR}"
    --include-candidate-risk-features "${INCLUDE_CANDIDATE_RISK_FEATURES}"
    --candidate-blank-logit-bias "${CANDIDATE_BLANK_LOGIT_BIAS}"
    --candidate-success-logit-scale "${CANDIDATE_SUCCESS_LOGIT_SCALE}"
    --candidate-blank-success-scale "${CANDIDATE_BLANK_SUCCESS_SCALE}"
    --candidate-pfq-anchor-logit-coef "${CANDIDATE_PFQ_ANCHOR_LOGIT_COEF}"
    --candidate-success-aux-coef "${CANDIDATE_SUCCESS_AUX_COEF}"
    --candidate-success-calib-coef "${CANDIDATE_SUCCESS_CALIB_COEF}"
    --candidate-success-max-calib-coef "${CANDIDATE_SUCCESS_MAX_CALIB_COEF}"
    --candidate-success-target-eps "${CANDIDATE_SUCCESS_TARGET_EPS}"
    --candidate-seq-state-dim "${CANDIDATE_SEQ_STATE_DIM}"
    --candidate-seq-state-coef "${CANDIDATE_SEQ_STATE_COEF}"
    --candidate-seq-fail-risk-feature-offset "${CANDIDATE_SEQ_FAIL_RISK_FEATURE_OFFSET}"
    --candidate-seq-train-only "${CANDIDATE_SEQ_TRAIN_ONLY}"
    --candidate-pairwise-rank-coef "${CANDIDATE_PAIRWISE_RANK_COEF}"
    --candidate-pairwise-rank-margin "${CANDIDATE_PAIRWISE_RANK_MARGIN}"
    --candidate-pairwise-rank-min-gap "${CANDIDATE_PAIRWISE_RANK_MIN_GAP}"
    --candidate-pairwise-rank-success-weight "${CANDIDATE_PAIRWISE_RANK_SUCCESS_WEIGHT}"
    --candidate-pairwise-rank-service-weight "${CANDIDATE_PAIRWISE_RANK_SERVICE_WEIGHT}"
    --candidate-pairwise-rank-risk-weight "${CANDIDATE_PAIRWISE_RANK_RISK_WEIGHT}"
    --candidate-pairwise-rank-fail-risk-weight "${CANDIDATE_PAIRWISE_RANK_FAIL_RISK_WEIGHT}"
    --candidate-pairwise-rank-rescue-weight "${CANDIDATE_PAIRWISE_RANK_RESCUE_WEIGHT}"
    --candidate-rescue-aux-coef "${CANDIDATE_RESCUE_AUX_COEF}"
    --candidate-rescue-pos-weight "${CANDIDATE_RESCUE_POS_WEIGHT}"
    --candidate-rescue-service-target "${CANDIDATE_RESCUE_SERVICE_TARGET}"
    --candidate-rescue-ttl-guard-ms "${CANDIDATE_RESCUE_TTL_GUARD_MS}"
    --candidate-rescue-max-target "${CANDIDATE_RESCUE_MAX_TARGET}"
    --candidate-decode-goodput-coef "${CANDIDATE_DECODE_GOODPUT_COEF}"
    --candidate-decode-goodput-target-bytes "${CANDIDATE_DECODE_GOODPUT_TARGET_BYTES}"
    --candidate-decode-goodput-max-target "${CANDIDATE_DECODE_GOODPUT_MAX_TARGET}"
    --candidate-decode-goodput-pos-weight "${CANDIDATE_DECODE_GOODPUT_POS_WEIGHT}"
    --candidate-argmax-decode-coef "${CANDIDATE_ARGMAX_DECODE_COEF}"
    --candidate-argmax-decode-pos-weight "${CANDIDATE_ARGMAX_DECODE_POS_WEIGHT}"
    --candidate-argmax-decode-headroom-rank-mode "${CANDIDATE_ARGMAX_DECODE_HEADROOM_RANK_MODE}"
    --candidate-blank-target-coef "${CANDIDATE_BLANK_TARGET_COEF}"
    --candidate-blank-target-threshold "${CANDIDATE_BLANK_TARGET_THRESHOLD}"
    --candidate-blank-target-margin "${CANDIDATE_BLANK_TARGET_MARGIN}"
    --candidate-blank-target-deadband "${CANDIDATE_BLANK_TARGET_DEADBAND}"
    --candidate-blank-target-pos-weight "${CANDIDATE_BLANK_TARGET_POS_WEIGHT}"
    --candidate-blank-target-fill-weight "${CANDIDATE_BLANK_TARGET_FILL_WEIGHT}"
    --candidate-blank-target-positive-only "${CANDIDATE_BLANK_TARGET_POSITIVE_ONLY}"
    --candidate-blank-target-service-guard "${CANDIDATE_BLANK_TARGET_SERVICE_GUARD}"
    --candidate-blank-target-service-min-success "${CANDIDATE_BLANK_TARGET_SERVICE_MIN_SUCCESS}"
    --candidate-blank-target-service-min-debt "${CANDIDATE_BLANK_TARGET_SERVICE_MIN_DEBT}"
    --candidate-blank-target-service-min-hol-ms "${CANDIDATE_BLANK_TARGET_SERVICE_MIN_HOL_MS}"
    --candidate-blank-target-service-max-scheduled-ratio "${CANDIDATE_BLANK_TARGET_SERVICE_MAX_SCHEDULED_RATIO}"
    --candidate-blank-target-service-fill-weight "${CANDIDATE_BLANK_TARGET_SERVICE_FILL_WEIGHT}"
    --candidate-blank-target-service-fill-margin "${CANDIDATE_BLANK_TARGET_SERVICE_FILL_MARGIN}"
    --candidate-blank-target-service-fill-min-success "${CANDIDATE_BLANK_TARGET_SERVICE_FILL_MIN_SUCCESS}"
    --safe-fill-max-per-cell "${SAFE_FILL_MAX_PER_CELL}"
    --safe-fill-min-quality-db "${SAFE_FILL_MIN_QUALITY_DB}"
    --safe-fill-max-quality-gap-db "${SAFE_FILL_MAX_QUALITY_GAP_DB}"
    --safe-fill-max-prg-risk "${SAFE_FILL_MAX_PRG_RISK}"
    --safe-fill-min-backlog-bytes "${SAFE_FILL_MIN_BACKLOG_BYTES}"
    --safe-fill-max-ue-tb-err "${SAFE_FILL_MAX_UE_TB_ERR}"
    --candidate-blank-budget-max-per-cell "${CANDIDATE_BLANK_BUDGET_MAX_PER_CELL}"
    --candidate-blank-budget-min-decision-logit "${CANDIDATE_BLANK_BUDGET_MIN_DECISION_LOGIT}"
    --candidate-decode-demand-slack-bytes "${CANDIDATE_DECODE_DEMAND_SLACK_BYTES}"
    --candidate-marginal-headroom-slack-bytes "${CANDIDATE_MARGINAL_HEADROOM_SLACK_BYTES}"
    --candidate-budgeted-select-mode "${CANDIDATE_BUDGETED_SELECT_MODE}"
    --candidate-budgeted-hard-cap "${CANDIDATE_BUDGETED_HARD_CAP}"
    --candidate-budgeted-usage-penalty "${CANDIDATE_BUDGETED_USAGE_PENALTY}"
    --candidate-budgeted-overcap-penalty "${CANDIDATE_BUDGETED_OVERCAP_PENALTY}"
    --candidate-budgeted-first-service-bonus "${CANDIDATE_BUDGETED_FIRST_SERVICE_BONUS}"
    --candidate-budgeted-tail-bias-coef "${CANDIDATE_BUDGETED_TAIL_BIAS_COEF}"
    --candidate-budgeted-tail-bias-service-target "${CANDIDATE_BUDGETED_TAIL_BIAS_SERVICE_TARGET}"
    --candidate-budgeted-tail-bias-ttl-guard-ms "${CANDIDATE_BUDGETED_TAIL_BIAS_TTL_GUARD_MS}"
    --candidate-budgeted-tail-bias-max "${CANDIDATE_BUDGETED_TAIL_BIAS_MAX}"
    --candidate-packet-tail-feature-enable "${CANDIDATE_PACKET_TAIL_FEATURE_ENABLE}"
    --candidate-packet-tail-rate-target-mbps "${CANDIDATE_PACKET_TAIL_RATE_TARGET_MBPS}"
    --candidate-packet-tail-delay-scale-ms "${CANDIDATE_PACKET_TAIL_DELAY_SCALE_MS}"
    --candidate-budgeted-rescue-bias-coef "${CANDIDATE_BUDGETED_RESCUE_BIAS_COEF}"
    --candidate-budgeted-rescue-bias-service-target "${CANDIDATE_BUDGETED_RESCUE_BIAS_SERVICE_TARGET}"
    --candidate-budgeted-rescue-bias-ttl-guard-ms "${CANDIDATE_BUDGETED_RESCUE_BIAS_TTL_GUARD_MS}"
    --candidate-budgeted-rescue-bias-max "${CANDIDATE_BUDGETED_RESCUE_BIAS_MAX}"
    --candidate-budgeted-completion-bonus "${CANDIDATE_BUDGETED_COMPLETION_BONUS}"
    --candidate-budgeted-completion-progress-weight "${CANDIDATE_BUDGETED_COMPLETION_PROGRESS_WEIGHT}"
    --candidate-budgeted-completion-max-score "${CANDIDATE_BUDGETED_COMPLETION_MAX_SCORE}"
    --candidate-budgeted-tail-quota-enable "${CANDIDATE_BUDGETED_TAIL_QUOTA_ENABLE}"
    --candidate-budgeted-tail-quota-min-score "${CANDIDATE_BUDGETED_TAIL_QUOTA_MIN_SCORE}"
    --candidate-budgeted-tail-quota-topk-ues "${CANDIDATE_BUDGETED_TAIL_QUOTA_TOPK_UES}"
    --candidate-budgeted-tail-quota-prgs-per-ue "${CANDIDATE_BUDGETED_TAIL_QUOTA_PRGS_PER_UE}"
    --candidate-budgeted-tail-quota-bonus "${CANDIDATE_BUDGETED_TAIL_QUOTA_BONUS}"
    --candidate-budgeted-tail-quota-order-bonus "${CANDIDATE_BUDGETED_TAIL_QUOTA_ORDER_BONUS}"
    --candidate-budgeted-tail-quota-max-score "${CANDIDATE_BUDGETED_TAIL_QUOTA_MAX_SCORE}"
    --candidate-budgeted-tail-quota-risk-max "${CANDIDATE_BUDGETED_TAIL_QUOTA_RISK_MAX}"
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
    --target-kl-min-updates "${TARGET_KL_MIN_UPDATES}"
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
    --write-diagnostics "${WRITE_DIAGNOSTICS}"
    --init-checkpoint "${INIT_CHECKPOINT}"
    --imitation-episodes "${IMITATION_EPISODES}"
    --imitation-max-steps "${IMITATION_MAX_STEPS}"
    --imitation-lr "${IMITATION_LR}"
    --teacher-mode "${TEACHER_MODE}"
    --teacher-aux-coef "${TEACHER_AUX_COEF}"
    --teacher-aux-min-coef "${TEACHER_AUX_MIN_COEF}"
    --teacher-aux-decay-iters "${TEACHER_AUX_DECAY_ITERS}"
    --teacher-soft-target "${TEACHER_SOFT_TARGET}"
    --teacher-soft-temperature "${TEACHER_SOFT_TEMPERATURE}"
    --teacher-soft-hit-bonus "${TEACHER_SOFT_HIT_BONUS}"
    --teacher-soft-blank-bonus "${TEACHER_SOFT_BLANK_BONUS}"
    --teacher-soft-blank-safe-penalty "${TEACHER_SOFT_BLANK_SAFE_PENALTY}"
    --teacher-soft-blank-no-safe-bonus "${TEACHER_SOFT_BLANK_NO_SAFE_BONUS}"
    --reward-mode "${REWARD_MODE}"
    --best-metric "${BEST_METRIC}"
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}"
    --early-stop-warmup-iters "${EARLY_STOP_WARMUP_ITERS}"
    --best-min-iter "${BEST_MIN_ITER}"
    --candidate-checkpoint-interval "${CANDIDATE_CHECKPOINT_INTERVAL}"
    --candidate-checkpoint-require-gate "${CANDIDATE_CHECKPOINT_REQUIRE_GATE}"
    --best-service-gap-penalty-mbps "${BEST_SERVICE_GAP_PENALTY_MBPS}"
    --best-unserved-ue-target "${BEST_UNSERVED_UE_TARGET}"
    --best-unserved-penalty-mbps "${BEST_UNSERVED_PENALTY_MBPS}"
    --best-tb-err-target "${BEST_TB_ERR_TARGET}"
    --best-tb-err-penalty-mbps "${BEST_TB_ERR_PENALTY_MBPS}"
    --best-expiry-penalty-mbps "${BEST_EXPIRY_PENALTY_MBPS}"
    --best-packet-rate-bonus-mbps "${BEST_PACKET_RATE_BONUS_MBPS}"
    --best-packet-per-packet-rate-bonus-mbps "${BEST_PACKET_PER_PACKET_RATE_BONUS_MBPS}"
    --best-packet-delay-penalty-mbps "${BEST_PACKET_DELAY_PENALTY_MBPS}"
    --best-ue-macro-rate-proxy-bonus-mbps "${BEST_UE_MACRO_RATE_PROXY_BONUS_MBPS}"
    --best-ue-p10-rate-proxy-gap-penalty-mbps "${BEST_UE_P10_RATE_PROXY_GAP_PENALTY_MBPS}"
    --best-ue-p10-rate-proxy-target-mbps "${BEST_UE_P10_RATE_PROXY_TARGET_MBPS}"
    --best-ue-macro-packet-rate-bonus-mbps "${BEST_UE_MACRO_PACKET_RATE_BONUS_MBPS}"
    --best-ue-p10-packet-rate-gap-penalty-mbps "${BEST_UE_P10_PACKET_RATE_GAP_PENALTY_MBPS}"
    --best-ue-p10-packet-rate-target-mbps "${BEST_UE_P10_PACKET_RATE_TARGET_MBPS}"
    --best-backlog-p90-penalty-mbps "${BEST_BACKLOG_P90_PENALTY_MBPS}"
    --best-hard-tb-err-max "${BEST_HARD_TB_ERR_MAX}"
    --best-hard-expiry-rate-max "${BEST_HARD_EXPIRY_RATE_MAX}"
    --best-hard-service-p10-min "${BEST_HARD_SERVICE_P10_MIN}"
    --best-hard-unserved-ue-max "${BEST_HARD_UNSERVED_UE_MAX}"
    --best-hard-backlog-p90-max "${BEST_HARD_BACKLOG_P90_MAX}"
    --best-hard-packet-rate-min-mbps "${BEST_HARD_PACKET_RATE_MIN_MBPS}"
    --best-hard-packet-per-packet-rate-min-mbps "${BEST_HARD_PACKET_PER_PACKET_RATE_MIN_MBPS}"
    --best-hard-packet-delay-max-ms "${BEST_HARD_PACKET_DELAY_MAX_MS}"
    --best-hard-ue-macro-rate-proxy-min-mbps "${BEST_HARD_UE_MACRO_RATE_PROXY_MIN_MBPS}"
    --best-hard-ue-p10-rate-proxy-min-mbps "${BEST_HARD_UE_P10_RATE_PROXY_MIN_MBPS}"
    --best-hard-ue-macro-packet-rate-min-mbps "${BEST_HARD_UE_MACRO_PACKET_RATE_MIN_MBPS}"
    --best-hard-ue-p10-packet-rate-min-mbps "${BEST_HARD_UE_P10_PACKET_RATE_MIN_MBPS}"
    --best-multi-seed-window-iters "${BEST_MULTI_SEED_WINDOW_ITERS}"
    --best-multi-seed-min-covered-seeds "${BEST_MULTI_SEED_MIN_COVERED_SEEDS}"
    --goodput-weight "${GOODPUT_WEIGHT}"
    --tb-err-weight "${TB_ERR_WEIGHT}"
    --expiry-weight "${EXPIRY_WEIGHT}"
    --conflict-weight "${CONFLICT_WEIGHT}"
    --ici-weight "${ICI_WEIGHT}"
    --urgent-backlog-weight "${URGENT_BACKLOG_WEIGHT}"
    --urgent-ttl-ms "${URGENT_TTL_MS}"
    --backlog-debt-weight "${BACKLOG_DEBT_WEIGHT}"
    --packet-rate-weight "${PACKET_RATE_WEIGHT}"
    --packet-per-packet-rate-weight "${PACKET_PER_PACKET_RATE_WEIGHT}"
    --packet-completion-weight "${PACKET_COMPLETION_WEIGHT}"
    --packet-delay-weight "${PACKET_DELAY_WEIGHT}"
    --packet-rate-scale-mbps "${PACKET_RATE_SCALE_MBPS}"
    --packet-per-packet-rate-scale-mbps "${PACKET_PER_PACKET_RATE_SCALE_MBPS}"
    --packet-completion-scale "${PACKET_COMPLETION_SCALE}"
    --packet-delay-scale-ms "${PACKET_DELAY_SCALE_MS}"
    --ue-macro-rate-proxy-weight "${UE_MACRO_RATE_PROXY_WEIGHT}"
    --ue-p10-rate-proxy-gap-weight "${UE_P10_RATE_PROXY_GAP_WEIGHT}"
    --ue-rate-proxy-scale-mbps "${UE_RATE_PROXY_SCALE_MBPS}"
    --ue-p10-rate-proxy-target-mbps "${UE_P10_RATE_PROXY_TARGET_MBPS}"
    --ue-macro-packet-rate-weight "${UE_MACRO_PACKET_RATE_WEIGHT}"
    --ue-p10-packet-rate-gap-weight "${UE_P10_PACKET_RATE_GAP_WEIGHT}"
    --ue-packet-rate-scale-mbps "${UE_PACKET_RATE_SCALE_MBPS}"
    --ue-p10-packet-rate-target-mbps "${UE_P10_PACKET_RATE_TARGET_MBPS}"
    --aged-backlog-weight "${AGED_BACKLOG_WEIGHT}"
    --unserved-ue-weight "${UNSERVED_UE_WEIGHT}"
    --service-gap-weight "${SERVICE_GAP_WEIGHT}"
    --service-rate-target "${SERVICE_RATE_TARGET}"
    --util-floor-weight "${UTIL_FLOOR_WEIGHT}"
    --util-floor-target "${UTIL_FLOOR_TARGET}"
    --util-floor-tb-err-guard "${UTIL_FLOOR_TB_ERR_GUARD}"
    --util-floor-tb-err-softness "${UTIL_FLOOR_TB_ERR_SOFTNESS}"
    --missed-safe-opportunity-weight "${MISSED_SAFE_OPPORTUNITY_WEIGHT}"
    --missed-safe-opportunity-scale "${MISSED_SAFE_OPPORTUNITY_SCALE}"
    --prg-efficiency-weight "${PRG_EFFICIENCY_WEIGHT}"
    --harmful-packing-weight "${HARMFUL_PACKING_WEIGHT}"
    --tb-err-target "${TB_ERR_TARGET}"
    --tb-err-over-target-weight "${TB_ERR_OVER_TARGET_WEIGHT}"
    --tb-err-goodput-gate-weight "${TB_ERR_GOODPUT_GATE_WEIGHT}"
    --tail-delta-weight "${TAIL_DELTA_WEIGHT}"
    --tail-pressure-weight "${TAIL_PRESSURE_WEIGHT}"
    --tail-delta-clip "${TAIL_DELTA_CLIP}"
    --tail-potential-discount "${TAIL_POTENTIAL_DISCOUNT}"
    --tail-potential-topk "${TAIL_POTENTIAL_TOPK}"
    --tail-service-target "${TAIL_SERVICE_TARGET}"
    --tail-ttl-guard-ms "${TAIL_TTL_GUARD_MS}"
    --expiry-delta-weight "${EXPIRY_DELTA_WEIGHT}"
    --expiry-pressure-weight "${EXPIRY_PRESSURE_WEIGHT}"
    --expiry-delta-clip "${EXPIRY_DELTA_CLIP}"
    --expiry-potential-discount "${EXPIRY_POTENTIAL_DISCOUNT}"
    --expiry-potential-topk "${EXPIRY_POTENTIAL_TOPK}"
    --expiry-ttl-guard-ms "${EXPIRY_TTL_GUARD_MS}"
    --expiry-reward-window-steps "${EXPIRY_REWARD_WINDOW_STEPS}"
    --expiry-reward-offered-bytes-per-tti "${EXPIRY_REWARD_OFFERED_BYTES_PER_TTI}"
    --rescue-credit-weight "${RESCUE_CREDIT_WEIGHT}"
    --rescue-miss-weight "${RESCUE_MISS_WEIGHT}"
    --rescue-topk "${RESCUE_TOPK}"
    --rescue-service-target "${RESCUE_SERVICE_TARGET}"
    --rescue-ttl-guard-ms "${RESCUE_TTL_GUARD_MS}"
    --rescue-max-score "${RESCUE_MAX_SCORE}"
    --rescue-goodput-bytes-scale "${RESCUE_GOODPUT_BYTES_SCALE}"
)
for kv in "${SIM_ENV[@]}"; do
    CMD+=(--sim-env "${kv}")
done

echo "[Stage-B HGraph] running online PPO fine-tuning"
"${CMD[@]}"
