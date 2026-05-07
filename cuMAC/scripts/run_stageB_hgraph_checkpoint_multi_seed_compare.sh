#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARCH="$(uname -m)"
PER_SEED_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
EVAL_SCRIPT="${ROOT_DIR}/training/stageb_hgraph/eval_checkpoint_online.py"
EVAL_MODULE="training.stageb_hgraph.eval_checkpoint_online"
AGGREGATE_SCRIPT="${ROOT_DIR}/cuMAC/scripts/aggregate_model_kpi_compare.py"

BUILD_METHOD="phase4"
USER_TAG=""
COMPARE_OUTPUT_DIR=""
SEED_LIST="41,42,43"
CHECKPOINT_PATH=""
MODEL_LABEL=""
BASELINE_MEAN_ROOT=""
BASELINE_MEAN_CSV=""
DECODE_MODE="argmax"
SAMPLE_SEED=42
EVAL_EPISODE_HORIZON=4000
EVAL_DEVICE="auto"
EVAL_PRINT_EVERY_STEPS=500
EVAL_UE_PRG_TOPK="auto"
EVAL_UE_PRG_DIVERSITY_EXTRA="auto"
EVAL_TB_ERR_EMA_ALPHA="auto"
EVAL_SLOT_REPAIR_MAX_PER_CELL="auto"
EVAL_SLOT_REPAIR_MIN_BACKLOG_BYTES="auto"
EVAL_SAVE_STEPS=0
EVAL_STATS_WARMUP_STEPS=0
PARALLEL_SEEDS=1
USER_EXEC_MODE=""
IGNORED_TOPOLOGY_SEED=""
FORWARD_ARGS=()
SIM_ENVS=("CUMAC_ONLINE_EXPORT_POST_EQ_SINR=1" "LD_LIBRARY_PATH=${ROOT_DIR}/build.${ARCH}/cuMAC/src:${LD_LIBRARY_PATH:-}")
EVAL_ENVS=("PYTHONUNBUFFERED=1")

usage() {
    cat <<EOF
Run a fixed HGraph checkpoint across multiple topology seeds, then aggregate KPI means.

Usage:
  $(basename "$0") [options]

Behavior:
  1) For each topology seed in --seed-list, run ${PER_SEED_SCRIPT##*/} with --online-bridge 1
  2) Connect ${EVAL_MODULE} to the online socket and drive the checkpoint policy
  3) Collect per-seed scenario kpi_summary.json outputs from the main experiment
  4) Aggregate mean/std/min/max model metrics per scenario
  5) If RR/PFQ mean CSV is provided, emit a unified RR/PFQ/HGraph compare table

Wrapper-specific options:
  --seed-list <csv>                 Topology seeds to run, e.g. 41,42,43 (default: ${SEED_LIST})
  --checkpoint-path <path>          Required HGraph checkpoint path
  --model-label <name>              Optional label for output columns/files
  --baseline-mean-root <dir>        Optional RR/PFQ aggregate root or scenario dir
  --baseline-mean-csv <path>        Optional RR/PFQ mean CSV for direct compare
  --decode-mode <m>                 argmax | sample (default: ${DECODE_MODE}; deployment-style deterministic path)
  --sample-seed <n>                 Sample RNG seed for decode-mode=sample (default: ${SAMPLE_SEED})
  --eval-episode-horizon <n>        Episode horizon passed to the online bridge evaluator (default: ${EVAL_EPISODE_HORIZON})
  --eval-device <m>                 auto | cpu | cuda (default: ${EVAL_DEVICE})
  --eval-print-every-steps <n>      Evaluator heartbeat interval (default: ${EVAL_PRINT_EVERY_STEPS})
  --eval-ue-prg-topk <n>            Evaluator UE-PRG top-k. "auto" inherits checkpoint args (default: ${EVAL_UE_PRG_TOPK})
  --eval-ue-prg-diversity-extra <n> Evaluator diversity candidates. "auto" inherits checkpoint args (default: ${EVAL_UE_PRG_DIVERSITY_EXTRA})
  --eval-tb-err-ema-alpha <v>       Evaluator TB-error EMA alpha. "auto" inherits checkpoint args (default: ${EVAL_TB_ERR_EMA_ALPHA})
  --eval-slot-repair-max-per-cell <n> Evaluator slot repair cap. "auto" inherits checkpoint args (default: ${EVAL_SLOT_REPAIR_MAX_PER_CELL})
  --eval-slot-repair-min-backlog-bytes <v> Evaluator slot repair backlog threshold. "auto" inherits checkpoint args (default: ${EVAL_SLOT_REPAIR_MIN_BACKLOG_BYTES})
  --eval-save-steps <0|1>           Write per-TTI eval_steps.jsonl (default: ${EVAL_SAVE_STEPS})
  --eval-stats-warmup-steps <n>     Ignore first N evaluator steps in eval stats_mean_* fields (default: ${EVAL_STATS_WARMUP_STEPS})
  --parallel-seeds <n>              Run up to n seed/evaluator pairs at once after one build-only step (default: ${PARALLEL_SEEDS})
  --sim-env <KEY=VALUE>             Extra simulator env var, may be repeated
  --eval-env <KEY=VALUE>            Extra evaluator env var, may be repeated
  --compare-output-dir <dir>        Base output directory for manifests and aggregated artifacts
  --build-method <m>                Build method for the first seed; later seeds use skip
  --tag <name>                      Optional base tag; per-seed runs expand it to <tag>_s<seed>
  --topology-seed <n>               Ignored by this wrapper; use --seed-list instead
  -h, --help                        Show this help

All other options are forwarded to ${PER_SEED_SCRIPT##*/}.
EOF
}

parse_seed_list() {
    local raw="$1"
    local normalized="${raw//,/ }"
    local token
    local seeds=()
    for token in ${normalized}; do
        if ! [[ "${token}" =~ ^[0-9]+$ ]]; then
            echo "Invalid seed in --seed-list: ${token}" >&2
            exit 1
        fi
        seeds+=("${token}")
    done
    if [[ ${#seeds[@]} -eq 0 ]]; then
        echo "--seed-list must not be empty" >&2
        exit 1
    fi
    printf '%s\n' "${seeds[@]}"
}

forward_arg_value() {
    local key="$1"
    shift || true
    local idx=1
    while [[ $# -gt 0 ]]; do
        local cur="$1"
        local nxt="${2:-}"
        if [[ "${cur}" == "${key}" ]]; then
            printf '%s\n' "${nxt}"
            return 0
        fi
        shift 2 || break
        idx=$((idx + 1))
    done
    return 1
}

normalize_label() {
    local raw="$1"
    local safe
    safe="$(echo "${raw}" | tr -c '[:alnum:]' '_' | sed 's/^_*//; s/_*$//')"
    if [[ -z "${safe}" ]]; then
        safe="hgraph"
    fi
    printf '%s\n' "${safe}"
}

resolve_path() {
    local raw="$1"
    if [[ "${raw}" == /* ]]; then
        printf '%s\n' "${raw}"
    else
        printf '%s\n' "${ROOT_DIR}/${raw}"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-list) SEED_LIST="$2"; shift 2 ;;
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --model-label) MODEL_LABEL="$2"; shift 2 ;;
        --baseline-mean-root) BASELINE_MEAN_ROOT="$2"; shift 2 ;;
        --baseline-mean-csv) BASELINE_MEAN_CSV="$2"; shift 2 ;;
        --decode-mode) DECODE_MODE="$2"; shift 2 ;;
        --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
        --eval-episode-horizon) EVAL_EPISODE_HORIZON="$2"; shift 2 ;;
        --eval-device) EVAL_DEVICE="$2"; shift 2 ;;
        --eval-print-every-steps) EVAL_PRINT_EVERY_STEPS="$2"; shift 2 ;;
        --eval-ue-prg-topk|--ue-prg-topk) EVAL_UE_PRG_TOPK="$2"; shift 2 ;;
        --eval-ue-prg-diversity-extra|--ue-prg-diversity-extra) EVAL_UE_PRG_DIVERSITY_EXTRA="$2"; shift 2 ;;
        --eval-tb-err-ema-alpha|--tb-err-ema-alpha) EVAL_TB_ERR_EMA_ALPHA="$2"; shift 2 ;;
        --eval-slot-repair-max-per-cell|--slot-repair-max-per-cell) EVAL_SLOT_REPAIR_MAX_PER_CELL="$2"; shift 2 ;;
        --eval-slot-repair-min-backlog-bytes|--slot-repair-min-backlog-bytes) EVAL_SLOT_REPAIR_MIN_BACKLOG_BYTES="$2"; shift 2 ;;
        --eval-save-steps) EVAL_SAVE_STEPS="$2"; shift 2 ;;
        --eval-stats-warmup-steps|--stats-warmup-steps) EVAL_STATS_WARMUP_STEPS="$2"; shift 2 ;;
        --parallel-seeds) PARALLEL_SEEDS="$2"; shift 2 ;;
        --sim-env) SIM_ENVS+=("$2"); shift 2 ;;
        --eval-env) EVAL_ENVS+=("$2"); shift 2 ;;
        --compare-output-dir) COMPARE_OUTPUT_DIR="$2"; shift 2 ;;
        --build-method) BUILD_METHOD="$2"; shift 2 ;;
        --tag) USER_TAG="$2"; shift 2 ;;
        --topology-seed) IGNORED_TOPOLOGY_SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for option: $1" >&2
                exit 1
            fi
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -x "${PER_SEED_SCRIPT}" ]]; then
    echo "Missing per-seed experiment script: ${PER_SEED_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Missing eval script: ${EVAL_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate script: ${AGGREGATE_SCRIPT}" >&2
    exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "--checkpoint-path is required" >&2
    exit 1
fi

forward_exec_mode="$(forward_arg_value --exec-mode "${FORWARD_ARGS[@]}" || true)"
if [[ "${forward_exec_mode}" == "gpu" ]]; then
    echo "[hgraph multi-seed] ERROR: --exec-mode=gpu is not supported for HGraph online-bridge evaluation." >&2
    echo "[hgraph multi-seed] Use --exec-mode both instead. Native RR/PFQ baselines may use gpu-only, but HGraph custom UE+PRG mode requires both." >&2
    exit 1
fi

CHECKPOINT_PATH="$(resolve_path "${CHECKPOINT_PATH}")"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ -n "${BASELINE_MEAN_ROOT}" ]]; then
    BASELINE_MEAN_ROOT="$(resolve_path "${BASELINE_MEAN_ROOT}")"
fi
if [[ -n "${BASELINE_MEAN_CSV}" ]]; then
    BASELINE_MEAN_CSV="$(resolve_path "${BASELINE_MEAN_CSV}")"
fi

if [[ -z "${MODEL_LABEL}" ]]; then
    MODEL_LABEL="$(basename "${CHECKPOINT_PATH}")"
    MODEL_LABEL="${MODEL_LABEL%.*}"
fi
MODEL_LABEL="$(normalize_label "${MODEL_LABEL}")"
DECODE_MODE="$(echo "${DECODE_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${DECODE_MODE}" != "argmax" && "${DECODE_MODE}" != "sample" ]]; then
    echo "--decode-mode must be argmax or sample" >&2
    exit 1
fi
if ! [[ "${PARALLEL_SEEDS}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL_SEEDS}" -lt 1 ]]; then
    echo "--parallel-seeds must be a positive integer" >&2
    exit 1
fi
if [[ "${EVAL_SAVE_STEPS}" != "0" && "${EVAL_SAVE_STEPS}" != "1" ]]; then
    echo "--eval-save-steps must be 0 or 1" >&2
    exit 1
fi
if ! [[ "${EVAL_STATS_WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "--eval-stats-warmup-steps must be a non-negative integer" >&2
    exit 1
fi

mapfile -t SEEDS < <(parse_seed_list "${SEED_LIST}")
timestamp="$(date +%Y%m%d_%H%M%S)"
default_base_tag="${MODEL_LABEL}_multiseed_compare"
base_tag="${USER_TAG:-${default_base_tag}}"
safe_base_tag="$(normalize_label "${base_tag}")"

if [[ -z "${COMPARE_OUTPUT_DIR}" ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/output/stageB_hgraph_multiseed_compare_${safe_base_tag}_${timestamp}"
elif [[ "${COMPARE_OUTPUT_DIR}" != /* ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/${COMPARE_OUTPUT_DIR}"
fi
mkdir -p "${COMPARE_OUTPUT_DIR}"

manifest_csv="${COMPARE_OUTPUT_DIR}/scenario_seed_manifest.csv"
echo "scenario,seed,run_tag,run_dir,kpi_summary_json,eval_summary_json,model_label,model_path,decode_mode,sample_seed" > "${manifest_csv}"

if [[ -n "${IGNORED_TOPOLOGY_SEED}" ]]; then
    echo "[hgraph multi-seed] ignoring --topology-seed=${IGNORED_TOPOLOGY_SEED}; using --seed-list=${SEED_LIST}" >&2
fi

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

run_one_seed() {
    local seed="$1"
    local current_build_method="$2"
    local run_tag="${safe_base_tag}_s${seed}"
    local socket_path="/tmp/cumac_stageb_hgraph_eval.sock.$$.$seed"
    local seed_eval_dir="${COMPARE_OUTPUT_DIR}/per_seed_eval/s${seed}"
    local main_log="${seed_eval_dir}/main_experiment.log"
    local eval_log="${seed_eval_dir}/checkpoint_eval.log"
    local row_file="${seed_eval_dir}/manifest_rows.csv"
    mkdir -p "${seed_eval_dir}"
    : > "${row_file}"

    local main_cmd=(
        "${PER_SEED_SCRIPT}"
        --build-method "${current_build_method}"
        --topology-seed "${seed}"
        --online-bridge 1
        --online-socket "${socket_path}"
        --custom-ue-prg 1
        --custom-policy gnnrl
        --gnnrl-action-mode joint
        --tag "${run_tag}"
    )
    if [[ "${PARALLEL_SEEDS}" -gt 1 && "${current_build_method}" == "skip" ]]; then
        main_cmd+=(--skip-compile-param-update 1)
    fi
    if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
        main_cmd+=("${FORWARD_ARGS[@]}")
    fi

    echo "[hgraph multi-seed] seed=${seed} build_method=${current_build_method} tag=${run_tag}"
    env "${SIM_ENVS[@]}" "${main_cmd[@]}" > "${main_log}" 2>&1 &
    local main_pid=$!

    local eval_cmd=(
        "${PYTHON_BIN}" -m "${EVAL_MODULE}"
        --checkpoint "${CHECKPOINT_PATH}"
        --socket-path "${socket_path}"
        --episode-horizon "${EVAL_EPISODE_HORIZON}"
        --max-steps "${EVAL_EPISODE_HORIZON}"
        --decode-mode "${DECODE_MODE}"
        --sample-seed "${SAMPLE_SEED}"
        --seed "${seed}"
        --device "${EVAL_DEVICE}"
        --print-every-steps "${EVAL_PRINT_EVERY_STEPS}"
        --save-steps "${EVAL_SAVE_STEPS}"
        --stats-warmup-steps "${EVAL_STATS_WARMUP_STEPS}"
        --out-dir "${seed_eval_dir}"
    )
    if [[ "${EVAL_UE_PRG_TOPK}" != "auto" ]]; then
        eval_cmd+=(--ue-prg-topk "${EVAL_UE_PRG_TOPK}")
    fi
    if [[ "${EVAL_UE_PRG_DIVERSITY_EXTRA}" != "auto" ]]; then
        eval_cmd+=(--ue-prg-diversity-extra "${EVAL_UE_PRG_DIVERSITY_EXTRA}")
    fi
    if [[ "${EVAL_TB_ERR_EMA_ALPHA}" != "auto" ]]; then
        eval_cmd+=(--tb-err-ema-alpha "${EVAL_TB_ERR_EMA_ALPHA}")
    fi
    if [[ "${EVAL_SLOT_REPAIR_MAX_PER_CELL}" != "auto" ]]; then
        eval_cmd+=(--slot-repair-max-per-cell "${EVAL_SLOT_REPAIR_MAX_PER_CELL}")
    fi
    if [[ "${EVAL_SLOT_REPAIR_MIN_BACKLOG_BYTES}" != "auto" ]]; then
        eval_cmd+=(--slot-repair-min-backlog-bytes "${EVAL_SLOT_REPAIR_MIN_BACKLOG_BYTES}")
    fi

    env "${EVAL_ENVS[@]}" "${eval_cmd[@]}" > "${eval_log}" 2>&1 || {
        kill "${main_pid}" 2>/dev/null || true
        wait "${main_pid}" 2>/dev/null || true
        echo "[hgraph multi-seed] eval failed for seed=${seed}; see ${eval_log}" >&2
        return 1
    }

    if ! wait "${main_pid}"; then
        echo "[hgraph multi-seed] main experiment failed for seed=${seed}; see ${main_log}" >&2
        return 1
    fi

    local out_base
    out_base="$(grep -F '[Stage-B] Done. Output base:' "${main_log}" | tail -n 1 | sed 's/.*Output base: //')"
    if [[ -z "${out_base}" ]]; then
        out_base="$(find "${ROOT_DIR}/output" -maxdepth 1 -type d -name "stageB_main_experiment_${run_tag}_*" | sort | tail -n 1)"
    fi
    if [[ -z "${out_base}" || ! -d "${out_base}" ]]; then
        echo "[hgraph multi-seed] failed to resolve output base for seed=${seed}" >&2
        return 1
    fi

    while IFS= read -r kpi_json; do
        local scenario_dir
        local scenario
        local eval_summary_json
        scenario_dir="$(dirname "${kpi_json}")"
        scenario="$(basename "${scenario_dir}")"
        eval_summary_json="${seed_eval_dir}/eval_summary.json"
        echo "${scenario},${seed},${run_tag},${out_base},${kpi_json},${eval_summary_json},${MODEL_LABEL},${CHECKPOINT_PATH},${DECODE_MODE},${SAMPLE_SEED}" >> "${row_file}"
    done < <(find "${out_base}" -mindepth 2 -maxdepth 2 -name "kpi_summary.json" | sort)
}

wait_for_seed_batch() {
    local rc=0
    local pid
    local other_pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            rc=1
            for other_pid in "${pids[@]}"; do
                kill "${other_pid}" 2>/dev/null || true
            done
        fi
    done
    pids=()
    if [[ "${rc}" -ne 0 ]]; then
        exit "${rc}"
    fi
}

if [[ "${PARALLEL_SEEDS}" -gt 1 && "${BUILD_METHOD}" != "skip" ]]; then
    build_log="${COMPARE_OUTPUT_DIR}/build_only.log"
    build_socket="/tmp/cumac_stageb_hgraph_eval_build.sock.$$"
    build_tag="${safe_base_tag}_build"
    build_cmd=(
        "${PER_SEED_SCRIPT}"
        --build-method "${BUILD_METHOD}"
        --build-only 1
        --topology-seed "${SEEDS[0]}"
        --online-bridge 1
        --online-socket "${build_socket}"
        --custom-ue-prg 1
        --custom-policy gnnrl
        --gnnrl-action-mode joint
        --tag "${build_tag}"
    )
    if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
        build_cmd+=("${FORWARD_ARGS[@]}")
    fi
    echo "[hgraph multi-seed] build-only build_method=${BUILD_METHOD} log=${build_log}"
    env "${SIM_ENVS[@]}" "${build_cmd[@]}" > "${build_log}" 2>&1
fi

run_index=0
pids=()
for seed in "${SEEDS[@]}"; do
    run_index=$((run_index + 1))
    current_build_method="${BUILD_METHOD}"
    if [[ "${PARALLEL_SEEDS}" -gt 1 ]] || (( run_index > 1 )); then
        current_build_method="skip"
    fi
    if [[ "${PARALLEL_SEEDS}" -le 1 ]]; then
        run_one_seed "${seed}" "${current_build_method}"
    else
        run_one_seed "${seed}" "${current_build_method}" &
        pids+=("$!")
        if [[ "${#pids[@]}" -ge "${PARALLEL_SEEDS}" ]]; then
            wait_for_seed_batch
        fi
    fi
done
if [[ "${PARALLEL_SEEDS}" -gt 1 && "${#pids[@]}" -gt 0 ]]; then
    wait_for_seed_batch
fi
for seed in "${SEEDS[@]}"; do
    row_file="${COMPARE_OUTPUT_DIR}/per_seed_eval/s${seed}/manifest_rows.csv"
    if [[ ! -s "${row_file}" ]]; then
        echo "[hgraph multi-seed] missing manifest rows for seed=${seed}: ${row_file}" >&2
        exit 1
    fi
    cat "${row_file}" >> "${manifest_csv}"
done

AGGREGATE_CMD=(
    "${PYTHON_BIN}" "${AGGREGATE_SCRIPT}"
    --manifest "${manifest_csv}"
    --output-dir "${COMPARE_OUTPUT_DIR}"
    --model-label "${MODEL_LABEL}"
)
if [[ -n "${BASELINE_MEAN_ROOT}" ]]; then
    AGGREGATE_CMD+=(--baseline-mean-root "${BASELINE_MEAN_ROOT}")
fi
if [[ -n "${BASELINE_MEAN_CSV}" ]]; then
    AGGREGATE_CMD+=(--baseline-mean-csv "${BASELINE_MEAN_CSV}")
fi

"${AGGREGATE_CMD[@]}"

echo "[hgraph multi-seed] manifest=${manifest_csv}"
echo "[hgraph multi-seed] output_dir=${COMPARE_OUTPUT_DIR}"
