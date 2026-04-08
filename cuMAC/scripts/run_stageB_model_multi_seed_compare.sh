#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PER_SEED_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
AGGREGATE_SCRIPT="${ROOT_DIR}/cuMAC/scripts/aggregate_model_kpi_compare.py"

BUILD_METHOD="phase4"
BASELINE_SCHEDULER="pfq"
USER_TAG=""
COMPARE_OUTPUT_DIR=""
SEED_LIST="41,42,43"
MODEL_PATH=""
MODEL_LABEL=""
BASELINE_MEAN_ROOT=""
BASELINE_MEAN_CSV=""
GNNRL_MODEL_DECODE_MODE="sample"
GNNRL_MODEL_SAMPLE_SEED=""
USER_EXEC_MODE=""
IGNORED_TOPOLOGY_SEED=""
FORWARD_ARGS=()

usage() {
    cat <<EOF
Run a fixed Stage-B ONNX model across multiple topology seeds, then aggregate model KPI means.

Usage:
  $(basename "$0") [options]

Behavior:
  1) For each topology seed in --seed-list, run ${PER_SEED_SCRIPT##*/}
     with --custom-ue-prg 1 --custom-policy gnnrl_model
  2) Collect per-seed scenario kpi_summary.json outputs
  3) Aggregate mean/std/min/max model metrics per scenario
  4) If RR/PFQ mean CSV is provided, emit a unified RR/PFQ/model compare table

Wrapper-specific options:
  --seed-list <csv>                 Topology seeds to run, e.g. 41,42,43 (default: ${SEED_LIST})
  --model-path <path>               Required ONNX model path
  --model-label <name>              Optional label for output columns/files (default: ONNX stem)
  --baseline-mean-root <dir>        Optional RR/PFQ aggregate root or scenario dir
  --baseline-mean-csv <path>        Optional RR/PFQ mean CSV for direct compare
  --baseline-scheduler <m>          Forwarded baseline label for logs; ignored by custom UE+PRG mode
  --gnnrl-model-decode-mode <m>     argmax | sample (default: ${GNNRL_MODEL_DECODE_MODE})
  --gnnrl-model-sample-seed <n>     Sample RNG seed; default is per-seed topology seed in sample mode
  --exec-mode <m>                   both | gpu; gpu is auto-upgraded to both for custom UE+PRG
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

normalize_model_label() {
    local raw="$1"
    local safe
    safe="$(echo "${raw}" | tr -c '[:alnum:]' '_' | sed 's/^_*//; s/_*$//')"
    if [[ -z "${safe}" ]]; then
        safe="model"
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
        --seed-list)
            SEED_LIST="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-label)
            MODEL_LABEL="$2"
            shift 2
            ;;
        --baseline-mean-root)
            BASELINE_MEAN_ROOT="$2"
            shift 2
            ;;
        --baseline-mean-csv)
            BASELINE_MEAN_CSV="$2"
            shift 2
            ;;
        --baseline-scheduler)
            BASELINE_SCHEDULER="$2"
            shift 2
            ;;
        --gnnrl-model-decode-mode)
            GNNRL_MODEL_DECODE_MODE="$2"
            shift 2
            ;;
        --gnnrl-model-sample-seed)
            GNNRL_MODEL_SAMPLE_SEED="$2"
            shift 2
            ;;
        --exec-mode)
            USER_EXEC_MODE="$2"
            shift 2
            ;;
        --compare-output-dir)
            COMPARE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-method)
            BUILD_METHOD="$2"
            shift 2
            ;;
        --tag)
            USER_TAG="$2"
            shift 2
            ;;
        --custom-ue-prg|--custom-policy)
            shift 2
            ;;
        --topology-seed)
            IGNORED_TOPOLOGY_SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
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
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate script: ${AGGREGATE_SCRIPT}" >&2
    exit 1
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "--model-path is required" >&2
    exit 1
fi

MODEL_PATH="$(resolve_path "${MODEL_PATH}")"
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ -n "${BASELINE_MEAN_ROOT}" ]]; then
    BASELINE_MEAN_ROOT="$(resolve_path "${BASELINE_MEAN_ROOT}")"
    if [[ ! -e "${BASELINE_MEAN_ROOT}" ]]; then
        echo "Baseline mean root not found: ${BASELINE_MEAN_ROOT}" >&2
        exit 1
    fi
fi
if [[ -n "${BASELINE_MEAN_CSV}" ]]; then
    BASELINE_MEAN_CSV="$(resolve_path "${BASELINE_MEAN_CSV}")"
    if [[ ! -f "${BASELINE_MEAN_CSV}" ]]; then
        echo "Baseline mean CSV not found: ${BASELINE_MEAN_CSV}" >&2
        exit 1
    fi
fi

if [[ -z "${MODEL_LABEL}" ]]; then
    MODEL_LABEL="$(basename "${MODEL_PATH}")"
    MODEL_LABEL="${MODEL_LABEL%.*}"
fi
MODEL_LABEL="$(normalize_model_label "${MODEL_LABEL}")"

GNNRL_MODEL_DECODE_MODE="$(echo "${GNNRL_MODEL_DECODE_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${GNNRL_MODEL_DECODE_MODE}" != "argmax" && "${GNNRL_MODEL_DECODE_MODE}" != "sample" ]]; then
    echo "--gnnrl-model-decode-mode must be argmax or sample" >&2
    exit 1
fi
if [[ "${GNNRL_MODEL_DECODE_MODE}" == "sample" && -n "${GNNRL_MODEL_SAMPLE_SEED}" ]]; then
    if ! [[ "${GNNRL_MODEL_SAMPLE_SEED}" =~ ^[0-9]+$ ]]; then
        echo "--gnnrl-model-sample-seed must be a non-negative integer" >&2
        exit 1
    fi
fi

RUN_EXEC_MODE="both"
if [[ -n "${USER_EXEC_MODE}" ]]; then
    USER_EXEC_MODE="$(echo "${USER_EXEC_MODE}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${USER_EXEC_MODE}" != "both" && "${USER_EXEC_MODE}" != "gpu" ]]; then
        echo "--exec-mode must be both or gpu" >&2
        exit 1
    fi
    if [[ "${USER_EXEC_MODE}" == "gpu" ]]; then
        echo "[model multi-seed] --exec-mode=gpu is incompatible with custom UE+PRG mode; auto-switch to both." >&2
        RUN_EXEC_MODE="both"
    else
        RUN_EXEC_MODE="${USER_EXEC_MODE}"
    fi
fi

mapfile -t SEEDS < <(parse_seed_list "${SEED_LIST}")

timestamp="$(date +%Y%m%d_%H%M%S)"
default_base_tag="${MODEL_LABEL}_multiseed_compare"
base_tag="${USER_TAG:-${default_base_tag}}"
safe_base_tag="$(normalize_model_label "${base_tag}")"

if [[ -z "${COMPARE_OUTPUT_DIR}" ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/output/stageB_model_multiseed_compare_${safe_base_tag}_${timestamp}"
elif [[ "${COMPARE_OUTPUT_DIR}" != /* ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/${COMPARE_OUTPUT_DIR}"
fi
mkdir -p "${COMPARE_OUTPUT_DIR}"

seed_runs_manifest="${COMPARE_OUTPUT_DIR}/seed_runs.csv"
scenario_seed_manifest="${COMPARE_OUTPUT_DIR}/scenario_seed_manifest.csv"
summary_file="${COMPARE_OUTPUT_DIR}/aggregate_summary.txt"

echo "seed,run_tag,run_dir,model_label,model_path,decode_mode,sample_seed" > "${seed_runs_manifest}"
echo "scenario,seed,run_tag,run_dir,kpi_summary_json,model_label,model_path,decode_mode,sample_seed" > "${scenario_seed_manifest}"

if [[ -n "${IGNORED_TOPOLOGY_SEED}" ]]; then
    echo "[model multi-seed] ignoring --topology-seed=${IGNORED_TOPOLOGY_SEED}; using --seed-list=${SEED_LIST}" >&2
fi

output_root="${ROOT_DIR}/output"
mkdir -p "${output_root}"

seed_index=0
for seed in "${SEEDS[@]}"; do
    seed_build_method="skip"
    if [[ ${seed_index} -eq 0 ]]; then
        seed_build_method="${BUILD_METHOD}"
    fi
    seed_sample_seed="${GNNRL_MODEL_SAMPLE_SEED}"
    if [[ "${GNNRL_MODEL_DECODE_MODE}" == "sample" && -z "${seed_sample_seed}" ]]; then
        seed_sample_seed="${seed}"
    fi

    seed_tag="${safe_base_tag}_s${seed}"
    before_file="${COMPARE_OUTPUT_DIR}/before_s${seed}.txt"
    after_file="${COMPARE_OUTPUT_DIR}/after_s${seed}.txt"

    find "${output_root}" -maxdepth 1 -type d -name "stageB_main_experiment_${seed_tag}_*" | sort > "${before_file}"

    echo "[model multi-seed] start seed=${seed} build_method=${seed_build_method} tag=${seed_tag}" >&2
    cmd=(
        "${PER_SEED_SCRIPT}"
        "${FORWARD_ARGS[@]}"
        --topology-seed "${seed}"
        --build-method "${seed_build_method}"
        --baseline-scheduler "${BASELINE_SCHEDULER}"
        --custom-ue-prg "1"
        --custom-policy "gnnrl_model"
        --model-path "${MODEL_PATH}"
        --gnnrl-model-decode-mode "${GNNRL_MODEL_DECODE_MODE}"
        --exec-mode "${RUN_EXEC_MODE}"
        --tag "${seed_tag}"
    )
    if [[ -n "${seed_sample_seed}" ]]; then
        cmd+=(--gnnrl-model-sample-seed "${seed_sample_seed}")
    fi
    "${cmd[@]}"

    find "${output_root}" -maxdepth 1 -type d -name "stageB_main_experiment_${seed_tag}_*" | sort > "${after_file}"
    run_dir="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
    if [[ -z "${run_dir}" ]]; then
        run_dir="$(tail -n 1 "${after_file}" || true)"
    fi
    if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        echo "Unable to locate output directory for seed=${seed} tag=${seed_tag}" >&2
        exit 1
    fi

    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "${seed}" \
        "${seed_tag}" \
        "${run_dir}" \
        "${MODEL_LABEL}" \
        "${MODEL_PATH}" \
        "${GNNRL_MODEL_DECODE_MODE}" \
        "${seed_sample_seed}" >> "${seed_runs_manifest}"

    scenario_count=0
    while IFS= read -r summary_json; do
        [[ -z "${summary_json}" ]] && continue
        scenario_dir="$(dirname "${summary_json}")"
        scenario="$(basename "${scenario_dir}")"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${scenario}" \
            "${seed}" \
            "${seed_tag}" \
            "${run_dir}" \
            "${summary_json}" \
            "${MODEL_LABEL}" \
            "${MODEL_PATH}" \
            "${GNNRL_MODEL_DECODE_MODE}" \
            "${seed_sample_seed}" >> "${scenario_seed_manifest}"
        scenario_count=$((scenario_count + 1))
    done < <(find "${run_dir}" -mindepth 2 -maxdepth 2 -type f -name "kpi_summary.json" | sort)

    if [[ ${scenario_count} -eq 0 ]]; then
        echo "No scenario kpi_summary.json found under ${run_dir}" >&2
        exit 1
    fi

    seed_index=$((seed_index + 1))
done

aggregate_cmd=(
    python3 "${AGGREGATE_SCRIPT}"
    --manifest "${scenario_seed_manifest}"
    --output-dir "${COMPARE_OUTPUT_DIR}"
    --model-label "${MODEL_LABEL}"
)
if [[ -n "${BASELINE_MEAN_ROOT}" ]]; then
    aggregate_cmd+=(--baseline-mean-root "${BASELINE_MEAN_ROOT}")
fi
if [[ -n "${BASELINE_MEAN_CSV}" ]]; then
    aggregate_cmd+=(--baseline-mean-csv "${BASELINE_MEAN_CSV}")
fi
"${aggregate_cmd[@]}"

{
    echo "model_label=${MODEL_LABEL}"
    echo "model_path=${MODEL_PATH}"
    echo "decode_mode=${GNNRL_MODEL_DECODE_MODE}"
    if [[ "${GNNRL_MODEL_DECODE_MODE}" == "sample" && -z "${GNNRL_MODEL_SAMPLE_SEED}" ]]; then
        echo "sample_seed=topology_seed"
    else
        echo "sample_seed=${GNNRL_MODEL_SAMPLE_SEED:-unused}"
    fi
    echo "exec_mode=${RUN_EXEC_MODE}"
    echo "seed_list=${SEED_LIST}"
    echo "seed_count=${#SEEDS[@]}"
    echo "seed_runs_manifest=${seed_runs_manifest}"
    echo "scenario_seed_manifest=${scenario_seed_manifest}"
    if [[ -n "${BASELINE_MEAN_ROOT}" ]]; then
        echo "baseline_mean_root=${BASELINE_MEAN_ROOT}"
    fi
    if [[ -n "${BASELINE_MEAN_CSV}" ]]; then
        echo "baseline_mean_csv=${BASELINE_MEAN_CSV}"
    fi
    echo "aggregate_manifest=${COMPARE_OUTPUT_DIR}/aggregate_manifest.csv"
} > "${summary_file}"

echo "[model multi-seed] output_dir=${COMPARE_OUTPUT_DIR}"
echo "[model multi-seed] seed_runs_manifest=${seed_runs_manifest}"
echo "[model multi-seed] aggregate_manifest=${COMPARE_OUTPUT_DIR}/aggregate_manifest.csv"
