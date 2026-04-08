#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PER_SEED_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_rr_pf_compare.sh"
AGGREGATE_SCRIPT="${ROOT_DIR}/cuMAC/scripts/aggregate_rr_pf_compare.py"

BUILD_METHOD="phase4"
USER_TAG=""
CUSTOM_UE_PRG="0"
COMPARE_TOP_N=10
COMPARE_OUTPUT_DIR=""
PF_BASELINE="pf"
SEED_LIST="41,42,43"
IGNORED_TOPOLOGY_SEED=""
FORWARD_ARGS=()

usage() {
    cat <<EOF
Run Stage-B native RR and PF/PFQ across multiple topology seeds, then aggregate mean metrics.

Usage:
  $(basename "$0") [options]

Behavior:
  1) For each topology seed in --seed-list, run ${PER_SEED_SCRIPT##*/}
  2) Collect per-seed RR vs PF/PFQ compare JSON outputs
  3) Aggregate mean/std/min/max per scenario into rr_vs_<pf-baseline>_compare_mean.*

Wrapper-specific options:
  --seed-list <csv>          Topology seeds to run, e.g. 41,42,43 (default: ${SEED_LIST})
  --pf-baseline <m>          Second baseline to compare against RR: pf | pfq (default: ${PF_BASELINE})
  --compare-top-n <n>        Forwarded to per-seed compare runs (default: ${COMPARE_TOP_N})
  --compare-output-dir <dir> Base output directory for all per-seed and aggregated artifacts
  --build-method <m>         Build method for the first seed; later seeds use skip
  --tag <name>               Optional base tag; per-seed runs expand it to <tag>_s<seed>
  --topology-seed <n>        Ignored by this wrapper; use --seed-list instead
  -h, --help                 Show this help

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-scheduler)
            shift 2
            ;;
        --pf-baseline)
            PF_BASELINE="$2"
            shift 2
            ;;
        --compare-top-n)
            COMPARE_TOP_N="$2"
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
        --custom-ue-prg)
            CUSTOM_UE_PRG="$2"
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --seed-list)
            SEED_LIST="$2"
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
    echo "Missing per-seed compare script: ${PER_SEED_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate script: ${AGGREGATE_SCRIPT}" >&2
    exit 1
fi
if [[ "${CUSTOM_UE_PRG}" != "0" ]]; then
    echo "This wrapper only supports native baseline runs with --custom-ue-prg 0." >&2
    exit 1
fi
if ! [[ "${COMPARE_TOP_N}" =~ ^[0-9]+$ ]]; then
    echo "--compare-top-n must be a non-negative integer" >&2
    exit 1
fi
PF_BASELINE="$(echo "${PF_BASELINE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PF_BASELINE}" != "pf" && "${PF_BASELINE}" != "pfq" ]]; then
    echo "--pf-baseline must be pf or pfq" >&2
    exit 1
fi

mapfile -t SEEDS < <(parse_seed_list "${SEED_LIST}")

timestamp="$(date +%Y%m%d_%H%M%S)"
default_base_tag="rr_${PF_BASELINE}_multi_seed_compare"
base_tag="${USER_TAG:-${default_base_tag}}"

if [[ -z "${COMPARE_OUTPUT_DIR}" ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/output/stageB_rr_${PF_BASELINE}_multiseed_compare_${base_tag}_${timestamp}"
elif [[ "${COMPARE_OUTPUT_DIR}" != /* ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/${COMPARE_OUTPUT_DIR}"
fi
mkdir -p "${COMPARE_OUTPUT_DIR}/per_seed"

seed_runs_manifest="${COMPARE_OUTPUT_DIR}/seed_runs.csv"
scenario_seed_manifest="${COMPARE_OUTPUT_DIR}/scenario_seed_manifest.csv"
summary_file="${COMPARE_OUTPUT_DIR}/aggregate_summary.txt"

echo "seed,compare_output_dir,compare_manifest,compare_summary,rr_wrapper_log,other_wrapper_log" > "${seed_runs_manifest}"
echo "scenario,seed,rr_dir,other_dir,other_baseline,compare_dir,compare_csv,compare_json,compare_txt" > "${scenario_seed_manifest}"

if [[ -n "${IGNORED_TOPOLOGY_SEED}" ]]; then
    echo "[RR/${PF_BASELINE} multi-seed] ignoring --topology-seed=${IGNORED_TOPOLOGY_SEED}; using --seed-list=${SEED_LIST}" >&2
fi

seed_index=0
for seed in "${SEEDS[@]}"; do
    seed_build_method="skip"
    if [[ ${seed_index} -eq 0 ]]; then
        seed_build_method="${BUILD_METHOD}"
    fi

    seed_out_dir="${COMPARE_OUTPUT_DIR}/per_seed/s${seed}"
    seed_tag="${base_tag}_s${seed}"
    mkdir -p "${seed_out_dir}"

    echo "[RR/${PF_BASELINE} multi-seed] start seed=${seed} build_method=${seed_build_method} tag=${seed_tag}" >&2
    "${PER_SEED_SCRIPT}" \
        "${FORWARD_ARGS[@]}" \
        --topology-seed "${seed}" \
        --build-method "${seed_build_method}" \
        --pf-baseline "${PF_BASELINE}" \
        --compare-top-n "${COMPARE_TOP_N}" \
        --compare-output-dir "${seed_out_dir}" \
        --tag "${seed_tag}"

    seed_manifest="${seed_out_dir}/compare_manifest.csv"
    seed_summary="${seed_out_dir}/compare_summary.txt"
    rr_log="${seed_out_dir}/rr_wrapper_run.log"
    other_log="${seed_out_dir}/${PF_BASELINE}_wrapper_run.log"

    if [[ ! -f "${seed_manifest}" ]]; then
        echo "Missing per-seed compare manifest: ${seed_manifest}" >&2
        exit 1
    fi

    printf '%s,%s,%s,%s,%s,%s\n' \
        "${seed}" \
        "${seed_out_dir}" \
        "${seed_manifest}" \
        "${seed_summary}" \
        "${rr_log}" \
        "${other_log}" >> "${seed_runs_manifest}"

    while IFS=, read -r scenario rr_dir other_dir other_baseline compare_dir; do
        [[ -z "${scenario}" || "${scenario}" == "scenario" ]] && continue
        compare_csv="${compare_dir}/rr_vs_${PF_BASELINE}_compare.csv"
        compare_json="${compare_dir}/rr_vs_${PF_BASELINE}_compare.json"
        compare_txt="${compare_dir}/rr_vs_${PF_BASELINE}_compare.txt"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${scenario}" \
            "${seed}" \
            "${rr_dir}" \
            "${other_dir}" \
            "${other_baseline}" \
            "${compare_dir}" \
            "${compare_csv}" \
            "${compare_json}" \
            "${compare_txt}" >> "${scenario_seed_manifest}"
    done < "${seed_manifest}"

    seed_index=$((seed_index + 1))
done

python3 "${AGGREGATE_SCRIPT}" \
    --manifest "${scenario_seed_manifest}" \
    --output-dir "${COMPARE_OUTPUT_DIR}" \
    --compare-baseline "${PF_BASELINE}"

{
    echo "compare_baseline=${PF_BASELINE}"
    echo "seed_list=${SEED_LIST}"
    echo "seed_count=${#SEEDS[@]}"
    echo "seed_runs_manifest=${seed_runs_manifest}"
    echo "scenario_seed_manifest=${scenario_seed_manifest}"
    echo "aggregate_manifest=${COMPARE_OUTPUT_DIR}/aggregate_manifest.csv"
} > "${summary_file}"

echo "[RR/${PF_BASELINE} multi-seed] output_dir=${COMPARE_OUTPUT_DIR}"
echo "[RR/${PF_BASELINE} multi-seed] seed_runs_manifest=${seed_runs_manifest}"
echo "[RR/${PF_BASELINE} multi-seed] aggregate_manifest=${COMPARE_OUTPUT_DIR}/aggregate_manifest.csv"
