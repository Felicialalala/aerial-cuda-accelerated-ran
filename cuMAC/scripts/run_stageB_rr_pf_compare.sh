#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_main_experiment.sh"
COMPARE_SCRIPT="${ROOT_DIR}/cuMAC/scripts/rr_vs_pf_compare.py"

BUILD_METHOD="phase4"
USER_TAG=""
CUSTOM_UE_PRG="0"
COMPARE_TOP_N=10
COMPARE_OUTPUT_DIR=""
FORWARD_ARGS=()

usage() {
    cat <<EOF
Run Stage-B native RR and PF back-to-back, then compare the results.

Usage:
  $(basename "$0") [options]

Behavior:
  1) Runs ${RUN_SCRIPT##*/} once with --baseline-scheduler rr
  2) Runs ${RUN_SCRIPT##*/} once with --baseline-scheduler pf
  3) Runs ${COMPARE_SCRIPT##*/} on every common passing scenario directory

Wrapper-specific options:
  --compare-top-n <n>         Top-N UE deltas in compare text output (default: ${COMPARE_TOP_N})
  --compare-output-dir <dir>  Base output directory for compare artifacts
  --tag <name>                Optional base tag; wrapper expands it to <tag>_rr and <tag>_pf
  --baseline-scheduler <m>    Ignored by this wrapper; both rr and pf are always executed
  -h, --help                  Show this help

All other options are forwarded to ${RUN_SCRIPT##*/}.
Typical example:
  $(basename "$0") \\
    --build-method cmake \\
    --fading-mode 0 \\
    --cdl-profiles NA \\
    --cdl-delay-spreads 0 \\
    --tti 2000 \\
    --custom-ue-prg 0 \\
    --packet-size-bytes 3000 \\
    --traffic-arrival-rate 0.2 \\
    --topology-seed 42 \\
    --exec-mode gpu
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-scheduler)
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

if [[ ! -x "${RUN_SCRIPT}" ]]; then
    echo "Missing run script: ${RUN_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
    echo "Missing compare script: ${COMPARE_SCRIPT}" >&2
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

timestamp="$(date +%Y%m%d_%H%M%S)"
base_tag="${USER_TAG:-rr_pf_compare}"
rr_tag="${base_tag}_rr"
pf_tag="${base_tag}_pf"

if [[ -z "${COMPARE_OUTPUT_DIR}" ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/output/stageB_rr_pf_compare_${base_tag}_${timestamp}"
elif [[ "${COMPARE_OUTPUT_DIR}" != /* ]]; then
    COMPARE_OUTPUT_DIR="${ROOT_DIR}/${COMPARE_OUTPUT_DIR}"
fi
mkdir -p "${COMPARE_OUTPUT_DIR}"

rr_wrapper_log="${COMPARE_OUTPUT_DIR}/rr_wrapper_run.log"
pf_wrapper_log="${COMPARE_OUTPUT_DIR}/pf_wrapper_run.log"
manifest_file="${COMPARE_OUTPUT_DIR}/compare_manifest.csv"
summary_file="${COMPARE_OUTPUT_DIR}/compare_summary.txt"

run_one() {
    local baseline="$1"
    local build_method="$2"
    local tag="$3"
    local wrapper_log="$4"
    local out_base

    echo "[RR/PF] start baseline=${baseline} build_method=${build_method} tag=${tag}" | tee "${wrapper_log}" >&2
    set +e
    "${RUN_SCRIPT}" "${FORWARD_ARGS[@]}" --build-method "${build_method}" --baseline-scheduler "${baseline}" --tag "${tag}" \
        2>&1 | tee -a "${wrapper_log}" >&2
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "[RR/PF] run failed baseline=${baseline} rc=${rc}" | tee -a "${wrapper_log}" >&2
        return ${rc}
    fi
    out_base="$(sed -n 's/^\[Stage-B\] Done\. Output base: //p' "${wrapper_log}" | tail -n 1)"
    if [[ -z "${out_base}" ]]; then
        echo "Failed to locate output base for baseline=${baseline}" >&2
        return 1
    fi
    printf '%s\n' "${out_base}"
}

second_build_method="skip"
if [[ "${BUILD_METHOD}" == "skip" ]]; then
    second_build_method="skip"
fi

rr_out_base="$(run_one rr "${BUILD_METHOD}" "${rr_tag}" "${rr_wrapper_log}")"
pf_out_base="$(run_one pf "${second_build_method}" "${pf_tag}" "${pf_wrapper_log}")"

echo "scenario,rr_dir,pf_dir,compare_dir" > "${manifest_file}"

compare_count=0
while IFS= read -r scenario_dir; do
    scenario="$(basename "${scenario_dir}")"
    rr_dir="${rr_out_base}/${scenario}"
    pf_dir="${pf_out_base}/${scenario}"
    if [[ ! -f "${rr_dir}/kpi_summary.json" || ! -f "${pf_dir}/kpi_summary.json" ]]; then
        continue
    fi
    out_dir="${COMPARE_OUTPUT_DIR}/${scenario}"
    mkdir -p "${out_dir}"
    python3 "${COMPARE_SCRIPT}" --rr "${rr_dir}" --pf "${pf_dir}" --output-dir "${out_dir}" --top-n "${COMPARE_TOP_N}"
    echo "${scenario},${rr_dir},${pf_dir},${out_dir}" >> "${manifest_file}"
    compare_count=$((compare_count + 1))
done < <(find "${rr_out_base}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${compare_count} -eq 0 ]]; then
    echo "No common scenario directories with kpi_summary.json were found between:" >&2
    echo "  RR: ${rr_out_base}" >&2
    echo "  PF: ${pf_out_base}" >&2
    exit 1
fi

{
    echo "rr_out_base=${rr_out_base}"
    echo "pf_out_base=${pf_out_base}"
    echo "compare_output_dir=${COMPARE_OUTPUT_DIR}"
    echo "compare_top_n=${COMPARE_TOP_N}"
    echo "compare_count=${compare_count}"
    echo "manifest=${manifest_file}"
} > "${summary_file}"

echo "[RR/PF] rr_out_base=${rr_out_base}"
echo "[RR/PF] pf_out_base=${pf_out_base}"
echo "[RR/PF] compare_output_dir=${COMPARE_OUTPUT_DIR}"
echo "[RR/PF] manifest=${manifest_file}"
