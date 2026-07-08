#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MULTI_SEED_SCRIPT="${ROOT_DIR}/cuMAC/scripts/run_stageB_rr_pf_multi_seed_compare.sh"
AGGREGATE_SCRIPT="${ROOT_DIR}/cuMAC/scripts/aggregate_stageB_baseline_eval_suite.py"

BUILD_METHOD="phase4"
SEED_LIST="41,42,43,44,45,46,47,48,49,50"
SUITE_OUTPUT_DIR=""
TAG_PREFIX="baseline_eval"
FORWARD_ARGS=()

usage() {
    cat <<EOF
Run the comprehensive Stage-B native baseline evaluation suite.

The suite runs:
  1) uniform RRQ vs PFQ
  2) coop_boundary RRQ vs PFQ
  3) uniform RR vs PF

All runs default to seed41-50 and enable packet-delay sample CSV export.

Usage:
  $(basename "$0") [options] [run_stageB_rr_pf_multi_seed_compare.sh options]

Suite options:
  --build-method <m>       Build method for the first suite leg: phase4 | cmake | skip (default: ${BUILD_METHOD})
  --seed-list <csv>        Seeds to average (default: ${SEED_LIST})
  --suite-output-dir <dir> Output root for all three suite legs
  --tag-prefix <name>      Tag prefix for generated run names (default: ${TAG_PREFIX})
  -h, --help               Show this help

Common forwarded defaults:
  --topology-scenario 3cell
  --total-ue-count 36
  --tti 4000
  --cell-radius 500
  --cell-assoc-mode strongest
  --packet-size-bytes 3000
  --traffic-arrival-rate 1.0
  --packet-ttl-ms 200
  --prbs-per-group 16
  --precoding svd
  --fading-mode 0 --cdl-profiles NA --cdl-delay-spreads 0
  --packet-delay-samples 1

Any extra option after the suite options is forwarded and can override these defaults when the
underlying wrapper sees it later in argv.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-method)
            BUILD_METHOD="$2"
            shift 2
            ;;
        --seed-list)
            SEED_LIST="$2"
            shift 2
            ;;
        --suite-output-dir)
            SUITE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --tag-prefix)
            TAG_PREFIX="$2"
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

if [[ ! -x "${MULTI_SEED_SCRIPT}" ]]; then
    echo "Missing multi-seed compare script: ${MULTI_SEED_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing suite aggregate script: ${AGGREGATE_SCRIPT}" >&2
    exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${SUITE_OUTPUT_DIR}" ]]; then
    SUITE_OUTPUT_DIR="${ROOT_DIR}/output/stageB_baseline_eval_suite_${TAG_PREFIX}_${timestamp}"
elif [[ "${SUITE_OUTPUT_DIR}" != /* ]]; then
    SUITE_OUTPUT_DIR="${ROOT_DIR}/${SUITE_OUTPUT_DIR}"
fi
mkdir -p "${SUITE_OUTPUT_DIR}"

COMMON_ARGS=(
    --seed-list "${SEED_LIST}"
    --topology-scenario 3cell
    --total-ue-count 36
    --tti 4000
    --cell-radius 500
    --cell-assoc-mode strongest
    --ue-voronoi-clip 1
    --traffic-percent 100
    --packet-size-bytes 3000
    --traffic-arrival-rate 1.0
    --packet-ttl-ms 200
    --prbs-per-group 16
    --precoding svd
    --fading-mode 0
    --cdl-profiles NA
    --cdl-delay-spreads 0
    --packet-delay-samples 1
)

run_leg() {
    local leg="$1"
    local build_method="$2"
    local reference="$3"
    local compare="$4"
    local placement="$5"
    local tag="${TAG_PREFIX}_${leg}"
    local out_dir="${SUITE_OUTPUT_DIR}/${leg}"

    echo "[Stage-B baseline suite] start leg=${leg} reference=${reference} compare=${compare} placement=${placement}"
    "${MULTI_SEED_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        "${FORWARD_ARGS[@]}" \
        --build-method "${build_method}" \
        --reference-baseline "${reference}" \
        --pf-baseline "${compare}" \
        --ue-placement "${placement}" \
        --tag "${tag}" \
        --compare-output-dir "${out_dir}"
}

run_leg "uniform_rrq_pfq" "${BUILD_METHOD}" "rrq" "pfq" "uniform"
run_leg "boundary_rrq_pfq" "skip" "rrq" "pfq" "coop_boundary"
run_leg "uniform_rr_pf" "skip" "rr" "pf" "uniform"

python3 "${AGGREGATE_SCRIPT}" \
    --suite-dir "${SUITE_OUTPUT_DIR}" \
    --scenario RAYLEIGH

{
    echo "suite_output_dir=${SUITE_OUTPUT_DIR}"
    echo "seed_list=${SEED_LIST}"
    echo "uniform_rrq_pfq=${SUITE_OUTPUT_DIR}/uniform_rrq_pfq"
    echo "boundary_rrq_pfq=${SUITE_OUTPUT_DIR}/boundary_rrq_pfq"
    echo "uniform_rr_pf=${SUITE_OUTPUT_DIR}/uniform_rr_pf"
    echo "suite_aggregate=${SUITE_OUTPUT_DIR}/suite_aggregate"
} > "${SUITE_OUTPUT_DIR}/suite_summary.txt"

echo "[Stage-B baseline suite] output=${SUITE_OUTPUT_DIR}"
