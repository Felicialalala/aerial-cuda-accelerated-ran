#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-----------------------------------------------------------------------------------
#This script is to build Aerial SDK
#The build directory is distinct based on the system architecture (aarch64|x86_64)
#-----------------------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

if [ ! -f "/.dockerenv" ]; then
  echo "You need to run this script inside the container"
  echo "Please run the following command to build the SDK:"
  echo " ./cuPHY-CP/container/run_aerial.sh"
  echo "  and then run this script"
  echo "  testBenches/phase4_test_scripts/build_aerial_sdk.sh"
  exit 1
fi

ARCH=$(uname -m)
TOOLCHAIN_FILE=""
DEFAULT_TOOLCHAIN=""

valid_tc_files=("bf3" "devkit" "grace-cross" "native" "r750" "x86-64")
valid_presets=("perf" "10_02" "10_04" "10_04_32dl")

show_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --build_dir <dir>, -b <dir>   Specify the build directory to use."
  echo
  echo "  --toolchain <tc_file>, -t <tc_file>"
  echo "                                Specify the CMake toolchain file, which can be one of:"
  echo "                                ${valid_tc_files[*]}"
  echo
  echo "  --build-only                  Allows skipping configuration and running only the build phase"
  echo
  echo "  --preset, -p                  Preset build options:"
  echo "                                perf       - Performance build (default)"
  echo "                                10_02      - FAPI 10_02"
  echo "                                10_04      - FAPI 10_04"
  echo "                                10_04_32dl - FAPI 10_04 with 32 DL layers"
  echo
  echo "  --dry-run, -d                 Display the CMake command without executing it"
  echo
  echo "  -- [cmake_options]            Pass additional options to CMake"
  echo "                                Takes precedence over cmake options in preset"
  echo
  echo "  -h, --help                    Show this help message and exit"
  echo
  echo "Default Values:"
  echo "  Build directory: <cuBB_SDK>/build.\$ARCH"
  echo "  Toolchain file: grace-cross (for aarch64) or devkit (for x86_64)"
  echo "  Preset: perf"
  echo
  echo "Examples:"
  echo "  $0 --build_dir build --toolchain native --preset perf -- -DCMAKE_BUILD_TYPE=Debug"
  echo "                                Configures a build in 'build' with the 'native' toolchain using the perf preset and Debug mode."
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --build_dir|-b)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --build_dir option"
        show_usage
        exit 1
      fi
      BUILD_DIR="$2"
      shift 2
      ;;
    --toolchain|-t)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        TOOLCHAIN_FILE="$cuBB_SDK/cuPHY/cmake/toolchains/$2"
        if [[ ! " ${valid_tc_files[*]} " =~ $2 ]]; then
            echo "Error: Invalid toolchain '$2'. Please provide one of the following: ${valid_tc_files[*]}"
            exit 1
        fi
        shift 2
      else
        echo "Error: --toolchain requires a non-empty argument."
        show_usage
        exit 1
      fi
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --preset|-p)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        PRESET="$2"
        if [[ ! " ${valid_presets[*]} " =~ $2 ]]; then
            echo "Error: Invalid preset '$2'. Please provide one of the following: ${valid_presets[*]}"
            exit 1
        fi
        shift 2
      else
        echo "Error: --preset requires a non-empty argument."
        show_usage
        exit 1
      fi
      ;;
    --dry-run|-d)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Capture additional CMake options
extra_cmake_options=("$@")

# Set default values
BUILD_ONLY=${BUILD_ONLY:-0}
PRESET=${PRESET:-perf}

# Initialize BUILD_DIR if not set via argument
if [[ -z "$BUILD_DIR" ]]; then
  BUILD_DIR="build.$ARCH"
fi

# Determine the default toolchain based on architecture if not provided
if [ "$ARCH" = "aarch64" ]; then
  DEFAULT_TOOLCHAIN="$cuBB_SDK/cuPHY/cmake/toolchains/grace-cross"
elif [ "$ARCH" = "x86_64" ]; then
  DEFAULT_TOOLCHAIN="$cuBB_SDK/cuPHY/cmake/toolchains/devkit"
fi

if [[ -z "$TOOLCHAIN_FILE" ]]; then
  TOOLCHAIN_FILE="$DEFAULT_TOOLCHAIN"
fi

if [[ "$PRESET" == "perf" ]]; then
  CMAKE_FLAGS="-GNinja \
    --log-level=debug \
    -DCMAKE_TOOLCHAIN_FILE=\"$TOOLCHAIN_FILE\" \
    -DSCF_FAPI_10_04=ON \
    -DENABLE_CONFORMANCE_TM_PDSCH_PDCCH=ON \
    -DENABLE_20C=ON \
    ${extra_cmake_options[@]}"
elif [[ "$PRESET" == "10_02" ]]; then
  CMAKE_FLAGS="-GNinja \
    --log-level=debug \
    -DCMAKE_TOOLCHAIN_FILE=\"$TOOLCHAIN_FILE\" \
    -DSCF_FAPI_10_04=OFF \
    ${extra_cmake_options[@]}"
elif [[ "$PRESET" == "10_04" ]]; then
  CMAKE_FLAGS="-GNinja \
    --log-level=debug \
    -DCMAKE_TOOLCHAIN_FILE=\"$TOOLCHAIN_FILE\" \
    -DSCF_FAPI_10_04=ON \
    -DENABLE_CONFORMANCE_TM_PDSCH_PDCCH=ON \
    ${extra_cmake_options[@]}"
elif [[ "$PRESET" == "10_04_32dl" ]]; then
  CMAKE_FLAGS="-GNinja \
    --log-level=debug \
    -DCMAKE_TOOLCHAIN_FILE=\"$TOOLCHAIN_FILE\" \
    -DSCF_FAPI_10_04=ON \
    -DENABLE_CONFORMANCE_TM_PDSCH_PDCCH=ON \
    -DENABLE_32DL=ON \
    ${extra_cmake_options[@]}"
else
  echo "ERROR: Unknown preset - $PRESET"
  exit 1
fi

rt_wrapper() {
    local uid gid
    uid=$(id -u)
    gid=$(id -g)

    sudo -E chrt -r 30 taskset --cpu-list "$(cat /sys/devices/system/cpu/isolated)" \
        setpriv --reuid "$uid" --regid "$gid" --init-groups env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH "$@" || "$@"
}

BUILD_CMD="rt_wrapper cmake --build \"$BUILD_DIR\" -j $(nproc)"
if [[ "$BUILD_ONLY" != "1" ]]; then
  BUILD_CMD="cmake -B$BUILD_DIR $CMAKE_FLAGS && $BUILD_CMD"
fi

# Execute or display based on dry-run
if [[ -z "${DRY_RUN:-}" || "$DRY_RUN" != "1" ]]; then

  eval $BUILD_CMD
  # Check if build succeeded
  if [[ $? -ne 0 ]]; then
    echo "Error: Build failed."
    exit 1
  fi
  echo "Build completed successfully."
else
  echo "Dry run:"
  echo "$BUILD_CMD"
fi
