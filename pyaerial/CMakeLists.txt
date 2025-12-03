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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(pyaerial LANGUAGES C CXX ASM)

# Though this is mentioned in the root CMakeLists.txt, in some projects, this directory
# is being added with add_subdirectory() hence the root CMakeLists.txt is not used and
# these variables are not set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

# This is by default ON in cuPHY too.
option(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

set(SOURCES_CPP pybind11/cuda_array_interface.cpp
                pybind11/pycuphy_srs_chest.cpp
                pybind11/pycuphy_srs_rx.cpp
                pybind11/pycuphy_srs_tx.cpp
                pybind11/pycuphy_srs_util.cpp
                pybind11/pycuphy_csirs_tx.cpp
                pybind11/pycuphy_csirs_rx.cpp
                pybind11/pycuphy_csirs_util.cpp
                pybind11/pycuphy_channel_est.cpp
                pybind11/pycuphy_channel_eq.cpp
                pybind11/pycuphy_noise_intf_est.cpp
                pybind11/pycuphy_cfo_ta_est.cpp
                pybind11/pycuphy_ldpc_decoder.cpp
                pybind11/pycuphy_ldpc_derate_match.cpp
                pybind11/pycuphy_rsrp.cpp
                pybind11/pycuphy_pusch_params.cpp
                pybind11/pycuphy_ldpc_util.cpp
                pybind11/pycuphy_crc_encode.cpp
                pybind11/pycuphy_crc_check.cpp
                pybind11/pycuphy_util.cpp
                pybind11/pycuphy_debug.cpp
                pybind11/pycuphy_chan_model.cpp)
add_library(pycuphycpp ${SOURCES_CPP})
target_link_libraries(pycuphycpp PUBLIC cuphy_channels cuphy chanModels ${Python3_LIBRARIES})
target_include_directories(pycuphycpp PRIVATE ../cuPHY/examples/common
                                              ../testBenches/chanModels/src)

set(SOURCES pybind11/cuda_array_interface.cpp
            pybind11/pycuphy_pybind.cpp
            pybind11/pycuphy_pdsch.cpp
            pybind11/pycuphy_dmrs.cpp
            pybind11/pycuphy_pusch.cpp
            pybind11/pycuphy_csirs_tx.cpp
            pybind11/pycuphy_csirs_rx.cpp
            pybind11/pycuphy_csirs_util.cpp
            pybind11/pycuphy_util.cpp
            pybind11/pycuphy_pdsch_params.cpp
            pybind11/pycuphy_pusch_params.cpp
            pybind11/pycuphy_channel_est.cpp
            pybind11/pycuphy_channel_eq.cpp
            pybind11/pycuphy_noise_intf_est.cpp
            pybind11/pycuphy_cfo_ta_est.cpp
            pybind11/pycuphy_srs_chest.cpp
            pybind11/pycuphy_srs_tx.cpp
            pybind11/pycuphy_srs_rx.cpp
            pybind11/pycuphy_srs_util.cpp
            pybind11/pycuphy_ldpc_decoder.cpp
            pybind11/pycuphy_ldpc_encoder.cpp
            pybind11/pycuphy_ldpc_rate_match.cpp
            pybind11/pycuphy_ldpc_derate_match.cpp
            pybind11/pycuphy_ldpc_util.cpp
            pybind11/pycuphy_crc_encode.cpp
            pybind11/pycuphy_crc_check.cpp
            pybind11/pycuphy_trt_engine.cpp
            pybind11/pycuphy_rsrp.cpp
            pybind11/pycuphy_chan_model.cpp
            pybind11/pycuphy_debug.cpp
            ../cuPHY/src/cuphy/tensor_desc.cpp
)

if (ENABLE_TESTS)
    add_subdirectory(tests/cpp)
endif()

find_package(gsl-lite REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(_pycuphy ${SOURCES})
target_link_libraries(_pycuphy PUBLIC cuphy_channels cuphy chanModels gsl-lite::gsl-lite CUDA::cudart)
target_include_directories(_pycuphy PRIVATE ../cuPHY/examples/common ../testBenches/chanModels/src)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_pycuphy PUBLIC VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Copy the necessary libraries to the output directory for pip install
function(copy_to_pycuphy_output_dir tgt)
    add_custom_command(TARGET _pycuphy POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:${tgt}>" "${CMAKE_CURRENT_SOURCE_DIR}/src/aerial/pycuphy/"
    )
endfunction()

# Remove any existing dynamic libraries from pycuphy output directory
# to avoid confusion if building with static libraries only
function(clean_up_pycuphy_output_dir)
    add_custom_command(TARGET _pycuphy POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E echo "Cleaning up pyaerial output directory"
        && "${CMAKE_COMMAND}" -E remove -f "${CMAKE_CURRENT_SOURCE_DIR}/src/aerial/pycuphy/*.so*")
endfunction()

clean_up_pycuphy_output_dir()
copy_to_pycuphy_output_dir(_pycuphy)

if (BUILD_SHARED_LIBS)
    copy_to_pycuphy_output_dir(cuphy)
    copy_to_pycuphy_output_dir(nvlog)
    copy_to_pycuphy_output_dir(chanModels)
endif()
