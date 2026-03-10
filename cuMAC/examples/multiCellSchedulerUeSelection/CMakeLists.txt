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

add_executable(multiCellSchedulerUeSelection)
target_sources(multiCellSchedulerUeSelection PRIVATE
    main.cpp
    ../network.cu
    ../trafficModel/trafficGenerator.cpp
    ../customScheduler/CustomUePrgScheduler.cpp
    ../customScheduler/GnnRlPolicyRuntime.cpp
    ../ml/trtEngine.cpp
    ../rlReplay/ReplayWriter.cpp
)
find_library(NVINFER libnvinfer.so)
find_library(NVONNXPARSER libnvonnxparser.so)
find_library(NVINFER_PLUGIN libnvinfer_plugin.so)
if (NOT NVINFER OR NOT NVONNXPARSER OR NOT NVINFER_PLUGIN)
    message(FATAL_ERROR "TensorRT libraries not found: NVINFER=${NVINFER}, NVONNXPARSER=${NVONNXPARSER}, NVINFER_PLUGIN=${NVINFER_PLUGIN}")
endif()

target_link_libraries(
    multiCellSchedulerUeSelection
    PRIVATE
        cumac
        ${HDF5_CXX_LIBRARIES}
        chanModels
        ${NVINFER}
        ${NVONNXPARSER}
        ${NVINFER_PLUGIN}
)
target_include_directories(
    multiCellSchedulerUeSelection
    PRIVATE
        ${HDF5_CXX_INCLUDE_DIRS}
        ${CHAN_MODELS_SRC_INCLUDE_DIR}
        ${cumac_SOURCE_DIR}/examples/tools
        ${cumac_SOURCE_DIR}/examples/ml
)
