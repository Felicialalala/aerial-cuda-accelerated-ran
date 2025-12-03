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

# Minimum CMake version required.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# Project options
project(data_lake LANGUAGES C CXX CUDA)
add_library(data_lake SHARED data_lake.cpp)

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_include_directories(data_lake PUBLIC ./ ../.././cuPHY/src/cuphy/ ../scfl2adapter/lib/scf_5g_fapi/ ../aerial-fh-driver/include/aerial-fh-driver/)
target_link_libraries(data_lake PUBLIC slot_command oran_utils nvipc nvlog app_config clickhouse-cpp-lib cityhash)


# Configure E3 Agent (always compiled, runtime configurable)
message(STATUS "Configuring E3 Agent ZMQ dependencies")

# Add E3 Agent source file
target_sources(data_lake PRIVATE e3_agent.cpp)

# ZMQ is already installed and we know the paths
set(ZMQ_INCLUDE_DIRS "/usr/include")
set(ZMQ_LIBRARIES "zmq")

message(STATUS "Using ZMQ include: ${ZMQ_INCLUDE_DIRS}")
message(STATUS "Using ZMQ library: ${ZMQ_LIBRARIES}")

# Add includes and libs
target_include_directories(data_lake PUBLIC 
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(data_lake PUBLIC 
    ${ZMQ_LIBRARIES}
)

# Add nlohmann-json for E3 messaging
find_package(nlohmann_json REQUIRED)
target_link_libraries(data_lake PUBLIC nlohmann_json::nlohmann_json)
