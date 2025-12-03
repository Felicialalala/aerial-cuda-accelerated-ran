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

# ---------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# ---------------------------------------------------------------------
# Project options
project(cuPHYL2Adapter LANGUAGES C CXX CUDA)

# ---------------------------------------------------------------------
# Additional packages
find_package(CUDAToolkit REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------
# Subdirectories
add_subdirectory(lib)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gt_common_libs/CMakeLists.txt")
    add_subdirectory(gt_common_libs)
    #add_subdirectory(cuPHY)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/cuphydriver/CMakeLists.txt")
        add_subdirectory(cuphydriver)
    endif ()
endif ()
