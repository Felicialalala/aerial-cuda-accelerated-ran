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
project(scfl2adapter LANGUAGES C CXX CUDA)

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# ----------------------------------------------------------------------
# Subdirectories
add_subdirectory(lib)
add_subdirectory(scf_app)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()