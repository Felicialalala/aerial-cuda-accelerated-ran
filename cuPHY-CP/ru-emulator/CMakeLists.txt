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

# ----------------------------------------------------------------------
# Set the version number for this project
set(ruemulator_VERSION_MAJOR 0)
set(ruemulator_VERSION_MINOR 1)

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------------------------------
# Project options
project(ruemulator LANGUAGES C CXX)



# ----------------------------------------------------------------------
# Check if build is with REPO or SUBMODULES
if (NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    message(VERBOSE "RU building with REPO")
    SET(SUBMODULE_BUILD FALSE)
else()
    message(VERBOSE "RU building with SUBMODULE")
    SET(SUBMODULE_BUILD TRUE)
endif()

# ----------------------------------------------------------------------
# Additional packages
find_package(CUDAToolkit REQUIRED)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)


if (AERIAL_RU_WAR)
    add_definitions(-DAERIAL_RU_WAR=1)
endif (AERIAL_RU_WAR)


# ----------------------------------------------------------------------
# Subdirectories
add_subdirectory(ru_emulator)
if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
