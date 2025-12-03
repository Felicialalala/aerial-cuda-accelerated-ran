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

set(SOURCES main.cpp
            launch_pattern.cpp
            test_mac_configs.cpp
            test_mac.cpp
            fapi_handler.cpp
            fapi_validate.cpp
            signal_handler.cpp
            dummy.cu)

if(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    add_definitions(-DENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
endif()

if (ENABLE_CUMAC AND TARGET cumac)
    add_compile_definitions(AERIAL_CUMAC_ENABLE)
    set(SOURCES ${SOURCES} cumac_pattern.cpp cumac_handler.cpp cumac_configs.cpp cumac_validate.cpp)
endif ()

if (TARGET scf_5g_fapi)
    add_compile_definitions(SCF_L2ADAPTER_EXISTS)
    set(SOURCES ${SOURCES} scf_fapi_handler.cpp)
    set(L2ADAPTER_LIBS ${L2ADAPTER_LIBS} scf_5g_fapi)
endif ()

message(VERBOSE "TESTMAC: SOURCES=${SOURCES}")
message(VERBOSE "TESTMAC: L2ADAPTER_LIBS=${L2ADAPTER_LIBS}")

set(TARGET test_mac)
add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find system dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-0.1)
find_package(Threads REQUIRED)

target_link_libraries(${TARGET} PRIVATE
    Threads::Threads
    nvlog
    nvipc
    nvphy_generic
    app_config
    cuphyoamlib
    cuphy_h
    cuphy_hdf5_h
    scf_5g_fapi_h
    oran_utils
    cuphy_examples_common
    fmtlog-shared
    fmt
    yaml-cpp
    ${YAML_LIBRARIES}
    hdf5_serial
    grpc++_reflection
)

# Add YAML include directories
target_include_directories(${TARGET} PRIVATE ${YAML_INCLUDE_DIRS})

if (ENABLE_CUMAC AND TARGET cumac)
    target_link_libraries(${TARGET} PRIVATE nvphy cumac_msg)
endif ()
