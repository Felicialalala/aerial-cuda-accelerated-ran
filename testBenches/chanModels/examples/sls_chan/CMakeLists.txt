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

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add example executable
add_executable(sls_chan_ex 
    dummy.cu
    sls_chan_ex.cpp
)

# Link against required libraries
target_link_libraries(sls_chan_ex
    PRIVATE
    chanModels
    ${CUDA_LIBRARIES}
    ${Python3_LIBRARIES}
    curand cuphy_hdf5 chanModels
)

if(USE_MATPLOTLIB)
    target_compile_definitions(sls_chan_ex
        PRIVATE
        USE_MATPLOTLIB
    )
endif()

# Include directories
target_include_directories(sls_chan_ex
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src  # Add path to chanModels source
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ../../src  # Relative path to chanModels source
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}  # Add standard library include paths
)

