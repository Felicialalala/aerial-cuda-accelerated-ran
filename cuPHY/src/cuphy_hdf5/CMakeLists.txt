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

add_library(cuphy_hdf5 STATIC cuphy_hdf5.cpp)
set_property(TARGET cuphy_hdf5 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(cuphy_hdf5 PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                             ${HDF5_INCLUDE_DIRS})
# CUDA::cudart is needed here since hdf5 is calling cudaStreamSynchronize so we need proper
# link line order here for consumers of cuBB
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuphy_hdf5 PUBLIC cuphy ${HDF5_C_LIBRARIES} CUDA::cudart)

# Header-only INTERFACE library for cuphy_hdf5 header
add_library(cuphy_hdf5_h INTERFACE)
add_library(cuphy::cuphy_hdf5_h ALIAS cuphy_hdf5_h)

target_include_directories(cuphy_hdf5_h INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(cuphy_hdf5_h INTERFACE cxx_std_20)

install(TARGETS cuphy_hdf5 DESTINATION lib)
install(FILES cuphy_hdf5.h DESTINATION include)
