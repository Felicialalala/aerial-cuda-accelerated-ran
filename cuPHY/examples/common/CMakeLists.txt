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

# HDF5 should already be found by parent CMakeLists.txt
# If not found, this will give a clear error
if(NOT HDF5_FOUND)
    message(FATAL_ERROR "HDF5 not found. Please ensure HDF5 is installed and findable by CMake.")
endif()


# Create header-only library target
add_library(hdf5hpp INTERFACE)
add_library(hdf5hpp::hdf5hpp ALIAS hdf5hpp)

# Set include directories - header is in current source directory
target_include_directories(hdf5hpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link HDF5 dependency
target_link_libraries(hdf5hpp
    INTERFACE
        hdf5::hdf5
)

# Set C++ standard requirement
target_compile_features(hdf5hpp 
    INTERFACE 
        cxx_std_17
)

# Installation configuration
include(GNUInstallDirs)

# Install the single header file
install(
    FILES hdf5hpp.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets for CMake integration
install(
    TARGETS hdf5hpp
    EXPORT hdf5hpp-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install export configuration
install(
    EXPORT hdf5hpp-targets
    FILE hdf5hpp-targets.cmake
    NAMESPACE hdf5hpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hdf5hpp
)

# Common utilities for examples - INTERFACE library
add_library(cuphy_examples_common INTERFACE)

target_include_directories(cuphy_examples_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Add HDF5 include directories (headers only, no linking)
target_include_directories(cuphy_examples_common INTERFACE ${HDF5_INCLUDE_DIRS})
