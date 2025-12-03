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

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TARGET aerial_metrics)
set(SOURCES aerial_metrics.cpp
            aerial_metrics.hpp
            prometheus_metrics_backend.cpp
            prometheus_metrics_backend.hpp)

add_library(${TARGET} SHARED ${SOURCES})
find_package(prometheus-cpp REQUIRED)
target_link_libraries(${TARGET} prometheus-cpp::core prometheus-cpp::pull)
target_include_directories(${TARGET} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
