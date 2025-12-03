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

add_executable(l2_muUeGrp_test)
target_sources(l2_muUeGrp_test PRIVATE l2_muUeGrp_test.cpp)
target_include_directories(l2_muUeGrp_test PUBLIC ${HDF5_CXX_INCLUDE_DIRS} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(l2_muUeGrp_test PRIVATE ${HDF5_CXX_LIBRARIES} cumac pthread nvlog nvipc yaml cumac_msg)

add_executable(cumac_muUeGrp_test)
target_sources(cumac_muUeGrp_test PRIVATE cumac_muUeGrp_test.cu)
target_include_directories(cumac_muUeGrp_test PUBLIC ${HDF5_CXX_INCLUDE_DIRS} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(cumac_muUeGrp_test PRIVATE ${HDF5_CXX_LIBRARIES} cumac pthread nvlog nvipc yaml cumac_msg)