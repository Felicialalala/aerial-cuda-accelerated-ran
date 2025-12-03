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

add_executable(test_hdf5 test_hdf5.cpp dummy.cu)
target_include_directories(test_hdf5 PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_hdf5 cuphy)
target_link_libraries(test_hdf5 cuphy_hdf5)
target_link_libraries(test_hdf5 gtest gtest_main)
add_test(NAME test_hdf5 COMMAND test_hdf5 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/hdf5)

add_executable(test_hdf5_host test_hdf5_host.cpp)
target_include_directories(test_hdf5_host PRIVATE ${PROJECT_SOURCE_DIR}/examples/common ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_hdf5_host ${HDF5_C_LIBRARIES})
target_link_libraries(test_hdf5_host gtest gtest_main)
add_test(NAME test_hdf5_host COMMAND test_hdf5_host WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/hdf5)
