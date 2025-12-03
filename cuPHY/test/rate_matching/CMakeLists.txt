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

add_executable(testRateMatching "${PROJECT_SOURCE_DIR}/src/cuphy/rate_matching/testRateMatching.cpp" dummy.cu)
add_dependencies(testRateMatching CRCLUTGEN)
target_link_libraries(testRateMatching cuphy cuphy_channels cuphy_hdf5)
target_link_libraries(testRateMatching gtest gtest_main)
add_test(testRateMatching testRateMatching)

add_executable(testRMtvs "${PROJECT_SOURCE_DIR}/src/cuphy/rate_matching/testRMtvs.cpp" dummy.cu)
add_dependencies(testRMtvs CRCLUTGEN)
target_link_libraries(testRMtvs cuphy cuphy_channels cuphy_hdf5)
target_link_libraries(testRMtvs gtest gtest_main)

