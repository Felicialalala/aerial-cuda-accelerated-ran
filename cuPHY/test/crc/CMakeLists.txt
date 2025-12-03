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

add_executable(crcTest "${PROJECT_SOURCE_DIR}/src/cuphy/crc/crc.cu" "${PROJECT_SOURCE_DIR}/src/cuphy/crc/crcTest.cu")
#add_executable(crcTime "${PROJECT_SOURCE_DIR}/src/cuphy/crc/crcTime.cu")
add_executable(prepareCrcBuffersTest "${PROJECT_SOURCE_DIR}/src/cuphy/crc/prepareCrcBuffersTest.cu")
target_link_libraries(crcTest cuphy)
#target_link_libraries(crcTime cuphy)
target_link_libraries(prepareCrcBuffersTest cuphy)
target_link_libraries(crcTest gtest gtest_main)
target_link_libraries(prepareCrcBuffersTest gtest gtest_main)
add_test(crcTest crcTest)
add_test(prepareCrcBuffersTest prepareCrcBuffersTest)
