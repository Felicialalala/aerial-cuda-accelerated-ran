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

########################################################################
# test_ldpc_internal_app_addr
set(SRCS test_ldpc_internal_app_addr.cu
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc_half2.cpp)
add_executable(test_ldpc_internal_app_addr ${SRCS})
target_include_directories(test_ldpc_internal_app_addr PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_app_addr cuphy)
target_link_libraries(test_ldpc_internal_app_addr gtest gtest_main)
target_compile_options(test_ldpc_internal_app_addr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_app_addr COMMAND test_ldpc_internal_app_addr)
#add_test(NAME test_ldpc_internal COMMAND test_ldpc_internal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/error_correction)

########################################################################
# test_ldpc_internal_rc
add_executable(test_ldpc_internal_rc test_ldpc_internal_rc.cu)
target_include_directories(test_ldpc_internal_rc PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_rc cuphy)
target_link_libraries(test_ldpc_internal_rc gtest gtest_main)
target_compile_options(test_ldpc_internal_rc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_rc COMMAND test_ldpc_internal_rc)

########################################################################
# test_ldpc_internal_loader
add_executable(test_ldpc_internal_loader test_ldpc_internal_loader.cu)
target_include_directories(test_ldpc_internal_loader PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_loader cuphy)
target_link_libraries(test_ldpc_internal_loader gtest gtest_main)
target_compile_options(test_ldpc_internal_loader PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_loader COMMAND test_ldpc_internal_loader)

#                        --keep
#                       --cudart static



