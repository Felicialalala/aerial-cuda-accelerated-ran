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
# criticalPathCudaApi
add_executable(criticalPathCudaApi criticalPathCudaApi.cu)
target_include_directories(criticalPathCudaApi PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(criticalPathCudaApi cuphy gtest gtest_main)
target_compile_options(criticalPathCudaApi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME criticalPathCudaApi COMMAND criticalPathCudaApi)

#                        --keep