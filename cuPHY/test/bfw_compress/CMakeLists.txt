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

# Add the blockFP_test
add_executable(blockFP_test blockFP_test.cu blockFP_test_main.cpp)
target_include_directories(blockFP_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/compression_decompression
)
target_link_libraries(blockFP_test CUDA::cudart nvlog CLI11::CLI11 cuphy_hdf5 CUDA::curand)

# Set CUDA properties for the test
set_target_properties(blockFP_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add the test to ctest
add_test(NAME blockFP_test COMMAND blockFP_test)