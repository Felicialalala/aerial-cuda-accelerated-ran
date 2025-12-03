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

# string(APPEND CMAKE_CUDA_FLAGS " --cudart shared -lineinfo ")
set(CMAKE_CUDA_STANDARD 17)

# include_directories("/usr/local/cuda-10.1/samples/common/inc")
# include_directories("/usr/local/cuda/include")

set(SOURCES test_cuda.cu)
add_library(cudaipctest STATIC ${SOURCES})
target_link_libraries(cudaipctest nvipc)

set(SOURCES test_memcpy.cu)
add_executable(test_memcopy ${SOURCES})

set (SOURCES cuda_ipc.cu)
add_executable(cuda_ipc ${SOURCES})
target_link_libraries(cuda_ipc nvipc cudaipctest -pthread)
