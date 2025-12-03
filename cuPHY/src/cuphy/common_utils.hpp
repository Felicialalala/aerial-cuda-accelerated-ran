/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUPHY_COMMON_UTILS_HPP
#define CUPHY_COMMON_UTILS_HPP

#include "nvlog.hpp"

// Ensure macro arg is only used once to avoid unintented
// side effects
// Call cudaGetLastError() to reset the error, so it is not
// caught in a subsequent call. There are exceptions such as cudaErrorIllegalAddress etc.
#define CUDA_CHECK(expr_to_check)\
_Pragma("vcast_dont_instrument_start")            \
do {                                              \
    const cudaError_t result = expr_to_check;     \
    if(result != cudaSuccess)                     \
    {                                             \
        cudaError_t last_error = cudaGetLastError();              \
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,      \
                "CUDA Runtime Error: {}:{}:{} (last error {})",   \
                __FILE__,                                         \
                __LINE__,                                         \
                cudaGetErrorString(result),                       \
                cudaGetErrorString(last_error));                  \
        if(cudaSuccess != last_error)                             \
        {                                                         \
            throw std::runtime_error(cudaGetErrorString(result)); \
        }                                                         \
    }                                                             \
} while (0) \
_Pragma("vcast_dont_instrument_end")                              \

#define CUDA_CHECK_NO_THROW(expr_to_check) \
do {            \
    const cudaError_t result = expr_to_check;              \
    _Pragma("vcast_dont_instrument_start");                \
    if(result != cudaSuccess)                              \
    {                                                      \
        cudaError_t last_error = cudaGetLastError();              \
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,      \
                "CUDA Runtime Error: {}:{}:{} (last error {})",   \
                __FILE__,                                         \
                __LINE__,                                         \
                cudaGetErrorString(result),                       \
                cudaGetErrorString(last_error));                  \
    }                                                             \
    _Pragma("vcast_dont_instrument_end");                         \
} while (0) \

#define CUDA_CHECK_PRINTF(expr_to_check)                          \
    const cudaError_t result = expr_to_check;                     \
    if(result != cudaSuccess)                                     \
    {                                                             \
        cudaError_t last_error = cudaGetLastError();              \
        fprintf(stderr,                                           \
                "CUDA Runtime Error: %s:%i:%s (last error %s)",   \
                __FILE__,                                         \
                __LINE__,                                         \
                cudaGetErrorString(result),                       \
                cudaGetErrorString(last_error));                  \
        if(cudaSuccess != last_error)                             \
        {                                                         \
            throw std::runtime_error(cudaGetErrorString(result)); \
        }                                                         \
    }

#endif // CUPHY_COMMON_UTILS_HPP
