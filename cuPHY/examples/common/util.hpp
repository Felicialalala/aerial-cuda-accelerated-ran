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

#if !defined(UTIL_HPP_INCLUDED_)
#define UTIL_HPP_INCLUDED_

#include "cuphy.h"

void gpu_ms_delay(uint32_t delay_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
void gpu_us_delay(uint32_t delay_us, int gpuId = 0, cudaStream_t cuStrm = 0, bool singleThrdBlk = false);
void gpu_ms_sleep(uint32_t sleep_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
void gpu_ns_delay_until(uint64_t* start_time_d, uint64_t time_offset_ns, cudaStream_t cuStrm);
void gpu_empty_kernel(cudaStream_t cuStrm = 0);
void get_sm_ids(int gpuId, uint32_t* pSmIds, uint32_t smIdsCnt, cudaStream_t cuStrm = 0, uint32_t delay_us = 1000);
void get_gpu_time(uint64_t *ptimer_d, cudaStream_t cuStrm);

#endif // !defined(UTIL_HPP_INCLUDED_)
