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

#include <stdio.h>

#include "test_kernel.h"

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    asm volatile ( "mov.u64 %0, %globaltimer;"   : "=l"( globaltimer ) );
    return globaltimer;
}

__global__ void test_kernel(uint64_t id, int delay_ns, uint64_t t_host_launch)
{
    uint64_t t_now = __globaltimer();
    uint64_t t_stop = t_now + delay_ns;

    printf("[GPUNow:%lu HostLaunch:%lu]: Inside test_kernel %lu with delay %d ns\n",t_now,t_host_launch,id,delay_ns);
    while (__globaltimer() < t_stop);
    printf("[GPUNow:%lu HostLaunch:%lu]: Exiting test_kernel %lu with delay %d ns\n",__globaltimer(),t_host_launch,id,delay_ns);
}

void launch_test_kernel(cudaStream_t& stream, uint64_t id, int delay_ns, uint64_t t_host_launch)
{
    test_kernel<<<1,1,0,stream>>>(id, delay_ns, t_host_launch);
}