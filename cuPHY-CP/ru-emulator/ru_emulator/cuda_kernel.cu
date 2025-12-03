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

// UNUSED CODE

#include "utils.hpp"
#define CHECKSUM_THREADS 1024

__global__ void kernel_adler32(uint8_t * i_buf, size_t i_elems, uint32_t * out)
{
    int idx = threadIdx.x;
    int sumA = 0, sumB = 0;
    __shared__ uint32_t A[CHECKSUM_THREADS];
    __shared__ uint32_t B[CHECKSUM_THREADS];

    for (int i = idx; i < (int)i_elems; i += CHECKSUM_THREADS)
    {
        sumA += i_buf[i];
        sumB = (sumB + ( (((int)i_elems)-i) * ((int)i_buf[i])) % ADLER32_MOD);
    }

    A[idx] = sumA;
    B[idx] = sumB;
    __syncthreads();

    for (int j = CHECKSUM_THREADS/2; j>0; j/=2)
    {
        if (idx<j)
        {
            A[idx] += A[idx+j];
            B[idx] += (B[idx+j] % ADLER32_MOD);
        }
        __syncthreads();
    }

    if (idx == 0)
    {
        A[0] += 1;
        A[0] = A[0]%ADLER32_MOD;
        B[0] += ( ((int)i_elems) );
        B[0] = ( B[0] % ADLER32_MOD );
        *out = (B[0] << 16) | A[0];
    }
}

extern "C"
uint32_t launch_checksum(uint8_t * i_buf, size_t size)
{
    uint64_t t1,t2,t3,t4,t5,t6,t7,t8;
    re_dbg("Checksum thread size: {}", CHECKSUM_THREADS);
    uint32_t out;
    cudaError_t result=cudaSuccess;

    uint8_t * d_buf;
    uint32_t * h_adler;

    t1 = get_ns();
    cudaMalloc(&h_adler, sizeof(uint32_t));
    result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        re_err(AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with {} ",__FILE__, __LINE__,cudaGetErrorString(result));
    }
    t2 = get_ns();
    cudaMalloc(&d_buf, size);
    result = cudaGetLastError();
    if (cudaSuccess != result)
    {
        re_err(AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with {} ",__FILE__, __LINE__,cudaGetErrorString(result));
    }
    t3 = get_ns();
    cudaMemcpy(d_buf, i_buf, size, cudaMemcpyHostToDevice);
    t4 = get_ns();
    kernel_adler32<<<1, CHECKSUM_THREADS, CHECKSUM_THREADS * sizeof(uint32_t) * 2>>>(d_buf, size, h_adler);
    t5 = get_ns();
    cudaDeviceSynchronize();
    t6 = get_ns();

    result = cudaGetLastError();
    if (cudaSuccess != result)
        re_err(AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with {} ",__FILE__, __LINE__,cudaGetErrorString(result));

    cudaMemcpy(&out, h_adler, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    t7 = get_ns();
    cudaFree(h_adler);
    cudaFree(d_buf);
    t8 = get_ns();

    re_dbg("CUDA {:4.2f} {:X}", ((double)(get_ns() - t1))/NS_X_US, out);
    re_dbg("cudaMalloc\t\t {:4.2f} us",((double)(t3 - t1))/NS_X_US);
    re_dbg("cudaMemcpy data buf\t {:4.2f} us",((double)(t4 - t3))/NS_X_US);
    re_dbg("kernel_adler32\t {:4.2f} us",((double)(t5 - t4))/NS_X_US);
    re_dbg("cudaDeviceSync\t {:4.2f} us",((double)(t6 - t5))/NS_X_US);
    re_dbg("cudaMemcpy\t\t {:4.2f} us",((double)(t7 - t6))/NS_X_US);
    re_dbg("cudaFree\t\t {:4.2f} us",((double)(t8 - t7))/NS_X_US);

    return out;
}
