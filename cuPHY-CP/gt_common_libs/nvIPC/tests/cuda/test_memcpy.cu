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

// #include "../common/book.h"
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define SIZE (64 * 1024 * 1024)

#define HANDLE_ERROR(x)                                                                 \
    do                                                                                  \
    {                                                                                   \
        if((x) != cudaSuccess) { printf("Error %s line%d\n", __FUNCTION__, __LINE__); } \
    } while(0)
#define HANDLE_NULL(x)

float cuda_malloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *       a = NULL, *dev_a = NULL;
    float       elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int*)malloc(size * sizeof(*a));
    if (a == NULL)
    {
        return 0.0;
    }
    memset(a, 0, size * sizeof(*a));
    HANDLE_NULL(a);
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            HANDLE_ERROR(
                cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(
                cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_register_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *       a, *dev_a;
    float       elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    a = (int*)malloc(size * sizeof(*a));
    if (a == NULL)
    {
        return 0.0;
    }
    memset(a, 0, size * sizeof(*a));
    HANDLE_NULL(a);

    int flag = cudaHostRegisterPortable | cudaHostRegisterMapped;
    HANDLE_ERROR(cudaHostRegister(a, size * sizeof(*a), flag));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            HANDLE_ERROR(
                cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(
                cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    HANDLE_ERROR(cudaHostUnregister(a));
    free(a);
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_alloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *       a, *dev_a;
    float       elapsedTime;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(
        cudaHostAlloc((void**)&a, size * sizeof(*a), cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            HANDLE_ERROR(
                cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
        else
            HANDLE_ERROR(
                cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    HANDLE_ERROR(cudaFreeHost(a));
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    return elapsedTime;
}

int main(void)
{
    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    // try it with cudaMalloc
    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using cudaMalloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using cudaMalloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cudaHostAlloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cudaHostAlloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    // try it with cudaMalloc
    elapsedTime = cuda_host_register_test(SIZE, true);
    printf("Time using cudaHostRegister:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_register_test(SIZE, false);
    printf("Time using cudaHostRegister:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));
}
