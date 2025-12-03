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
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>

#include "gpu_blockFP.h"

// Beta coefficients for the UL gain scaling with a 4-bit exponent.
// From Tim Martin:
// Beta = 65504 / sqrt(FS)
// FS = 2 ^ (2 * (2^exponentbits - 1 + mantissabits - 1))
// E.g with 4 exponent bits and 9 mantissa bits, beta = 65504 / sqrt (2^46)
const static __constant__ float c_betadecomp[] = {
    0.0f,                     // 0 bits, not used
    1.9990234375f,            // 1-bit
    .99951171875f,            // 2-bit
    .499755859375f,           // 3-bit
    .2498779296875f,          // 4-bit
    .12493896484375f,         // 5-bit
    .062469482421875f,        // 6-bit
    .0312347412109375f,       // 7-bit
    .01561737060546875f,      // 8-bit
    .007808685302734375f,     // 9-bit
    .0039043426513671875f,    // 10-bit
    .00195217132568359375f,   // 11-bit
    .000976085662841796875f,  // 12-bit
    .0004880428314208984375f, // 13-bit
    .00024402141571044921f,   // 14-bit
    .0001220107078552246f,    // 15-bit
    1.9990234375f             // 16-bit, uncompressed (no exponent)
};

#define NTHREADS 512

#define BETA 26373509.9f // For compression

template <int inplace>
__global__ void compress(half *prb, uint8_t *output, float beta,
                         int32_t nprb, int compbits)
{
    __shared__ uint8_t smem[16 * NTHREADS]; // Shared memory, need 16 bytes per thread
    int32_t prbBlock = NTHREADS / 4;
    int32_t firstPrb = blockIdx.x * prbBlock;
    int32_t nprbBlock = min(prbBlock, nprb - firstPrb);
    int32_t offset_in = firstPrb * 24;
    int32_t offset_out = (inplace || compbits == 16) ? firstPrb * 48 : firstPrb * (3 * compbits + 1);
    scale_compress_blockFP<inplace>(prb + offset_in, output + offset_out, beta,
                                    nprbBlock, compbits, threadIdx.x, NTHREADS, smem);
}

template <int inplace>
__global__ void decompress(uint8_t *input, half *prb, int32_t nprb, int32_t compbits, float invbeta)
{
    int32_t prbBlock = NTHREADS / 4;
    int32_t firstPrb = blockIdx.x * prbBlock;
    int32_t nprbBlock = min(prbBlock, nprb - firstPrb);
    int32_t offset_in = (inplace || compbits == 16) ? firstPrb * 48 : firstPrb * (3 * compbits + 1);
    int32_t offset_out = firstPrb * 24;
    // Using 1/BETA for QC purposes instead of: float scalar = c_betadecomp[compbits];
    float scalar = invbeta;
    decompress_scale_blockFP<inplace>(input + offset_in, prb + offset_out, scalar,
                                      nprbBlock, compbits, threadIdx.x, NTHREADS);
}

void diff(half *orig, half *decomp, int32_t nprb)
{
    float maxerr = 0.0f;
    int imax = 0;
    for (int i = 0; i < nprb * 24; i++)
    {
        float diff = fabsf((float)orig[i] - (float)decomp[i]);
        if (diff > maxerr)
        {
            maxerr = diff;
            imax = i;
        }
    }
    printf("Max error = %e at i=%d, orig = %f vs %f\n",
           maxerr, imax, (float)orig[imax], (float)decomp[imax]);
}

int main(int argc, char **argv)
{
    const int nprb = 273 * 16 * 14;

    half *prb, *h_prb, *h_ref;
    uint8_t *output;
    size_t prbsize = nprb * 24 * sizeof(half);
    cudaMalloc((void **)&prb, prbsize);
    cudaMalloc((void **)&output, prbsize);
    cudaMallocHost((void **)&h_prb, prbsize);
    cudaMallocHost((void **)&h_ref, prbsize);

    cudaEvent_t t0, t1, t2;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    float t_comp, t_decomp;

    for (int j = 0; j < nprb; j++)
        for (int i = 0; i < 24; i += 2)
        {
            h_ref[j * 24 + i] = (half)(0.5f + i * .04f - j * 1.6667E-5);
            h_ref[j * 24 + i + 1] = (half) - (0.5f + (i + 1) * .04f - j * 1.6667E-5);
        }

    float beta = BETA;

    dim3 threads(NTHREADS, 1, 1);
    dim3 blocks((nprb * 4 - 1) / NTHREADS + 1, 1, 1);

    printf("Benchmark using %d PRBs\n", nprb);
    printf("\nWARNING: Timings are not very accurate for short workloads!\n");
    printf("Use the profiler accurate timings.\n\n");

    for (int compbits = 7; compbits <= 16; compbits++)
    {
        float beta = sqrtf(powf(2.0f, 30.0f + 2.0f * compbits - 1) * 31.622777f) / (12.0f * 273.0f);
        if (compbits == 16)
            beta = 10000.0f; //sqrtf (powf (2.0f, 30.0f) * 31.622777f) / (12.0f * 273.0f);

        printf("\n*** %2d compressed bits ***\n", compbits);

        char msg[128];
        sprintf(msg, "%d_bits", compbits);
        nvtxRangePushA(msg);

        printf("1) Out-of-place:\n");
        cudaMemcpy(prb, h_ref, prbsize, cudaMemcpyDefault);
        cudaEventRecord(t0);
        compress<false><<<blocks, threads>>>(prb, output, beta, nprb, compbits);
        cudaEventRecord(t1);
        decompress<false><<<blocks, threads>>>(output, prb, nprb, compbits, 1. / beta);
        cudaEventRecord(t2);
        if (cudaMemcpy(h_prb, prb, prbsize, cudaMemcpyDefault) != cudaSuccess)
            printf("Error kernel\n");
        else
            diff(h_ref, h_prb, nprb);
        cudaEventElapsedTime(&t_comp, t0, t1);
        cudaEventElapsedTime(&t_decomp, t1, t2);
        printf("Tcomp = %.3f ms, Tdecomp time = %.3f ms\n", t_comp, t_decomp);

        printf("2) In-place:\n");
        cudaMemcpy(prb, h_ref, prbsize, cudaMemcpyDefault);
        cudaEventRecord(t0);
        compress<true><<<blocks, threads>>>(prb, nullptr, beta, nprb, compbits);
        cudaEventRecord(t1);
        decompress<true><<<blocks, threads>>>((uint8_t *)prb, nullptr, nprb, compbits, 1. / beta);
        cudaEventRecord(t2);
        if (cudaMemcpy(h_prb, prb, prbsize, cudaMemcpyDefault) != cudaSuccess)
            printf("Error kernel\n");
        else
            diff(h_ref, h_prb, nprb);
        cudaEventElapsedTime(&t_comp, t0, t1);
        cudaEventElapsedTime(&t_decomp, t1, t2);
        printf("Tcomp = %.3f ms, Tdecomp time = %.3f ms\n", t_comp, t_decomp);

        nvtxRangePop();
    }

    return 0;
}