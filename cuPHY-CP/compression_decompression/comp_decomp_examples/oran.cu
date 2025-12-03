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
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>

#include "gpu_blockFP.h"
#include "gpu_uLaw.h"
#include "gpu_blockScaling.h"

// Elapsed in ms
#define ELAPSED(t1, t2) ((t2.tv_sec - t1.tv_sec) * 1E3 + (t2.tv_usec - t1.tv_usec) * 1E-3)

// Using 32 PRBs per CTA, enough to keep up to 128 threads busy.
// Warning: At 512 PRBs per CTA, the shared memory needed will exceed 48KB, and would
// require explicit opt-in on architectures supporting it.
#define NPRB_CTA 64
#define NTHREADS 256

enum
{
    BLOCKFP,
    BLOCKSCALING,
    ULAW
};

template <int algo, int network_order_input>
__global__ void compress(short *input, unsigned char *output, int nprb, int compbits, int niter)
{
    __shared__ unsigned char smout[NPRB_CTA * 49]; // 49B per PRB if using 16b comp (= larger than input!)

    // Offset and number of PRBs for this CTA
    int first_prb = blockIdx.x * NPRB_CTA;
    int nprb_block = min(NPRB_CTA, nprb - first_prb);

    int4 *iptr = (int4 *) (input + first_prb * 24); // Assuming proper 128b alignment

    // Loop on shared memory for the benchmark, to avoid being bandwidth bound
    for (int iter = 0; iter < niter; iter++)
    {
        if (algo == BLOCKFP)
            compress_blockFP<network_order_input>(iptr, smout, nprb_block,
                                                  compbits, threadIdx.x, blockDim.x);
        else if (algo == ULAW)
            compress_uLaw<network_order_input>(iptr, smout, nprb_block,
                                               compbits, threadIdx.x, blockDim.x);
        else if (algo == BLOCKSCALING)
            compress_blockScaling<network_order_input, false>(iptr, smout, nprb_block,
                                                             compbits, threadIdx.x, blockDim.x);
        __syncthreads();
    }

    // Write the compressed data back to global memory
    size_t compprb_bytes = 3 * compbits + 1; // (24 * compbits) / 8 + 1
    int offset = first_prb * compprb_bytes;
    for (int i = threadIdx.x; i < nprb_block * compprb_bytes; i += blockDim.x)
        output[offset + i] = smout[i];
}


// Decompression benchmark kernel, iterating in shared memory
template <int algo, bool network_order_output>
__global__ void decompress(unsigned char *input, short *output, int nprb, int compbits, int niter)
{
    __shared__ unsigned char smin[NPRB_CTA * 49]; // 49B per PRB if using 16b comp (= larger than output!)
    __shared__ int4 smout[NPRB_CTA * 3];

    // Offset and number of PRBs for this CTA
    int first_prb = blockIdx.x * NPRB_CTA;
    int nprb_block = min(NPRB_CTA, nprb - first_prb);

    // Load the data into shared memory
    size_t compprb_bytes = (24 * compbits) / 8 + 1;
    size_t offset = first_prb * compprb_bytes;
    for (int i = threadIdx.x; i < nprb_block * compprb_bytes; i += blockDim.x)
        smin[i] = input[offset + i];
    __syncthreads();

    // Loop on shared memory for the benchmark
    for (int iter = 0; iter < niter; iter++)
    {
        if (algo == BLOCKFP)
            decompress_blockFP<network_order_output>(smin, (int4 *)smout, nprb_block,
                                                     compbits, threadIdx.x, blockDim.x);
        else if (algo == ULAW)
            decompress_uLaw<network_order_output>(smin, (int4 *)smout, nprb_block,
                                                  compbits, threadIdx.x, blockDim.x);
        else if (algo == BLOCKSCALING)
            decompress_blockScaling<network_order_output>(smin, (int4 *)smout, nprb_block,
                                                          compbits, threadIdx.x, blockDim.x);
        __syncthreads();
    }
    // Write the uncompressed data back to global memory
    offset = first_prb * 24UL;
    for (int i = threadIdx.x; i < nprb_block * 24; i += blockDim.x)
        output[offset + i] = ((short *)smout)[i];
}

void short2network(short *v, int n)
{
    for (int i = 0; i < n; i++)
        v[i] = ((v[i] >> 8) & 0xff) | (v[i] << 8);
}

template <int algo>
void checkResults(short *input, short *result, int nprb, int compbits)
{
    char name[13];
    for (int i = 0; i < nprb; i++)
    {
        int vin = input[i * 24];
        int vres = result[i * 24];
        int vmax = vin;
        int vmin = vin;
        int mindiff = vin - vres;
        int maxdiff = vin - vres;
        for (int j = 1; j < 24; j++)
        {
            vin = input[i * 24 + j];
            vres = result[i * 24 + j];
            vmax = max(vmax, vin);
            vmin = min(vmin, vin);
            mindiff = min(mindiff, vin - vres);
            maxdiff = max(maxdiff, vin - vres);
        }
        if (algo == BLOCKFP)
            sprintf(name, "BLOCKFP     ");
        else if (algo == ULAW)
            sprintf(name, "ULAW        ");
        else if (algo == BLOCKSCALING)
            sprintf(name, "BLOCKSCALING");
        else
        {
            printf("ERROR: QC not supported for this algorithm\n");
            return;
        }
        if (mindiff != 0 || maxdiff != 0)
        {
            if (algo == BLOCKFP)
            {
                uint maxamp = max(vmax, abs(vmin) - 1);
                if (maxamp < (1 << compbits) && mindiff != 0)
                {
                    printf("Error, PRB %d, compbits=%d, mindiff = %d\n", i, compbits, mindiff);
                    printf("Compbits BLOCKFP = %d:Failed\n", compbits);
                    return;
                }
                int shift = 33 - __builtin_ia32_lzcnt_u32(maxamp) - compbits;
                if (shift < 0)
                    shift = 0;
                int error = 1 << shift;
                if (maxdiff >= error)
                {
                    printf("Error, PRB %d maxdiff = %d > %d, mindiff = %d, amplitudes = (%d, %d) on %d bits\n",
                           i, maxdiff, error, mindiff, vmin, vmax, compbits);
                    printf("Compbits = %d:Failed\n", compbits);
                    return;
                }
            }
            else if (algo == ULAW)
            {
                int error = 4 << (16 - compbits);
                if (maxdiff >= error || -mindiff >= error)
                {
                    printf("Error, PRB %d maxdiff = %d > %d, mindiff = %d, amplitudes = (%d, %d) on %d bits\n",
                           i, maxdiff, error, mindiff, vmin, vmax, compbits);
                    printf("Compbits ULAW = %d:Failed\n", compbits);
                    return;
                }

                uint maxamp = max(vmax, abs(vmin));
            }
            else if (algo == BLOCKSCALING)
            {
                uint maxamp = max(vmax, abs(vmin) - 1);
                int shift = shift = __builtin_ia32_lzcnt_u32(maxamp) - 17;
                if (shift > 7)
                    shift = 7;
                int error = 1 << (16 - compbits - shift);
                if (16 - compbits - shift < 0) error = 1;
                if (maxdiff > error)
                {
                    printf("Error, PRB %d maxdiff = %d > %d, mindiff = %d, amplitudes = (%d, %d) on %d bits\n",
                           i, maxdiff, error, mindiff, vmin, vmax, compbits);
                    printf("Compbits BLOCKSCALING = %d:Failed\n", compbits);
                    return;
                }
            }
        }
    }
    printf("Compbits = %2d %s: PASSED\n", compbits, name);
}

template <int algo>
void runBenchmark(short *input, unsigned char *output, short *decomp,
                  const int nprb, const int minbits, const int maxbits, const int niter)
{
    if (algo == BLOCKFP)
        printf("\nBlock FP compression:\n");
    else if (algo == ULAW)
        printf("\nuLaw compression:\n");
    else if (algo == BLOCKSCALING)
        printf("\nBlock scaling compression:\n");

    dim3 blocks((nprb - 1) / NPRB_CTA + 1, 1, 1);
    dim3 threads(NTHREADS, 1, 1);

    for (int compbits = minbits; compbits <= maxbits; compbits++)
    {
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        compress<algo, false><<<blocks, threads>>>(input, output, nprb, compbits, niter);
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            printf("Comp: Kernel failed\n");
            return;
        }
        gettimeofday(&t2, NULL);
        printf("Compression   %2d bits, time = %.3f ms\n",
               compbits, ELAPSED(t1, t2));

        gettimeofday(&t1, NULL);
        decompress<algo, false><<<blocks, threads>>>(output, decomp, nprb, compbits, niter);
        if (cudaDeviceSynchronize() != cudaSuccess)
        {
            printf("Decomp: Kernel failed\n");
            return;
        }
        gettimeofday(&t2, NULL);
        printf("Decompression %2d bits, time = %.3f ms\n",
               compbits, ELAPSED(t1, t2));
    }
}

int main()
{
    const bool debug = true;
    const bool benchmark = true;

    const int minbits = 7;
    const int maxbits = 16;

    // Debugging
    if (debug)
    {
        const int nprb = 1000;
        short *input, *decomp;
        unsigned char *output;
        cudaMallocManaged((void **)&input, nprb * 24 * sizeof(short));
        cudaMallocManaged((void **)&output, nprb * 49);
        cudaMallocManaged((void **)&decomp, nprb * 24 * sizeof(short));
        dim3 blocks((nprb - 1) / NPRB_CTA + 1, 1, 1);
        dim3 threads(NTHREADS, 1, 1);

        // for (int i = 0; i < nprb * 24; i++)
        //     input[i] = 255;
        // compress<BLOCKFP, false><<<blocks, threads>>>(input, output, nprb, 9, 1);
        // cudaDeviceSynchronize();

        for (int i = 0; i < nprb * 24; i++)
            input[i] = i & 1 ? -i : i;

        // int ib = 21;
        // int bb = 12;
        // blocks.x = 1;
        // compress<BLOCKSCALING, false><<<blocks, threads>>>(input + ib*24, output + ib*24, 1, bb, 1);
        // decompress<BLOCKSCALING, false><<<blocks, threads>>>(output + ib*24, decomp + ib*24, 1, bb, 1);
        // cudaDeviceSynchronize();
        // checkResults<BLOCKSCALING>(input+ ib*24, decomp+ ib*24, 1, bb);
        // return 0;

        printf("\nValidation tests:\n");
        for (int compbits = minbits; compbits <= maxbits; compbits++)
        {
            // BlockFP
            compress<BLOCKFP, false><<<blocks, threads>>>(input, output, nprb, compbits, 1);
            decompress<BLOCKFP, false><<<blocks, threads>>>(output, decomp, nprb, compbits, 1);
            cudaDeviceSynchronize();
            checkResults<BLOCKFP>(input, decomp, nprb, compbits);

            // uLaw
            compress<ULAW, false><<<blocks, threads>>>(input, output, nprb, compbits, 1);
            decompress<ULAW, false><<<blocks, threads>>>(output, decomp, nprb, compbits, 1);
            cudaDeviceSynchronize();
            checkResults<ULAW>(input, decomp, nprb, compbits);

            // blockScaling
            compress<BLOCKSCALING, false><<<blocks, threads>>>(input, output, nprb, compbits, 1);
            decompress<BLOCKSCALING, false><<<blocks, threads>>>(output, decomp, nprb, compbits, 1);
            cudaDeviceSynchronize();
            checkResults<BLOCKSCALING>(input, decomp, nprb, compbits);
        }
        printf("\n");
        cudaFree(input);
        cudaFree(output);
        cudaFree(decomp);
    }

    // Benchmarking
    if (benchmark)
    {
        // Assuming new data is coming at 50 Gbps every 33us
        // we're processing a block of 5E10 / 3.3E-5 = 1.65MB? Small!!!
        // At 48B per PRB, that's around 34K PRBs
        // const int nprb = 34375;
        // Or 16 layers x 273 PRBs x 14 symbols = 61152 PRBs
        const int nprb = 61152;
        const int niter = 1;
        short *h_input, *input, *decomp;
        unsigned char *output;
        size_t input_size = nprb * 24 * sizeof(short);
        size_t output_size = nprb * 49UL;
        cudaMalloc((void **)&input, input_size);
        cudaMalloc((void **)&output, output_size);
        cudaMalloc((void **)&decomp, input_size);
        cudaMallocHost((void **)&h_input, input_size);

        for (int i = 0; i < nprb * 24; i++)
            h_input[i] = i & 1 ? -i : i;
        cudaMemcpy(input, h_input, input_size, cudaMemcpyDefault);

        runBenchmark<BLOCKFP>(input, output, decomp, nprb, minbits, maxbits, niter);

        runBenchmark<ULAW>(input, output, decomp, nprb, minbits, maxbits, niter);

        runBenchmark<BLOCKSCALING>(input, output, decomp, nprb, minbits, maxbits, niter);

        cudaFree(input);
        cudaFree(output);
        cudaFree(decomp);
        cudaFreeHost(h_input);
    }

    return 0;
}
