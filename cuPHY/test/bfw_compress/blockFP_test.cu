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

#include <cstdio>
#include <algorithm>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <curand.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <arpa/inet.h>

#include "tensor_desc.hpp"
#include "cuphy_hdf5.hpp"

#include "bfw_blockFP.cuh"
#include "blockFP_test_kernels.h"

#define CUCHK(call)                                                                                                     \
    {                                                                                                                   \
        cudaError_t err = call;                                                                                         \
        if(cudaSuccess != err)                                                                                          \
        {                                                                                                               \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr);                                                                                             \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

#define CURAND_CALL(x)                                      \
    do                                                      \
    {                                                       \
        if((x) != CURAND_STATUS_SUCCESS)                    \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                            \
        }                                                   \
    } while(0)

template <typename TCompute, uint32_t N_ANT, uint32_t N_LAYERS, uint32_t MAX_PRBG_PER_LAYER, uint32_t N_THREADS>
__global__ void kcomp(const typename complex_from_scalar<TCompute>::type* input,
                      uint8_t*                                            output,
                      uint32_t                                            ngrps,
                      float                                               beta,
                      uint8_t                                             compbits,
                      int16_t                                             beam_id_offset)
{
    uint32_t tid         = threadIdx.x;
    uint32_t iprbgrp     = blockIdx.x;
    uint32_t input_index = iprbgrp * N_ANT * N_LAYERS; // (ANTENNAS, LAYERS, PRB_GRP)

    using TComplex = typename complex_from_scalar<TCompute>::type;

    int32_t compbytes = (compbits == 16) ? N_ANT * 4 : 2 * N_ANT / 8 * compbits + 1;
    if(compbits == 32) compbytes = N_ANT * sizeof(TComplex);
    compbytes += 2*(beam_id_offset>=0);
    uint32_t output_index = iprbgrp * compbytes; // (ANTENNAS, PRB_GRP, LAYERS)

    __shared__ TComplex smemBlkC[N_LAYERS][N_ANT + 1];

    // Copy the input data in shared memory, shift with -0.5
    uint32_t ty = tid / N_ANT;
    uint32_t tx = tid % N_ANT;
    for(uint32_t y = ty; y < N_LAYERS; y += N_THREADS / N_ANT)
    {
        // smemBlkC[ty + y][tx] = input[input_index + tid];
        TComplex v = input[input_index + tid];
        v.x -= 0.5f;
        v.y -= 0.5f;
        smemBlkC[y][tx] = v;
        input_index += N_THREADS;
    }

    __syncthreads();

    bfw_scale_compress_blockFP<TCompute, N_ANT + 1, N_ANT, N_LAYERS, MAX_PRBG_PER_LAYER, N_THREADS>(
        &smemBlkC[0][0],       // Shared memory input pointer for the antennas
        output + output_index, // Output pointer for the first antenna
        beta,                  // Scaling factor
        compbits,              // Number of compressed bits, if 16=uncompressed, 32=FP pass-through
        beam_id_offset,        // Beam ID offset
        tid,                   // 1D thread rank
        ngrps);                // Stride between 2 layers (number of PRB groups)
}

// Simple decompression kernel. Decompress N_CHUNK PRBs with a group of threads,
// write them in transposed order
template <typename TCompute, uint32_t N_ANT, uint32_t N_CHUNK, uint32_t N_LAYERS, uint32_t N_THREADS>
__global__ void kdecomp(const uint8_t*                                input,
                        typename complex_from_scalar<TCompute>::type* output,
                        uint32_t                                      ngrps,
                        float                                         invbeta,
                        uint8_t                                       compbits,
                        bool                                          packed_beam_id)
{
    using TComplex = typename complex_from_scalar<TCompute>::type;
    __shared__ TComplex smem[N_CHUNK][N_ANT + 1];
    uint32_t            smem_stride = N_ANT + 1;

    uint32_t compbytes = (compbits == 16) ? 4 * N_ANT : 2 * N_ANT / 8 * compbits + 1;
    if(compbits == 32) compbytes = N_ANT * sizeof(TComplex);
    compbytes += 2*(packed_beam_id);
    uint32_t first_prb = blockIdx.x * N_CHUNK;
    input += first_prb * compbytes;

    // Actual number of PRBs to decompress
    int n_decomp = min(N_CHUNK, ngrps * N_LAYERS - first_prb);
    bfw_decompress_blockFP<TCompute, N_ANT, N_THREADS>(input, &smem[0][0], smem_stride, invbeta, n_decomp, compbits, packed_beam_id, threadIdx.x);

    __syncthreads();

    // Write the PRBS, one thread per antenna
    int tx = threadIdx.x % N_ANT;
    int ty = threadIdx.x / N_ANT;
    for(int y = ty; y < n_decomp; y += N_THREADS / N_ANT)
    {
        // PRB index
        uint32_t prb_in = first_prb + y;
        // Convert from (ANTENNAS, PRB_GRP, LAYERS) to (ANTENNAS, LAYERS, PRB_GRP)
        uint32_t ilayer = prb_in / ngrps;
        uint32_t igrp   = prb_in % ngrps;

        uint32_t output_idx = (igrp * N_LAYERS + ilayer) * N_ANT + tx;
        TComplex v          = smem[y][tx];
        v.x += 0.5f;
        v.y += 0.5f;
        output[output_idx] = v;
    }
}

// These constants are fixed for this test configuration as they are template parameters for kernels.
constexpr uint32_t N_ANT_CONST    = 64;
constexpr uint32_t N_LAYERS_CONST = 16;
constexpr uint32_t MAX_PRBG_PER_LAYER = 137;

void execute_blockfp_processing(
    uint32_t nfreqs,
    uint8_t compbits,
    float beta,
    int16_t beam_id_offset,
    const std::string& filename,
    bool verbose
)
{
    // Use the constexpr values for N_ANT and N_LAYERS for kernel instantiation
    constexpr uint32_t N_ANT    = N_ANT_CONST;
    constexpr uint32_t N_LAYERS = N_LAYERS_CONST;

    using TCompute = float;
    using TComplex = typename complex_from_scalar<TCompute>::type;

    // Allocate input data, initialize with random numbers or from file
    cuphy::stream cuStrm;
    curandGenerator_t gen = nullptr; // Initialize to nullptr
    TComplex* input;
    TComplex* decomp;
    size_t    input_size = N_ANT * N_LAYERS * nfreqs * sizeof(TComplex);
    CUCHK(cudaMallocManaged((void**)&input, input_size));
    CUCHK(cudaMallocManaged((void**)&decomp, input_size));

    if(!filename.empty()) {
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(filename.c_str());
        cuphy::tensor_pinned compressed_output_ref = cuphy::tensor_from_dataset(fInput.open_dataset("bfwCompBuf0"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStrm.handle());
        cuphy::tensor_pinned compressed_input_ref = cuphy::tensor_from_dataset(fInput.open_dataset("bfwBuf0"), CUPHY_C_64F, cuphy::tensor_flags::align_tight, cuStrm.handle());
    } else {
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
        CURAND_CALL(curandGenerateUniform(gen, reinterpret_cast<float*>(input), 2 * N_ANT * N_LAYERS * nfreqs));    

    }
    
    if (verbose) {
        std::cout << "Running with parameters:" << std::endl;
        std::cout << "  N_ANT: " << N_ANT << std::endl;
        std::cout << "  N_LAYERS: " << N_LAYERS << std::endl;
        std::cout << "  nfreqs: " << nfreqs << std::endl;
        std::cout << "  compbits: " << static_cast<int>(compbits) << std::endl;
        std::cout << "  beta: " << beta << std::endl;
        if (!filename.empty()) {
            std::cout << "  filename: " << filename << std::endl;
        }
    }

    // Allocate output data
    uint8_t* output;
    size_t   output_size;
    if(compbits >= 16) // uncompressed
    {
        output_size = nfreqs * N_LAYERS * 2 * N_ANT * compbits / 8;
    }
    else
    {
        output_size = nfreqs * N_LAYERS * (2 * N_ANT * compbits / 8 + 1 + 2*(beam_id_offset>=0)); // +1 byte for compparam, 2 for beam_id
    }
    CUCHK(cudaMallocManaged((void**)&output, output_size));

    // One block per layer
    constexpr uint32_t N_THREADS = 64;
    
    kcomp<TCompute, N_ANT, N_LAYERS, MAX_PRBG_PER_LAYER, N_THREADS><<<nfreqs, N_THREADS>>>(input, output, nfreqs, beta, compbits, beam_id_offset);
    CUCHK(cudaGetLastError()); // Check for errors after kernel launch

    // Decompress using chunks of PRBs
    constexpr uint32_t chunksize = 16;
    dim3               blocks((nfreqs * N_LAYERS + chunksize - 1) / chunksize);
    float              invbeta = 1.0f / beta;
    kdecomp<TCompute, N_ANT, chunksize, N_LAYERS, N_THREADS><<<blocks, N_THREADS>>>(output, decomp, nfreqs, invbeta, compbits, beam_id_offset>=0);
    CUCHK(cudaGetLastError()); // Check for errors after kernel launch

    CUCHK(cudaDeviceSynchronize());
    TComplex maxerr = {0};
    for(int i = 0; i < N_ANT * N_LAYERS * nfreqs; i++)
    {
        TComplex diff;
        diff.x = fabsf(input[i].x - decomp[i].x);
        diff.y = fabsf(input[i].y - decomp[i].y);
        if(diff.x > maxerr.x)
            maxerr.x = diff.x;
        if(diff.y > maxerr.y)
            maxerr.y = diff.y;
    }
    
    // Calculate bundle_size for verbose printing, ensure it matches kernel logic
    size_t bundle_size;
    if (compbits == 32) bundle_size = N_ANT * sizeof(TComplex);
    else if (compbits == 16) bundle_size = N_ANT * 4; // Corresponds to 2 * N_ANT * 16 / 8
    else bundle_size = (2 * N_ANT * compbits / 8 + 1);
    bundle_size += 2*(beam_id_offset>=0);

    if (verbose) {
        printf("Compression details:\n");
        printf("  Compression bits: %d\n", compbits);
        printf("  Compression beta: %f\n", beta);
        printf("  Number of antennas: %u\n", N_ANT);
        printf("  Number of layers: %u\n", N_LAYERS);
        printf("  Number of frequencies: %u\n", nfreqs);
        printf("  Calculated bundle size per layer output: %zu bytes\n", bundle_size);
        if (output_size > 0 && nfreqs * N_LAYERS > 0) { // Ensure there's output to print
            printf("  First bundle (first layer output, max 64 bytes shown):\n\t");
            size_t print_len = std::min(bundle_size, (size_t)64); // Print at most 64 bytes or bundle_size
            for(size_t i = 0; i < print_len; i++) {
                printf("%02X ", output[i]);
                if(i % 16 == 15 && i < print_len -1) printf("\n\t");
            }
            printf("\n");

            if(beam_id_offset >= 0 && compbits != 32 && compbits != 16) { // Beam IDs are relevant for compressed formats
                printf("Beam IDs (if applicable and packed): \n\t");
                // The beam ID is at the start of each bundle, after the comp param
                // The original loop for beam IDs was: for(int i = 1; i < output_size; i += bundle_size)
                // This assumes beam_id is after the first byte (compParam). Offset might be different if bundle_size includes it differently.
                // The definition of compbytes in kcomp: (compbits == 16) ? N_ANT * 4 : 2 * N_ANT / 8 * compbits + 1; if(compbits == 32) compbytes = N_ANT * sizeof(TComplex); compbytes += 2*(beam_id_offset>=0);
                // So beam_id is at the end of the data payload. Let's adjust the printing.
                // The byte containing compParam is output[0] for the first bundle. The beam_id is at output[bundle_size-2] and output[bundle_size-1]
                // The loop should iterate through bundles.
                for(size_t layer_idx = 0; layer_idx < nfreqs * N_LAYERS; ++layer_idx) {
                    if (layer_idx * bundle_size + bundle_size > output_size) break; // Boundary check
                    // The beam_id is stored after the compressed data and the exponent byte.
                    // Original logic for compbytes: `2 * N_ANT / 8 * compbits + 1` is data + exponent.
                    // Then `compbytes += 2*(beam_id_offset>=0);` means beam_id is at the end.
                    // So, for a given bundle starting at `bundle_start_idx`, beam_id is at `bundle_start_idx + bundle_size - 2`.
                    size_t bundle_start_idx = layer_idx * bundle_size;
                    uint16_t beam_id_val = ntohs(*reinterpret_cast<uint16_t*>(&output[bundle_start_idx + bundle_size - 2]));
                    printf("%04X ", beam_id_val);
                    if((layer_idx + 1) % 8 == 0 && layer_idx < (nfreqs*N_LAYERS-1) ) printf("\n\t"); // Print 8 per line
                    if (layer_idx >= 63) { printf("... (further beam IDs truncated)"); break; } // Limit printing
                }
                printf("\n");
            }
        }
    }
    
    printf("Max error = (%f, %f)\n", maxerr.x, maxerr.y);
    if(maxerr.x > pow(2, -1*compbits+1) || maxerr.y > pow(2, -1*compbits+1)) {
        printf("Max error is too high (>%f). Exiting.\n", pow(2, -1*compbits+1));
        exit(EXIT_FAILURE);
    }
    
    // Free memory
    CUCHK(cudaFree(input));
    CUCHK(cudaFree(output));
    CUCHK(cudaFree(decomp));
    if (gen) { // Only destroy if it was created
        curandDestroyGenerator(gen);
    }
}