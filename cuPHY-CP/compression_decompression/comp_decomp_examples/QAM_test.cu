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
#include <unistd.h>
#include <stdlib.h>
#include <curand.h>
#include <cstdint>
#include <cuda_fp16.h>

#include "QAM_param.cuh"
#include "QAM_comp.cuh"
#include "QAM_decomp.cuh"

#define CUCHK(call)                                                              \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (cudaSuccess != err)                                                  \
        {                                                                        \
            fprintf(stderr, "Cuda error in file '%s' line %i : %s.\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err));                          \
            fflush(stderr);                                                      \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

// Kernel to convert CURAND's FP32 data into FP16
__global__ void fp32tofp16(float *input, half *output, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = (half)(input[i] - 0.5f); // Center around 0
}

// Scale the PRBs by the proper scaler depending on the mask,
// and set the missing PRBs to zero
__global__ void prep_prbs(half **__restrict__ list_inputs,       // Input data, contiguous array [N_ANTENNAS x N_SYMBOLS x N_PRBS_PER_SYMBOL x 24] half
                          QamPrbParam **__restrict__ prb_params, // PRB parameters, for each list
                          const float2 *__restrict__ scalers,    // PRB scaling, for each list
                          const int *__restrict__ nprbs,         // Number of PRBs in each list
                          int nlists)                            // Number of PRB lists
{
    int listid = blockIdx.x;
    int nprbs_list = nprbs[listid];
    float2 scale = scalers[listid];
    int nwarps = blockDim.x / 32;
    int laneid = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;
    if (laneid >= 24)
        return;

    for (int i = warpid; i < nprbs_list; i += nwarps)
    {
        QamPrbParam par = prb_params[listid][i];
        half value = list_inputs[listid][i * 24 + laneid];
        if (par.mask_on<0>(laneid / 2))
            value *= sqrtf(2.0f) * scale.x;
        else if (par.mask_on<1>(laneid / 2))
            value *= sqrtf(2.0f) * scale.y;
        else
            value = (half)0.0f;
        list_inputs[listid][i * 24 + laneid] = value;
    }
}

struct RndGen
{
    curandGenerator_t gen;
    RndGen()
    {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    }
    ~RndGen()
    {
        curandDestroyGenerator(gen);
    }
    void randomize(float *data, int n)
    {
        curandGenerateUniform(gen, data, n);
    }
    void randomize(half *data, int n)
    {
        float *tmp;
        cudaMalloc((void **)&tmp, n * sizeof(float));
        curandGenerateUniform(gen, tmp, n);
        fp32tofp16<<<(n + 1023) / 1024, 1024>>>(tmp, data, n);
        cudaDeviceSynchronize();
        cudaFree(tmp);
    }
    void randomize(uint *data, int n)
    {
        curandGenerate(gen, data, n);
    }
};

#define MAXMODCOMPPRBBYTES 12 // QAM256 = 4 bits per element = 1 byte per RE = 12 bytes per PRB

void getWidth(int i, QamListParam::qamwidth &width0, QamListParam::qamwidth &width1)
{
    switch (i % 5)
    {
    case 0:
        width0 = QamListParam::MODCOMP_BPSK;
        break;
    case 1:
        width0 = QamListParam::MODCOMP_QPSK;
        break;
    case 2:
        width0 = QamListParam::MODCOMP_QAM16;
        break;
    case 3:
        width0 = QamListParam::MODCOMP_QAM64;
        break;
    case 4:
        width0 = QamListParam::MODCOMP_QAM256;
    }
    switch ((i / 5) % 5)
    {
    case 0:
        width1 = QamListParam::MODCOMP_BPSK;
        break;
    case 1:
        width1 = QamListParam::MODCOMP_QPSK;
        break;
    case 2:
        width1 = QamListParam::MODCOMP_QAM16;
        break;
    case 3:
        width1 = QamListParam::MODCOMP_QAM64;
        break;
    case 4:
        width1 = QamListParam::MODCOMP_QAM256;
    }
}

int main()
{
    RndGen rndgen;

    const int nlists = 40;

    int *nprbs;
    half **inputs, **decomp, **cpu_decomp;
    uint8_t **outputs, **cpu_outputs;
    QamListParam *list_params;
    QamPrbParam **prb_params;
    CUCHK(cudaMallocManaged((void **)&nprbs, nlists * sizeof(int)));
    CUCHK(cudaMallocManaged((void **)&list_params, nlists * sizeof(QamListParam)));
    CUCHK(cudaMallocManaged((void **)&inputs, nlists * sizeof(half *)));
    CUCHK(cudaMallocManaged((void **)&decomp, nlists * sizeof(half *)));
    CUCHK(cudaMallocManaged((void **)&cpu_decomp, nlists * sizeof(half *)));
    CUCHK(cudaMallocManaged((void **)&outputs, nlists * sizeof(uint8_t *)));
    CUCHK(cudaMallocManaged((void **)&cpu_outputs, nlists * sizeof(uint8_t *)));
    CUCHK(cudaMallocManaged((void **)&prb_params, nlists * sizeof(QamPrbParam *)));

    // Generate random scaling factors for each list
    float2 *scalers;
    CUCHK(cudaMallocManaged((void **)&scalers, nlists * sizeof(float2)));
    rndgen.randomize(reinterpret_cast<float *>(scalers), 2 * nlists);
    CUCHK(cudaDeviceSynchronize());
    for (int i = 0; i < nlists; i++)
        scalers[i] = make_float2(scalers[i].x * 256.0f, scalers[i].y * 256.0f);

    for (int i = 0; i < nlists; i++)
    {
        nprbs[i] = max(1, ((i + 1) * 100 % 128));
        CUCHK(cudaMallocManaged((void **)&inputs[i], nprbs[i] * 24 * sizeof(half)));
        CUCHK(cudaMallocManaged((void **)&decomp[i], nprbs[i] * 24 * sizeof(half)));
        CUCHK(cudaMallocManaged((void **)&cpu_decomp[i], nprbs[i] * 24 * sizeof(half)));
        CUCHK(cudaMallocManaged((void **)&outputs[i], nprbs[i] * MAXMODCOMPPRBBYTES));
        CUCHK(cudaMallocManaged((void **)&cpu_outputs[i], nprbs[i] * MAXMODCOMPPRBBYTES));
        CUCHK(cudaMemset(outputs[i], 0, nprbs[i] * MAXMODCOMPPRBBYTES));
        CUCHK(cudaMallocManaged((void **)&prb_params[i], nprbs[i] * sizeof(QamPrbParam)));

        // Generate masks, alternate between single and dual masks
        uint *tmpmasks;
        CUCHK(cudaMallocManaged((void **)&tmpmasks, nprbs[i] * sizeof(uint)));
        bool dualmasks = i % 3 > 0;
        bool missingprbs = i % 3 == 2;
        if (dualmasks)
            rndgen.randomize(tmpmasks, nprbs[i]);
        else
            cudaMemset(tmpmasks, 0xff, nprbs[i] * sizeof(uint)); // Only one mask, mask0
        CUCHK(cudaDeviceSynchronize());
        for (int j = 0; j < nprbs[i]; j++)
            if (missingprbs)
            {
                uint mask0 = tmpmasks[j] & 0xfff;
                uint mask1 = (~mask0 & (tmpmasks[i] >> 16)) & 0xfff;
                prb_params[i][j].set(mask0, mask1);
            }
            else
                prb_params[i][j].set(tmpmasks[j] & 0xfff, (tmpmasks[j] & 0xfff) ^ 0xfff);

        // Pick 2 QAM formats, and set the shift bits
        QamListParam::qamwidth width0, width1;
        getWidth(i, width0, width1);
        bool shift0 = dualmasks && width0 != QamListParam::MODCOMP_BPSK &&
                      QamListParam::get_bits_per_value(width0) > QamListParam::get_bits_per_value(width1);
        bool shift1 = dualmasks && width1 != QamListParam::MODCOMP_BPSK &&
                      QamListParam::get_bits_per_value(width1) > QamListParam::get_bits_per_value(width0);
        list_params[i].set(width0, shift0, shift1);

        // Generate random PRB values
        rndgen.randomize(inputs[i], nprbs[i] * 24);
    }
    // Prep the PRB values for all the lists
    prep_prbs<<<nlists, 256>>>(inputs, prb_params, scalers, nprbs, nlists);
    CUCHK(cudaDeviceSynchronize());

    // Compress
    QAM_Comp::gpu_compress_QAM_lists(inputs, list_params, prb_params, scalers, outputs, nprbs, nlists);
    QAM_Comp::cpu_compress_QAM_lists(inputs, list_params, prb_params, scalers, cpu_outputs, nprbs, nlists);

    CUCHK(cudaDeviceSynchronize());

    // Compare the compressed data (bitwise)
    int errs = 0;
    for (int i = 0; i < nlists; i++)
    {
        QamListParam lp = list_params[i];
        int prb_bytes = 3 * lp.get_bits_per_value();
        for (int j = 0; j < nprbs[i] * prb_bytes; j++)
            errs += (outputs[i][j] == cpu_outputs[i][j] ? 0 : 1);
    }
    printf("Compression on CPU and GPU : %s (%d errors)\n", errs ? "mismatch" : "binary match", errs);

    // Uncompress
    QAM_Decomp::gpu_decompress_QAM_lists(outputs, list_params, prb_params, scalers, decomp, nprbs, nlists);
    QAM_Decomp::gpu_decompress_QAM_lists(cpu_outputs, list_params, prb_params, scalers, cpu_decomp, nprbs, nlists);

    CUCHK(cudaDeviceSynchronize());

    // Compare the CPU and GPU decompressed result
    errs = 0;
    for (int i = 0; i < nlists; i++)
        for (int j = 0; j < nprbs[i] * 24; j++)
            errs += (decomp[i][j] == cpu_decomp[i][j] ? 0 : 1);
    printf("Decompression on CPU and GPU : %s (%d errors)\n", errs ? "mismatch" : "binary match", errs);

    // Compare and make sure the max error is correct for the scaler and QAM format.
    // Missing REs must remain zero
    int gpu_errs = 0;
    int cpu_errs = 0;
    for (int i = 0; i < nlists; i++)
    {
        QamListParam lp = list_params[i];
        float maxerr0 = lp.get_i2f_fact() * scalers[i].x * sqrtf(2.0f);
        float maxerr1 = lp.get_i2f_fact() * scalers[i].y * sqrtf(2.0f);
        for (int j = 0; j < nprbs[i]; j++)
        {
            QamPrbParam par = prb_params[i][j];
            for (int ielem = 0; ielem < 12; ielem++)
            {
                float gpu_diff_i = fabsf(decomp[i][j * 24 + 2 * ielem] - inputs[i][j * 24 + 2 * ielem]);
                float gpu_diff_q = fabsf(decomp[i][j * 24 + 2 * ielem + 1] - inputs[i][j * 24 + 2 * ielem + 1]);
                float cpu_diff_i = fabsf(cpu_decomp[i][j * 24 + 2 * ielem] - inputs[i][j * 24 + 2 * ielem]);
                float cpu_diff_q = fabsf(cpu_decomp[i][j * 24 + 2 * ielem + 1] - inputs[i][j * 24 + 2 * ielem + 1]);
                if (par.mask_on<0>(ielem))
                {
                    if (gpu_diff_i > maxerr0)
                    {
                        gpu_errs++;
                    }
                    if (gpu_diff_q > maxerr0)
                    {
                        gpu_errs++;
                    }
                    // gpu_errs += gpu_diff_i <= maxerr0 ? 0 : 1;
                    // gpu_errs += gpu_diff_q <= maxerr0 ? 0 : 1;
                    cpu_errs += cpu_diff_i <= maxerr0 ? 0 : 1;
                    cpu_errs += cpu_diff_q <= maxerr0 ? 0 : 1;
                }
                else if (par.mask_on<1>(ielem))
                {
                    if (gpu_diff_i > maxerr1)
                    {
                        gpu_errs++;
                    }
                    if (gpu_diff_q > maxerr1)
                    {
                        gpu_errs++;
                    }
                    // gpu_errs += gpu_diff_i <= maxerr1 ? 0 : 1;
                    // gpu_errs += gpu_diff_q <= maxerr1 ? 0 : 1;
                    cpu_errs += cpu_diff_i <= maxerr1 ? 0 : 1;
                    cpu_errs += cpu_diff_q <= maxerr1 ? 0 : 1;
                }
                else
                {
                    // Expecting zero error on missing REs
                    if (gpu_diff_i != 0.0f)
                    {
                        gpu_errs++;
                    }
                    if (gpu_diff_q != 0.0f)
                    {
                        gpu_errs++;
                    }
                    // gpu_errs += gpu_diff_i == 0.0f ? 0 : 1;
                    // gpu_errs += gpu_diff_q == 0.0f ? 0 : 1;
                    cpu_errs += cpu_diff_i == 0.0f ? 0 : 1;
                    cpu_errs += cpu_diff_q == 0.0f ? 0 : 1;
                }
            }
        }
    }
    printf("GPU results after decompression : %d errors\n", gpu_errs);
    printf("CPU results after decompression : %d errors\n", cpu_errs);

    return 0;
}