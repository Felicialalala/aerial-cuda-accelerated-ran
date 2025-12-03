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

#include <cuda_runtime.h>

#include "gpu_blockFP.h"
#include "gpu_uLaw.h"
#include "gpu_blockScaling.h"

// Elapsed in ms
#define ELAPSED(t1, t2) ((t2.tv_sec - t1.tv_sec) * 1E3 + (t2.tv_usec - t1.tv_usec) * 1E-3)

__global__ void initUncompressed(int16_t *data, int nval)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nval)
        data[index] = index & 0xffff;
}

__global__ void compress(int16_t *input, uint8_t *compressed, int nprb, int compbits)
{
    // Single CTA compression
    compress_blockFP<false>((int4 *)input, compressed,
                            nprb, compbits, threadIdx.x, blockDim.x);
}

__global__ void uncompress(uint8_t *compressed, int16_t *uncompressed, int npackets, int prbPerPacket, int compbits)
{
    int warpId = threadIdx.x / 32;
    int nwarps = blockDim.x / 32;
    if (warpId > 0)
    {
        // Each worker warp decompresses one packet
        for (int ipacket = warpId - 1; ipacket < npackets; ipacket += nwarps - 1)
        {
            int compPacketbOffset = ipacket * prbPerPacket * (compbits * 3 + 1);
            int packetOffset = ipacket * prbPerPacket * 24;
            decompress_blockFP<false>((unsigned char *)(compressed + compPacketbOffset),
                                      (int4 *)(uncompressed + packetOffset),
                                      prbPerPacket, compbits, (int)(threadIdx.x & 31), 32);
        }
    }
}

__global__ void uncompress_2blocks(uint8_t *compressed, int16_t *uncompressed, int npackets, int prbPerPacket, int compbits)
{
    int warpId = threadIdx.x / 32;
    int nwarps = blockDim.x / 32;

    int firstPacket = blockIdx.x * (nwarps - 1) + warpId;
    int packetStride = gridDim.x * (nwarps - 1);
    if (warpId > 0)
    {
        // Each worker warp decompresses one packet
        for (int ipacket = firstPacket; ipacket < npackets; ipacket += packetStride)
        {
            int compPacketbOffset = ipacket * prbPerPacket * (compbits * 3 + 1);
            int packetOffset = ipacket * prbPerPacket * 24;
            decompress_blockFP<false>((unsigned char *)(compressed + compPacketbOffset),
                                      (int4 *)(uncompressed + packetOffset),
                                      prbPerPacket, compbits, (int)(threadIdx.x & 31), 32);
        }
    }
}


int main()
{
    int npackets = 560;
    int prbPerPacket = 30;
    int compbits = 9;
    int nprb = npackets * prbPerPacket;
    int nval = nprb * 24;

    int16_t *input;
    uint8_t *compressed;
    int16_t *uncompressed;

    size_t uncompSize = nval * sizeof(uint16_t);
    size_t compSize = nprb * (3 * compbits + 1);

    cudaMallocManaged((void **)&input, uncompSize);
    cudaMallocManaged((void **)&compressed, compSize);
    cudaMallocManaged((void **)&uncompressed, uncompSize);

    initUncompressed<<<(nval + 1023) / 1024, 1024>>>(input, nval);
    compress<<<1, 512>>>(input, compressed, nprb, compbits);
    uncompress<<<1, 512>>>(compressed, uncompressed, npackets, prbPerPacket, compbits);
    // uncompress_2blocks<<<2, 1024>>>(compressed, uncompressed, npackets, prbPerPacket, compbits);
    cudaDeviceSynchronize();

    return cudaDeviceSynchronize();
}