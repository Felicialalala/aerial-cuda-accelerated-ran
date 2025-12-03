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

template <int nworkers, int workerSize>
class producerConsumerSync
{
    // Fine grain producer - consumers synchronization,
    // with one producer warp and nworkers consumers (of workerSize threads each).
    // Using 2 x nworkers barriers, implementing arrive and wait on each barrier.
    // Each CTA can use up to 16 barriers = 8 workers max.
public:
    static_assert(nworkers <= 8);
    static_assert(workerSize % 32 == 0);

    __device__ inline static void waitForWorker(int worker)
    {
        // Sync on barrier A = barrier[worker]
        asm("bar.sync %0, %1;" ::"r"(worker), "r"(workerSize + 32));
    }

    __device__ inline static void workerReady(int worker)
    {
        // Arrive at barrier A = barrier[worker]
        asm("bar.arrive %0, %1;" ::"r"(worker), "r"(workerSize + 32));
    }

    __device__ inline static void waitForProducer(int worker)
    {
        // Sync on barrier B = barrier[nworkers+worker]
        asm("bar.sync %0, %1;" ::"r"(nworkers + worker), "r"(workerSize + 32));
    }

    __device__ inline static void producerReady(int worker)
    {
        // Arrive at barrier B = barrier[nworkers+worker]
        asm("bar.arrive %0, %1;" ::"r"(nworkers + worker), "r"(workerSize + 32));
    }
};

template <int nworkers, int workerSize>
__global__ void decompress(uint8_t *compressed, int16_t *uncompressed, int npackets, int prbPerPacket, int compbits)
{
    // E.g with nworkers=4, workerSize=64, total = 32 + 4 x 64 = 288 threads
    // Threads 0:31 = primary
    // Threads 32:95 = worker 0
    // Threads 96:159 = worker 1
    // Threads 160:223 = worker 2
    // Threads 224:287 = worker 3.
    __shared__ int shmPkt[nworkers];

    producerConsumerSync<nworkers, workerSize> pcs;

    if (threadIdx.x < 32) // Primary warp
    {
        int worker = 0;
        // Loop on all the packets
        for (int ipacket = 0; ipacket < npackets; ipacket++)
        {
            // Wait for worker to be ready to consume
            pcs.waitForWorker(worker);

            if (threadIdx.x == 0)
                shmPkt[worker] = ipacket;

            // Signal worker the packet has been delivered
            pcs.producerReady(worker);

            // Rotate worker
            worker++;
            if (worker == nworkers)
                worker = 0;
        }
        // Signal all the workers there's no more work (packet = -1)
        for (int worker = 0; worker < nworkers; worker++)
        {
            pcs.waitForWorker(worker);
            if (threadIdx.x == 0)
                shmPkt[worker] = -1;
            pcs.producerReady(worker);
        }
    }
    else // workers
    {
        int workerId = (threadIdx.x - 32) / workerSize;
        int workerLaneId = (threadIdx.x - 32) % workerSize;

        // Loop until we get a negative packet number
        while (1)
        {
            // Signal the producer: ready to consume
            pcs.workerReady(workerId);

            // Wait for producer to deliver a packet
            pcs.waitForProducer(workerId);

            // Read packet
            int packetId = shmPkt[workerId];
            // if (workerLaneId == 0)
            //     printf("Worker %d got packet %d\n", workerId, packetId);
            if (packetId == -1)
                return;

            // Uncompress packet
            int compPacketbOffset = packetId * prbPerPacket * (compbits * 3 + 1);
            int packetOffset = packetId * prbPerPacket * 24;
            decompress_blockFP<false, workerSize>((unsigned char *)(compressed + compPacketbOffset),
                                                  (int4 *)(uncompressed + packetOffset),
                                                  prbPerPacket, compbits, workerLaneId);
        }
    }
}

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

int main()
{
    constexpr int nworkers = 8;
    constexpr int workerSize = 64;
    const int npackets = 560;
    const int prbPerPacket = 30;
    int compbits = 9;
    int nprb = npackets * prbPerPacket;
    int nval = nprb * 24;

    int16_t *input;
    uint8_t *compressed;
    int16_t *uncompressed;

    size_t uncompSize = nval * sizeof(uint16_t);
    size_t compSize = nprb * (3 * compbits + 1);

    cudaMalloc((void **)&input, uncompSize);
    cudaMalloc((void **)&compressed, compSize);
    cudaMalloc((void **)&uncompressed, uncompSize);

    initUncompressed<<<(nval + 1023) / 1024, 1024>>>(input, nval);
    compress<<<1, 512>>>(input, compressed, nprb, compbits);

    dim3 threads(nworkers * workerSize + 32, 1, 1);
    decompress<nworkers, workerSize><<<1, threads>>>(compressed, uncompressed, npackets, prbPerPacket, compbits);

    return cudaDeviceSynchronize();
}
