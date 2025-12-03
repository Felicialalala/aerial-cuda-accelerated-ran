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

#include <cuda_fp16.h>
#include "gpu_packing.h"

// ****************************************************************************
// scale_compress_fixed:
// -----------------------
// Scale and convert FP16 PRBs to fixed point
// input: FP16 input array, must be aligned on 128-bit boundary
// prb_ptrs: Individual PRB pointers, must be 16-bit aligned.
//           Each PRB must have a size of (3 * compbits) bytes,
//           A nullptr value means the PRB is invalid.
// beta: Scaling factor.
// nprb: Number of PRBs to compress (some may be invalid, based on the prb_ptr value)
// compbits: Number of fixed point bits per value.
// tid: Thread ID participating in the compression (threads must be coalesced).
// nthreads: Number of threads participating in the compression, one or more full warps.
// sm: shared memory pointer to accelerate data packing, need 16 bytes per thread.
// sm_prb_ptr: shared memory pointer to accelerate pointer sharing,
//             size = (nthreads / 4) * sizeof (void *)

__device__ inline void scale_compress_fixed(const half *__restrict__ input,
                                            uint8_t **__restrict__ prb_ptrs,
                                            const float beta,
                                            int32_t nprb,
                                            int32_t compbits,
                                            int32_t tid,
                                            int32_t nthreads,
                                            uint8_t *sm,
                                            uint8_t **sm_prb_ptr)
{
    // Use 3 threads per PRB, 8 values per thread, 10 PRBs per warp.
    // Threads 30, 31 of each warp aren't doing any useful work
    int warpId = tid / 32;
    int warpLane = tid % 32;
    int activeThread = warpLane < 30;
    int laneid = warpLane % 3;
    int prbid = warpLane / 3 + warpId * 10;
    int ctaprbs = (nthreads / 32) * 10;
    if (!activeThread)
        prbid--; // Threads (30,31) will load the same input as threads (27,28)

    int4 *input_vec = (int4 *)input;
    uint8_t *prefetch_prb_ptr = nullptr;
    int4 prefetch_vin;

    // Prefetch first value. Only reading if the PRB is valid
    if (prbid < nprb)
    {
        prefetch_vin = input_vec[prbid * 3 + laneid];
        prefetch_prb_ptr = prb_ptrs[prbid];
    }

    // Loop on all the PRBs.
    // If nprb and nthreads are known at compile time, this loop can be optimized out or unrolled.
    for (int prbloop = 0; prbloop < nprb; prbloop += ctaprbs)
    {
        // Input data: Using uint4, to read 4 consecutive pairs per thread = 8 x FP16
        union u128
        {
            int4 v4;
            half vh[8];
        } vin;
        vin.v4 = prefetch_vin;
        uint8_t *prb_ptr = prefetch_prb_ptr;
        // Prefetch the inputs for the next iteration
        int nextprb = prbloop + prbid + ctaprbs;
        if (nextprb < nprb)
        {
            prefetch_vin = input_vec[nextprb * 3 + laneid];
            prefetch_prb_ptr = prb_ptrs[nextprb];
        }

        bool valid_prb = prb_ptr != nullptr;
        uint32_t valid_mask = __ballot_sync(~0, valid_prb); // Valid PRB mask for the whole warp

        if (valid_mask)
        {
            // Write the PRB pointer in shared memory (instead of shuffles later)
            if (laneid == 0 && activeThread)
                sm_prb_ptr[prbid] = prb_ptr;

            // Apply the scaling factor in FP32, and convert to integers
            int32_t vi[4], vq[4];
            for (int i = 0; i < 4; i++)
            {
                vi[i] = (int)((float)vin.vh[2 * i] * beta);
                vq[i] = (int)((float)vin.vh[2 * i + 1] * beta);
            }

            // Pack the bytes in shared memory and write to the output.  Fixed point has no compression parameters
            packOutput<3,0>(valid_mask, activeThread, prbid, laneid, tid, vi, vq, 0, compbits, sm, sm_prb_ptr);
        }
    }
}




// ****************************************************************************
// Fixed point decompression and scaling
// Similar approach to the compression code, even though there is no intra-warp reduction.
// The 8 values per thread x 3 threads is kept in order to easily unpack the data
// at various bit rates, with always a multiple of 8 bits per thread.
// Output must be 128-bit aligned (8x 16-bit values in an int4).
//
// input: Compressed input containing packed PRBs, with (3 * compbits) bytes per PRB.
//         If inplace is true, each output PRB overwrites its input, the output parameter is ignored.
// output: FP16 output array, must be aligned on 128-bit boundary
// beta: Scaling factor.
// nprb: Number of PRBs to decompress
// compbits: Number of bits to represent each value
// tid: Thread ID participating in the decompression (threads must be coalesced).
// nthreads: Number of threads participating in the compression, one or more full warps.

template <bool inplace = false>
__device__ inline void decompress_scale_fixed(uint8_t *input, __half *output, float beta,
                                              const int32_t nprb, const int32_t compbits,
                                              const int32_t tid, const int32_t nthreads)
{
    // Use 4 threads per PRB (3 active threads), 4 pairs of (I,Q) per thread
    int32_t laneid = tid & 3;
    int32_t prbid = tid / 4;

    // We don't need threads with lane id 3.
    if (laneid == 3)
        return;

    // Loop on all the PRBs, 4 threads per PRB
    for (int prbloop = prbid; prbloop < nprb; prbloop += nthreads / 4)
    {
        int32_t vi[4], vq[4];
        int32_t shift;
        int32_t prbstride = (inplace) ? 48 : 3 * compbits;
        unpackInput<0>(input, prbloop, prbstride, laneid, vi, vq, shift, compbits);

        // Expand the values back to 32-bit integers
        // shift left first then right to propagate the sign bits
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (vi[i] << (32 - compbits)) >> (32 - compbits);
            vq[i] = (vq[i] << (32 - compbits)) >> (32 - compbits);
        }

        // We'll be writing vectors of 128-bit = 8 x half values
        union u128
        {
            int4 v4;
            half vh[8];
        } vec;
        int4 *output_vec = inplace ? (int4 *)input : (int4 *)output;

        // Apply beta scaling factor in FP32, then convert to FP16
        for (int i = 0; i < 4; i++)
        {
            vec.vh[2 * i]     = (half)((float)vi[i] * beta);
            vec.vh[2 * i + 1] = (half)((float)vq[i] * beta);
        }

        output_vec[prbloop * 3 + laneid] = vec.v4;
    }
}