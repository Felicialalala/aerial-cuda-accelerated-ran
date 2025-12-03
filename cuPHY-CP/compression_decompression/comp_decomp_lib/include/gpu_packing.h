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

#pragma once


// According to ORAN-WG4.CUS.0-v01.00.pdf, Table 6-2 page 95,
// the compression parameter (1 byte) is stored before the compressed data.
// The bit ordering of the compressed output is described in Annex D page 158.

// Write function will be optimized for each number of bits.
template <int compbits, int compParamLen>
__device__ inline void warpWrite(int tid,              // Global thread index
                                 uint32_t valid_mask,  // Mask of the threads with valid PRBs in the warp
                                 uint8_t *sm,          // Shared memory with packed data
                                 uint8_t **sm_prb_ptr, // PRB pointers
                                 int threadsPerPrb)    // Should be a template parameter, but can't do partial specialization
{
    constexpr int prbStride = 3 * compbits + compParamLen;
    const int warpPrbs = 32 / threadsPerPrb;
    const int warpLaneId = tid & 31;
    // First PRB of this warp
    int first_prb = (tid / 32) * warpPrbs;

    // Offset of the first PRB in shared memory
    int offset = first_prb * prbStride;

    // Which threads will write the last bytes, if prbStride is not a multiple of 32
    bool lastWriter = warpLaneId < (prbStride & 31);

    __syncwarp(0xffffffff); // Protect the shared memory

    // Write the valid PRBs of this warp, 1 byte per thread at a time, into the final destination
#pragma unroll
    for (int i = 0; i < warpPrbs; i++)
    {
        uint8_t *ptr = sm_prb_ptr[first_prb + i];
        if (valid_mask & (1 << (threadsPerPrb * i)))
        {
            int b;
            for (b = 0; b < prbStride - 32; b += 32)
                ptr[warpLaneId + b] = sm[offset + warpLaneId + b];
            if (lastWriter)
                ptr[warpLaneId + b] = sm[offset + warpLaneId + b];
        }
        offset += prbStride;
    }
    __syncwarp(0xffffffff); // Protect the shared memory
}

// Specialized function for 9 bits using a granularity of 16bits
// 28 bytes per PRB, use 14 threads x 2B to write a PRB
template <>
__device__ inline void warpWrite<9,1>(int tid,              // Global thread index
                                    uint32_t valid_mask,  // Mask of the threads with valid PRBs in the warp
                                    uint8_t *sm,          // Shared memory with packed data
                                    uint8_t **sm_prb_ptr, // PRB pointers
                                    int threadsPerPrb)    // Should be a template parameter, but can't do partial specialization
{
    // shared mem is now viewed as 16-bit.
    uint16_t *sm16 = reinterpret_cast<uint16_t *>(sm);
    constexpr int prbStride = 14; // 28 bytes, as 14 x 16-bits values
    const int warpPrbs = 32 / threadsPerPrb;
    const int halfwarpLaneId = tid & 15;
    const int halfwarpPrb = (tid & 16) / 16; // 0 for the first half of the warp, 1 for the second half
    const bool writer = (halfwarpLaneId < 14);

    // First PRB of this warp
    int first_prb = (tid / 32) * warpPrbs;

    // Offset of the first PRB in shared memory
    int offset = (first_prb + halfwarpPrb) * prbStride;

    __syncwarp(0xffffffff); // Protect the shared memory

    // Odd PRBs : shift down the mask by 1 PRB once for all
    // so we don't need to adjust the shift in the loop
    if (halfwarpPrb)
        valid_mask >>= threadsPerPrb;

        // Write the valid PRBs of this warp, 2 PRBS at a time
#pragma unroll
    for (int i = 0; i < warpPrbs; i += 2)
    {
        uint16_t *ptr = reinterpret_cast<uint16_t *>(sm_prb_ptr[first_prb + i + halfwarpPrb]);
        if (valid_mask & (1 << (threadsPerPrb * i)) && writer)
            ptr[halfwarpLaneId] = sm16[offset + halfwarpLaneId];
        offset += 2 * prbStride;
    }
    __syncwarp(0xffffffff); // Protect the shared memory
}

// Pack the compressed data (compbits per value), with 4 (I,Q) pairs per thread x 3 threads.
// Use shared memory to speed up packing for coalesced writes
// Synchronizing at the warp level. All the threads of each warp must be present.

template <int threadsPerPrb, int compParamLen = 1>
__device__ inline void packOutput(const uint32_t &valid_mask, // Valid PRB mask for the warp
                                  const bool &activeThread,   // True if the thread is actively working on a PRB
                                  const int32_t &prbid,       // PRB number for this thread
                                  const int32_t &laneid,      // Lane id inside a subgroup of 4 threads working on the same PRB
                                  const int32_t &tid,         // Thread ID in the current group of threads
                                  const int32_t vi[4],        // I values
                                  const int32_t vq[4],        // Q values
                                  const int32_t &compParam,   // Compression parameter
                                  const int32_t &compbits,    // Number of compressed bits. 16 means no compression
                                  uint8_t *sm,                // Shared memory used to reorder per-warp data
                                  uint8_t **sm_prb_ptr)       // Shared memory of the PRB pointers
{
    int32_t prbStride = 3 * compbits + compParamLen;

    // Offset in shared memory for this thread
    int offset = prbid * prbStride + laneid * compbits;

    // Compression parameter
    if (laneid == 0 && activeThread && (compParamLen > 0))
        sm[offset] = compParam;

    // Write the data in shared memory (except threads with laneid=3)
    // then write to global memory using a specialized warpWrite call
    offset += compParamLen;
    if (compbits == 7)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] << 1) | ((vq[0] >> 6) & 0x01); // 7 + 1, remains 6
            sm[offset + 1] = (vq[0] << 2) | ((vi[1] >> 5) & 0x03); // 6 + 2, remains 5
            sm[offset + 2] = (vi[1] << 3) | ((vq[1] >> 4) & 0x07); // 5 + 3, remains 4
            sm[offset + 3] = (vq[1] << 4) | ((vi[2] >> 3) & 0x0f); // 4 + 4, remains 3
            sm[offset + 4] = (vi[2] << 5) | ((vq[2] >> 2) & 0x1f); // 3 + 5, remains 2
            sm[offset + 5] = (vq[2] << 6) | ((vi[3] >> 1) & 0x3f); // 2 + 6, remains 1
            sm[offset + 6] = (vi[3] << 7) | (vq[3] & 0x7f);        // 1 + 7
        }
        warpWrite<7,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 8)
    {
        if (activeThread)
        {
            for (int i = 0; i < 4; i++)
            {
                sm[offset + 2 * i]     = vi[i];
                sm[offset + 2 * i + 1] = vq[i];
            }
        }
        warpWrite<8,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 10)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 2);                         // 8, remains 2
            sm[offset + 1] = (vi[0] << 6) | ((vq[0] >> 4) & 0x3f); // 2 + 6, remains 4
            sm[offset + 2] = (vq[0] << 4) | ((vi[1] >> 6) & 0x0f); // 4 + 4, remains 6
            sm[offset + 3] = (vi[1] << 2) | ((vq[1] >> 8) & 0x03); // 6 + 2, remains 8
            sm[offset + 4] = vq[1];                                // 8
            sm[offset + 5] = (vi[2] >> 2);                         // 8, remains 2
            sm[offset + 6] = (vi[2] << 6) | ((vq[2] >> 4) & 0x3f); // 2 + 6, remains 4
            sm[offset + 7] = (vq[2] << 4) | ((vi[3] >> 6) & 0x0f); // 4 + 4, remains 6
            sm[offset + 8] = (vi[3] << 2) | ((vq[3] >> 8) & 0x03); // 6 + 2, remains 8
            sm[offset + 9] = vq[3];                                // 8
        }
        warpWrite<10,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 11)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 3);                          // 8, remains 3
            sm[offset + 1] = (vi[0] << 5) | ((vq[0] >> 6) & 0x1f);  // 3 + 5, remains 6
            sm[offset + 2] = (vq[0] << 2) | ((vi[1] >> 9) & 0x03);  // 6 + 2, remains 9
            sm[offset + 3] = (vi[1] >> 1);                          // 8, remains 1
            sm[offset + 4] = (vi[1] << 7) | ((vq[1] >> 4) & 0x7f);  // 1 + 7, remains 4
            sm[offset + 5] = (vq[1] << 4) | ((vi[2] >> 7) & 0x0f);  // 4 + 4, remains 7
            sm[offset + 6] = (vi[2] << 1) | ((vq[2] >> 10) & 0x01); // 7 + 1, remains 10
            sm[offset + 7] = (vq[2] >> 2);                          // 8, remains 2
            sm[offset + 8] = (vq[2] << 6) | ((vi[3] >> 5) & 0x3f);  // 2 + 6, remains 5
            sm[offset + 9] = (vi[3] << 3) | ((vq[3] >> 8) & 0x07);  // 5 + 3, remains 8
            sm[offset + 10] = vq[3];                                // 8
        }
        warpWrite<11,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 12)
    {
        if (activeThread)
        {
            for (int i = 0; i < 4; i++)
            {
                sm[offset + 3 * i]     = vi[i] >> 4;
                sm[offset + 3 * i + 1] = (vi[i] << 4) | ((vq[i] >> 8) & 0x0f);
                sm[offset + 3 * i + 2] = vq[i];
            }
        }
        warpWrite<12,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 13)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 5);                           // 8, remains 5
            sm[offset + 1] = (vi[0] << 3) | ((vq[0] >> 10) & 0x07);  // 5 + 3, remains 10
            sm[offset + 2] = (vq[0] >> 2);                           // 8, remains 2
            sm[offset + 3] = (vq[0] << 6) | ((vi[1] >> 7) & 0x3f);   // 2 + 6, remains 7
            sm[offset + 4] = (vi[1] << 1) | ((vq[1] >> 12) & 0x01);  // 7 + 1, remains 12
            sm[offset + 5] = (vq[1] >> 4);                           // 8, Remains 4
            sm[offset + 6] = (vq[1] << 4) | ((vi[2] >> 9) & 0x0f);   // 4 + 4, remains 9
            sm[offset + 7] = (vi[2] >> 1);                           // 8, remains 1
            sm[offset + 8] = (vi[2] << 7) | ((vq[2] >> 6) & 0x7f);   // 1 + 7, remains 6
            sm[offset + 9] = (vq[2] << 2) | ((vi[3] >> 11) & 0x03); // 6 + 2, remains 11
            sm[offset + 10] = (vi[3] >> 3);                          // 8, remains 3
            sm[offset + 11] = (vi[3] << 5) | ((vq[3] >> 8) & 0x1f);  // 3 + 5, remains 8
            sm[offset + 12] = vq[3];
        }
        warpWrite<13,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 15)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 7);                           // 8, remains 7
            sm[offset + 1] = (vi[0] << 1) | ((vq[0] >> 14) & 0x01);  // 7 + 1, remains 14
            sm[offset + 2] = (vq[0] >> 6);                           // 8, remains 6
            sm[offset + 3] = (vq[0] << 2) | ((vi[1] >> 13) & 0x03);  // 6 + 2, remains 13
            sm[offset + 4] = (vi[1] >> 5);                           // 8, remains 5
            sm[offset + 5] = (vi[1] << 3) | ((vq[1] >> 12) & 0x07);  // 5 + 3, Remains 12
            sm[offset + 6] = (vq[1] >> 4);                           // 8, remains 4
            sm[offset + 7] = (vq[1] << 4) | ((vi[2] >> 11) & 0x0f);  // 4 + 4, remains 11
            sm[offset + 8] = (vi[2] >> 3);                           // 8, remains 3
            sm[offset + 9] = (vi[2] << 5) | ((vq[2] >> 10) & 0x1f);  // 3 + 5, remains 10
            sm[offset + 10] = (vq[2] >> 2);                          // 8, remains 2
            sm[offset + 11] = (vq[2] << 6) | ((vi[3] >> 9) & 0x3f);  // 2 + 6, Remains 9
            sm[offset + 12] = (vi[3] >> 1);                          // 8, remains 1
            sm[offset + 13] = (vi[3] << 7) | ((vq[3] >> 8) & 0x7f);  // 1 + 7, remains 8
            sm[offset + 14] = vq[3];
        }
        warpWrite<15,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 16)
    {
        int32_t* word_offset = reinterpret_cast<int32_t*>(&sm[offset]);
        for (int i = 0; i < 4; i++)
        {
            word_offset[i]   = __byte_perm(vi[i],vq[i],0x4501);
        }
        warpWrite<16,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    // Most used values at the end (compiler will likely put them in reverse order)
    else if (compbits == 14)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 6);                           // 8, remains 6
            sm[offset + 1] = (vi[0] << 2) | ((vq[0] >> 12) & 0x03);  // 6 + 2, remains 12
            sm[offset + 2] = (vq[0] >> 4);                           // 8, remains 4
            sm[offset + 3] = (vq[0] << 4) | ((vi[1] >> 10) & 0x0f);  // 4 + 4, remains 10
            sm[offset + 4] = (vi[1] >> 2);                           // 8, remains 2
            sm[offset + 5] = (vi[1] << 6) | ((vq[1] >> 8) & 0x3f);   // 2 + 6, Remains 8
            sm[offset + 6] = vq[1];                                  // 8, remains 0
            sm[offset + 7] = (vi[2] >> 6);                           // 8, remains 6
            sm[offset + 8] = (vi[2] << 2) | ((vq[2] >> 12) & 0x03);  // 6 + 2, remains 12
            sm[offset + 9] = (vq[2] >> 4);                           // 8, remains 4
            sm[offset + 10] = (vq[2] << 4) | ((vi[3] >> 10) & 0x0f); // 4 + 4, remains 10
            sm[offset + 11] = (vi[3] >> 2);                          // 8, remains 2
            sm[offset + 12] = (vi[3] << 6) | ((vq[3] >> 8) & 0x3f);  // 2 + 6, remains 8
            sm[offset + 13] = vq[3];
        }
        warpWrite<14,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
    else if (compbits == 9)
    {
        if (activeThread)
        {
            sm[offset]     = (vi[0] >> 1);                         // 8, remains 1
            sm[offset + 1] = (vi[0] << 7) | ((vq[0] >> 2) & 0x7f); // 1 + 7, remains 2
            sm[offset + 2] = (vq[0] << 6) | ((vi[1] >> 3) & 0x3f); // 2 + 6, remains 3
            sm[offset + 3] = (vi[1] << 5) | ((vq[1] >> 4) & 0x1f); // 3 + 5, remains 4
            sm[offset + 4] = (vq[1] << 4) | ((vi[2] >> 5) & 0x0f); // 4 + 4, remains 5
            sm[offset + 5] = (vi[2] << 3) | ((vq[2] >> 6) & 0x07); // 5 + 3, remains 6
            sm[offset + 6] = (vq[2] << 2) | ((vi[3] >> 7) & 0x03); // 6 + 2, remains 7
            sm[offset + 7] = (vi[3] << 1) | ((vq[3] >> 8) & 0x01); // 7 + 1, remains 8
            sm[offset + 8] = vq[3];                                // 8
        }
        warpWrite<9,compParamLen>(tid, valid_mask, sm, sm_prb_ptr, threadsPerPrb);
    }
}



// Special pass-through write for uncompressed data.
// Due to 16b alignment, each 48B PRB can be written 2B at a time, using 24 threads
template <int threadsPerPrb>
__device__ inline void writeUncompressed(const uint32_t &valid_mask, // Valid PRB mask for the warp
                                         const bool &activeThread,   // True if the thread is actively working on a PRB
                                         const int32_t &prbid,       // PRB number for this thread
                                         const int32_t &tid,         // Thread ID in the current group of threads
                                         const int32_t &laneid,      // Lane id inside a subgroup of 4 threads working on the same PRB
                                         int4 vin,                   // Input value, 16B per thread
                                         uint8_t *sm,                // Shared memory used to reorder per-warp data
                                         uint8_t **sm_prb_ptr)       // Shared memory of the PRB pointers
{
    constexpr int warpPrbs = 32 / threadsPerPrb;

    // Write the bytes in shared memory using 16B.
    int4 *smvec = reinterpret_cast<int4 *>(sm + prbid * 48);
    int warpLaneId = tid & 31;
    if (activeThread)
        smvec[laneid] = vin;

    __syncwarp(0xffffffff); // Protect the shared memory

    // Write the PRBs of this warp
    int first_prb = (tid / 32) * warpPrbs;
    for (int i = 0; i < warpPrbs; i++)
    {
        bool valid = valid_mask & (1 << (threadsPerPrb * i));
        // Use 24 threads, and write 2 bytes per thread (FP16)
        half *smh = reinterpret_cast<half *>(sm) + (first_prb + i) * 24;
        half *ptr = reinterpret_cast<half *>(sm_prb_ptr[first_prb + i]);
        if (valid && warpLaneId < 24)
            ptr[warpLaneId] = smh[warpLaneId];
    }
    __syncwarp(0xffffffff); // Protect the shared memory
}



// According to ORAN-WG4.CUS.0-v01.00.pdf, Table 6-2 page 95,
// the compression parameter (1 byte) is stored before the compressed data.
// The bit ordering of the compressed output is described in Annex D page 158.

// Pack the compressed data (compbits per value), with 4 (I,Q) pairs per thread x 3 threads.
// Use shared memory to speed up packing for coalesced writes
// Synchronizing at the warp level. All the threads of each warp must be present.

template <uint8_t compParamLen = 1>
__device__ inline void packOutput(uint8_t *output,         // Global memory address where the first PRB will be written
                                  uint8_t **prb_ptrs,      // Pointer to list of per-PRB packet locations, or nullptr if GPU comms disabled 
                                  const int32_t prb_offset, // Absolute offset into PRB table
                                  const int32_t prbid,     // PRB number for this thread
                                  const int32_t nprb,      // Total number of PRBs (not to exceed)
                                  const int32_t prbStride, // Stride in bytes between 2 PRBs in output array
                                  const int32_t tid,       // Thread ID in the currenr group of threads
                                  const int32_t laneid,    // Lane id inside a subgroup of 4 threads working on the same PRB
                                  const int32_t vi[4],     // I values
                                  const int32_t vq[4],     // Q values
                                  const int32_t compParam, // Compression parameter
                                  const int32_t compbits,  // Number of compressed bits. 16 means no compression
                                  uint8_t *sm)             // Shared memory used to reorder per-warp data
{
    // Offset in shared memory for this thread
    int offset = prbid * prbStride + laneid * compbits;

    // Compression parameter. Don't write if no compression is used.
    if (laneid == 3 && compParamLen > 0)
        sm[offset] = compParam;

    // Write the data in shared memory. Threads with laneid3 are not active here.
    if (laneid < 3)
    {
        offset += compParamLen;
        if (compbits == 7)
        {
            sm[offset]     = (vi[0] << 1) | ((vq[0] >> 6) & 0x01); // 7 + 1, remains 6
            sm[offset + 1] = (vq[0] << 2) | ((vi[1] >> 5) & 0x03); // 6 + 2, remains 5
            sm[offset + 2] = (vi[1] << 3) | ((vq[1] >> 4) & 0x07); // 5 + 3, remains 4
            sm[offset + 3] = (vq[1] << 4) | ((vi[2] >> 3) & 0x0f); // 4 + 4, remains 3
            sm[offset + 4] = (vi[2] << 5) | ((vq[2] >> 2) & 0x1f); // 3 + 5, remains 2
            sm[offset + 5] = (vq[2] << 6) | ((vi[3] >> 1) & 0x3f); // 2 + 6, remains 1
            sm[offset + 6] = (vi[3] << 7) | (vq[3] & 0x7f);        // 1 + 7
        }
        else if (compbits == 8)
        {
            for (int i = 0; i < 4; i++)
            {
                sm[offset + 2 * i]     = vi[i];
                sm[offset + 2 * i + 1] = vq[i];
            }
        }
        else if (compbits == 9)
        {
            sm[offset]     = (vi[0] >> 1);                         // 8, remains 1
            sm[offset + 1] = (vi[0] << 7) | ((vq[0] >> 2) & 0x7f); // 1 + 7, remains 2
            sm[offset + 2] = (vq[0] << 6) | ((vi[1] >> 3) & 0x3f); // 2 + 6, remains 3
            sm[offset + 3] = (vi[1] << 5) | ((vq[1] >> 4) & 0x1f); // 3 + 5, remains 4
            sm[offset + 4] = (vq[1] << 4) | ((vi[2] >> 5) & 0x0f); // 4 + 4, remains 5
            sm[offset + 5] = (vi[2] << 3) | ((vq[2] >> 6) & 0x07); // 5 + 3, remains 6
            sm[offset + 6] = (vq[2] << 2) | ((vi[3] >> 7) & 0x03); // 6 + 2, remains 7
            sm[offset + 7] = (vi[3] << 1) | ((vq[3] >> 8) & 0x01); // 7 + 1, remains 8
            sm[offset + 8] = vq[3];                                // 8
        }
        else if (compbits == 10)
        {
            sm[offset]     = (vi[0] >> 2);                         // 8, remains 2
            sm[offset + 1] = (vi[0] << 6) | ((vq[0] >> 4) & 0x3f); // 2 + 6, remains 4
            sm[offset + 2] = (vq[0] << 4) | ((vi[1] >> 6) & 0x0f); // 4 + 4, remains 6
            sm[offset + 3] = (vi[1] << 2) | ((vq[1] >> 8) & 0x03); // 6 + 2, remains 8
            sm[offset + 4] = vq[1];                                // 8
            sm[offset + 5] = (vi[2] >> 2);                         // 8, remains 2
            sm[offset + 6] = (vi[2] << 6) | ((vq[2] >> 4) & 0x3f); // 2 + 6, remains 4
            sm[offset + 7] = (vq[2] << 4) | ((vi[3] >> 6) & 0x0f); // 4 + 4, remains 6
            sm[offset + 8] = (vi[3] << 2) | ((vq[3] >> 8) & 0x03); // 6 + 2, remains 8
            sm[offset + 9] = vq[3];                               // 8
        }
        else if (compbits == 11)
        {
            sm[offset]     = (vi[0] >> 3);                          // 8, remains 3
            sm[offset + 1] = (vi[0] << 5) | ((vq[0] >> 6) & 0x1f);  // 3 + 5, remains 6
            sm[offset + 2] = (vq[0] << 2) | ((vi[1] >> 9) & 0x03);  // 6 + 2, remains 9
            sm[offset + 3] = (vi[1] >> 1);                          // 8, remains 1
            sm[offset + 4] = (vi[1] << 7) | ((vq[1] >> 4) & 0x7f);  // 1 + 7, remains 4
            sm[offset + 5] = (vq[1] << 4) | ((vi[2] >> 7) & 0x0f);  // 4 + 4, remains 7
            sm[offset + 6] = (vi[2] << 1) | ((vq[2] >> 10) & 0x01); // 7 + 1, remains 10
            sm[offset + 7] = (vq[2] >> 2);                          // 8, remains 2
            sm[offset + 8] = (vq[2] << 6) | ((vi[3] >> 5) & 0x3f);  // 2 + 6, remains 5
            sm[offset + 9] = (vi[3] << 3) | ((vq[3] >> 8) & 0x07); // 5 + 3, remains 8
            sm[offset + 10] = vq[3];                                // 8
        }
        else if (compbits == 12)
        {
            for (int i = 0; i < 4; i++)
            {
                sm[offset + 3 * i]     = vi[i] >> 4;
                sm[offset + 3 * i + 1] = (vi[i] << 4) | ((vq[i] >> 8) & 0x0f);
                sm[offset + 3 * i + 2] = vq[i];
            }
        }
        else if (compbits == 13)
        {
            sm[offset]     = (vi[0] >> 5);                           // 8, remains 5
            sm[offset + 1] = (vi[0] << 3) | ((vq[0] >> 10) & 0x07);  // 5 + 3, remains 10
            sm[offset + 2] = (vq[0] >> 2);                           // 8, remains 2
            sm[offset + 3] = (vq[0] << 6) | ((vi[1] >> 7) & 0x3f);   // 2 + 6, remains 7
            sm[offset + 4] = (vi[1] << 1) | ((vq[1] >> 12) & 0x01);  // 7 + 1, remains 12
            sm[offset + 5] = (vq[1] >> 4);                           // 8, Remains 4
            sm[offset + 6] = (vq[1] << 4) | ((vi[2] >> 9) & 0x0f);   // 4 + 4, remains 9
            sm[offset + 7] = (vi[2] >> 1);                           // 8, remains 1
            sm[offset + 8] = (vi[2] << 7) | ((vq[2] >> 6) & 0x7f);   // 1 + 7, remains 6
            sm[offset + 9] = (vq[2] << 2) | ((vi[3] >> 11) & 0x03); // 6 + 2, remains 11
            sm[offset + 10] = (vi[3] >> 3);                          // 8, remains 3
            sm[offset + 11] = (vi[3] << 5) | ((vq[3] >> 8) & 0x1f);  // 3 + 5, remains 8
            sm[offset + 12] = vq[3];
        }
        else if (compbits == 14)
        {
            sm[offset]     = (vi[0] >> 6);                           // 8, remains 6
            sm[offset + 1] = (vi[0] << 2) | ((vq[0] >> 12) & 0x03);  // 6 + 2, remains 12
            sm[offset + 2] = (vq[0] >> 4);                           // 8, remains 4
            sm[offset + 3] = (vq[0] << 4) | ((vi[1] >> 10) & 0x0f);  // 4 + 4, remains 10
            sm[offset + 4] = (vi[1] >> 2);                           // 8, remains 2
            sm[offset + 5] = (vi[1] << 6) | ((vq[1] >> 8) & 0x3f);   // 2 + 6, Remains 8
            sm[offset + 6] = vq[1];                                  // 8, remains 0
            sm[offset + 7] = (vi[2] >> 6);                           // 8, remains 6
            sm[offset + 8] = (vi[2] << 2) | ((vq[2] >> 12) & 0x03);  // 6 + 2, remains 12
            sm[offset + 9] = (vq[2] >> 4);                          // 8, remains 4
            sm[offset + 10] = (vq[2] << 4) | ((vi[3] >> 10) & 0x0f); // 4 + 4, remains 10
            sm[offset + 11] = (vi[3] >> 2);                          // 8, remains 2
            sm[offset + 12] = (vi[3] << 6) | ((vq[3] >> 8) & 0x3f);  // 2 + 6, remains 8
            sm[offset + 13] = vq[3];
        }
        else if (compbits == 15)
        {
            sm[offset]     = (vi[0] >> 7);                           // 8, remains 7
            sm[offset + 1] = (vi[0] << 1) | ((vq[0] >> 14) & 0x01);  // 7 + 1, remains 14
            sm[offset + 2] = (vq[0] >> 6);                           // 8, remains 6
            sm[offset + 3] = (vq[0] << 2) | ((vi[1] >> 13) & 0x03);  // 6 + 2, remains 13
            sm[offset + 4] = (vi[1] >> 5);                           // 8, remains 5
            sm[offset + 5] = (vi[1] << 3) | ((vq[1] >> 12) & 0x07);  // 5 + 3, Remains 12
            sm[offset + 6] = (vq[1] >> 4);                           // 8, remains 4
            sm[offset + 7] = (vq[1] << 4) | ((vi[2] >> 11) & 0x0f);  // 4 + 4, remains 11
            sm[offset + 8] = (vi[2] >> 3);                           // 8, remains 3
            sm[offset + 9] = (vi[2] << 5) | ((vq[2] >> 10) & 0x1f); // 3 + 5, remains 10
            sm[offset + 10] = (vq[2] >> 2);                          // 8, remains 2
            sm[offset + 11] = (vq[2] << 6) | ((vi[3] >> 9) & 0x3f);  // 2 + 6, Remains 9
            sm[offset + 12] = (vi[3] >> 1);                          // 8, remains 1
            sm[offset + 13] = (vi[3] << 7) | ((vq[3] >> 8) & 0x7f);  // 1 + 7, remains 8
            sm[offset + 14] = vq[3];
        }
        else if (compbits == 16)
        {
            int32_t* word_offset = reinterpret_cast<int32_t*>(&sm[offset]);
            for (int i = 0; i < 4; i++)
            {
                word_offset[i]   = __byte_perm(vi[i],vq[i],0x4501);
            }
        }
    }
    // Write the shared memory to global, at the warp level to avoid calling syncthreads()
    // which would break the warp granularity of the compression device functions.

    // Get the offset in shared memory for the first thread of this warp
    offset = __shfl_sync(0xffffffff, offset, 0);
    int warpLaneId = tid & 31;

    // Get the offset in global memory for the first PRB of this warp
    int globalOffset = prbid * prbStride;
    globalOffset = __shfl_sync(0xffffffff, globalOffset, 0);

    // Max offset not to exceed in global memory
    int maxOffset = nprb * prbStride;

    __syncwarp(0xffffffff);

    // Write 1 byte per thread, until the 8 PRBs of this warp have been written. The GPU comms case is
    // a little trickier since the shared memory layout is no longer 4 threads per PRB, but rather
    // all threads participate in a single PRB. We need to figure out which PRB we're in to write
    // to the correct spot.
    if (prb_ptrs) { // GPU init comms enabled
        for (int i = warpLaneId; i < 8 * prbStride; i += 32)
        {
            uint8_t* prb_ptr = prb_ptrs[prb_offset + (globalOffset + i)/prbStride];
            if (prb_ptr) {            
                if (globalOffset + i < maxOffset) {
                    prb_ptr[i%prbStride] = sm[offset + i];
                }
            }
        }
    }
    else {
        for (int i = warpLaneId; i < 8 * prbStride; i += 32)
        {          
            if (globalOffset + i < maxOffset) {
                output[globalOffset + i] = sm[offset + i];
            }
        }        
    }

    __syncwarp(0xffffffff);
}

// Unpack the compressed data (compbits per value), with 4 (I,Q) pairs per thread x 3 threads.
// Warning: Only the lower 16-bit of vi and vq are populated, negative numbers are not properly signed!
template <uint8_t compParamLen = 1>
__device__ inline void unpackInput(uint8_t *input,
                                   int32_t prbid,
                                   int32_t prbStride,
                                   uint32_t laneid,
                                   int32_t vi[4],
                                   int32_t vq[4],
                                   int32_t &compParam,
                                   int32_t compbits)
{
    int offset = prbid * prbStride;
    if(compParamLen == 1)
    {
        compParam = input[offset];
    } else {
        compParam = 0;
    }
    offset += laneid * compbits + compParamLen;
    if (compbits == 7)
    {
        vi[0] = input[offset] >> 1;                                       // 7, remains 1
        vq[0] = ((input[offset]     & 0x01) << 6) | (input[offset + 1] >> 2); // 1 + 6, remains 2
        vi[1] = ((input[offset + 1] & 0x03) << 5) | (input[offset + 2] >> 3); // 2 + 5, remains 3
        vq[1] = ((input[offset + 2] & 0x07) << 4) | (input[offset + 3] >> 4); // 3 + 4, remains 4
        vi[2] = ((input[offset + 3] & 0x0f) << 3) | (input[offset + 4] >> 5); // 4 + 3, remains 5
        vq[2] = ((input[offset + 4] & 0x1f) << 2) | (input[offset + 5] >> 6); // 5 + 2, remains 6
        vi[3] = ((input[offset + 5] & 0x3f) << 1) | (input[offset + 6] >> 7); // 6 + 1, remains 7
        vq[3] = input[offset + 6] & 0x7f;                                     // 7
    }
    else if (compbits == 8)
    {
        for (int i = 0; i < 4; i++)
        {
            vi[i] = input[offset + 2 * i];
            vq[i] = input[offset + 2 * i + 1];
        }
    }
    else if (compbits == 9)
    {
        vi[0] = (input[offset] << 1) | (input[offset + 1] >> 7);              // 8 + 1, remains 7
        vq[0] = ((input[offset + 1] & 0x7f) << 2) | (input[offset + 2] >> 6); // 7 + 2, remains 6
        vi[1] = ((input[offset + 2] & 0x3f) << 3) | (input[offset + 3] >> 5); // 6 + 3, remains 5
        vq[1] = ((input[offset + 3] & 0x1f) << 4) | (input[offset + 4] >> 4); // 5 + 4, remains 4
        vi[2] = ((input[offset + 4] & 0x0f) << 5) | (input[offset + 5] >> 3); // 4 + 5, remains 3
        vq[2] = ((input[offset + 5] & 0x07) << 6) | (input[offset + 6] >> 2); // 3 + 6, remains 2
        vi[3] = ((input[offset + 6] & 0x03) << 7) | (input[offset + 7] >> 1); // 2 + 7, remains 1
        vq[3] = ((input[offset + 7] & 0x01) << 8) | input[offset + 8];        // 1 + 8
    }
    else if (compbits == 10)
    {
        vi[0] = (input[offset] << 2) | (input[offset + 1] >> 6);              // 8 + 2, remains 6
        vq[0] = ((input[offset + 1] & 0x3f) << 4) | (input[offset + 2] >> 4); // 6 + 4, remains 4
        vi[1] = ((input[offset + 2] & 0x0f) << 6) | (input[offset + 3] >> 2); // 4 + 6, remains 2
        vq[1] = ((input[offset + 3] & 0x03) << 8) | (input[offset + 4]);      // 2 + 8
        vi[2] = (input[offset + 5] << 2) | (input[offset + 6] >> 6);          // 8 + 2, remains 6
        vq[2] = ((input[offset + 6] & 0x3f) << 4) | (input[offset + 7] >> 4); // 6 + 4, remains 4
        vi[3] = ((input[offset + 7] & 0x0f) << 6) | (input[offset + 8] >> 2); // 4 + 6, remains 2
        vq[3] = ((input[offset + 8] & 0x03) << 8) | (input[offset + 9]);      // 2 + 8
    }
    else if (compbits == 11)
    {
        vi[0] = (input[offset] << 3) | (input[offset + 1] >> 5);                                          // 8 + 3, remains 5
        vq[0] = ((input[offset + 1] & 0x1f) << 6) | (input[offset + 2] >> 2);                             // 5 + 6, remains 2
        vi[1] = ((input[offset + 2] & 0x03) << 9) | (input[offset + 3] << 1) | (input[offset + 4] >> 7);  // 2 + 8 + 1, remains 7
        vq[1] = ((input[offset + 4] & 0x7f) << 4) | (input[offset + 5] >> 4);                             // 7 + 4, remains 4
        vi[2] = ((input[offset + 5] & 0x0f) << 7) | (input[offset + 6] >> 1);                             // 4 + 7, remains 1
        vq[2] = ((input[offset + 6] & 0x01) << 10) | (input[offset + 7] << 2) | (input[offset + 8] >> 6); // 1 + 8 + 2, remains 6
        vi[3] = ((input[offset + 8] & 0x3f) << 5) | (input[offset + 9] >> 3);                             // 6 + 5, remains 3
        vq[3] = ((input[offset + 9] & 0x07) << 8) | (input[offset + 10]);                                 // 3 + 8
    }
    else if (compbits == 12)
    {
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (input[offset + 3 * i] << 4) | (input[offset + 3 * i + 1] >> 4);
            vq[i] = ((input[offset + 3 * i + 1] & 0xf) << 8) | input[offset + 3 * i + 2];
        }
    }
    else if (compbits == 13)
    {
        vi[0] = (input[offset] << 5) | (input[offset + 1] >> 3);                                            // 8 + 5, remains 3
        vq[0] = ((input[offset + 1] & 0x07) << 10) | (input[offset + 2] << 2) | (input[offset + 3] >> 6);   // 3 + 8 + 2, remains 6
        vi[1] = ((input[offset + 3] & 0x3f) << 7) | (input[offset + 4] >> 1);                               // 6 + 7, remains 1
        vq[1] = ((input[offset + 4] & 0x01) << 12) | (input[offset + 5] << 4) | (input[offset + 6] >> 4);   // 1 + 8 + 4, remains 4
        vi[2] = ((input[offset + 6] & 0x0f) << 9) | (input[offset + 7] << 1) | (input[offset + 8] >> 7);    // 4 + 8 + 1, remains 7
        vq[2] = ((input[offset + 8] & 0x7f) << 6) | (input[offset + 9] >> 2);                               // 7 + 6, remains 2
        vi[3] = ((input[offset + 9] & 0x03) << 11) | (input[offset + 10] << 3) | (input[offset + 11] >> 5); // 2 + 8 + 3, remains 5
        vq[3] = ((input[offset + 11] & 0x1f) << 8) | input[offset + 12];                                    // 5 + 8
    }
    else if (compbits == 14)
    {
        vi[0] = (input[offset] << 6) | (input[offset + 1] >> 2);                                             // 8 + 6, remains 2
        vq[0] = ((input[offset + 1] & 0x03) << 12) | (input[offset + 2] << 4) | (input[offset + 3] >> 4);    // 2 + 8 + 4, remains 4
        vi[1] = ((input[offset + 3] & 0x0f) << 10) | (input[offset + 4] << 2) | (input[offset + 5] >> 6);    // 4 + 8 + 2, remains 6
        vq[1] = ((input[offset + 5] & 0x3f) << 8) | input[offset + 6];                                       // 6 + 8
        vi[2] = (input[offset + 7] << 6) | (input[offset + 8] >> 2);                                         // 8 + 6, remains 2
        vq[2] = ((input[offset + 8] & 0x03) << 12) | (input[offset + 9] << 4) | (input[offset + 10] >> 4);   // 2 + 8 + 4, remains 4
        vi[3] = ((input[offset + 10] & 0x0f) << 10) | (input[offset + 11] << 2) | (input[offset + 12] >> 6); // 4 + 8 + 2, remains 6
        vq[3] = ((input[offset + 12] & 0x3f) << 8) | input[offset + 13];                                     // 6 + 8
    }
    else if (compbits == 15)
    {
        vi[0] = (input[offset] << 7) | (input[offset + 1] >> 1);                                            // 8 + 7, remains 1
        vq[0] = ((input[offset + 1] & 0x01) << 14) | (input[offset + 2] << 6) | (input[offset + 3] >> 2);   // 1 + 8 + 6, remains 2
        vi[1] = ((input[offset + 3] & 0x03) << 13) | (input[offset + 4] << 5) | (input[offset + 5] >> 3);   // 2 + 8 + 5, remains 3
        vq[1] = ((input[offset + 5] & 0x07) << 12) | (input[offset + 6] << 4) | (input[offset + 7] >> 4);   // 3 + 8 + 4, remains 4
        vi[2] = ((input[offset + 7] & 0x0f) << 11) | (input[offset + 8] << 3) | (input[offset + 9] >> 5);   // 4 + 8 + 3, remains 5
        vq[2] = ((input[offset + 9] & 0x1f) << 10) | (input[offset + 10] << 2) | (input[offset + 11] >> 6); // 5 + 8 + 2, remains 6
        vi[3] = ((input[offset + 11] & 0x3f) << 9) | (input[offset + 12] << 1) | (input[offset + 13] >> 7); // 6 + 8 + 1, remains 7
        vq[3] = ((input[offset + 13] & 0x7f) << 8) | input[offset + 14];                                    // 7 + 8
    }
    else if (compbits == 16)
    {
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (input[offset + 4 * i    ] << 8) + (input[offset + 4 * i + 1]);
            vq[i] = (input[offset + 4 * i + 2] << 8) + (input[offset + 4 * i + 3]);
        }
    }
}