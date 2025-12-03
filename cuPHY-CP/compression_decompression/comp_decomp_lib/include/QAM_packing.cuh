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

#include <cstdint>
#include <cuda_fp16.h>

// Pack 4 REs (4 IQ pairs = 8 values) into 1, 2, 3 or 4 bytes, and write to global memory.
__host__ __host__ __device__ inline void packQamOutput_x4(uint8_t *output,   // Output address for the current thread
                                                          const int2 viq[4], // IQ values
                                                          int compbits)      // Compressed bits per value BPSK=2, QPSK=1, QAM16=2, QAM64=3, QAM256=4
{
    if (compbits == 4) // QAM256
    {
        uint32_t *out32 = reinterpret_cast<uint32_t *>(output);
        *out32 = ((viq[0].y & 0xf) << 24) | ((viq[0].x & 0xf) << 28) | // network order (big endian)
                 ((viq[1].y & 0xf) << 16) | ((viq[1].x & 0xf) << 20) |
                 ((viq[2].y & 0xf) << 8)  | ((viq[2].x & 0xf) << 12) |
                 ((viq[3].y & 0xf) << 0)  | ((viq[3].x & 0xf) << 4);
    }
    else if (compbits == 3) // QAM64
    {
        output[0] = ((viq[2].x & 0x6) >> 1) | ((viq[3].y & 0x7) << 2) | // 2 + 3 +
                    ((viq[3].x & 0x7) << 5);                           //  2, remains 1
        output[1] = ((viq[1].y & 0x4) >> 2) | ((viq[1].x & 0x7) << 1) | // 1 + 3 +
                    ((viq[2].y & 0x7) << 4) | ((viq[2].x & 0x1) << 7);  //  3 + 1, remains 2
        output[2] = ((viq[0].y & 0x7) << 0) | ((viq[0].x & 0x7) << 3) | // 3 + 3 +
                    ((viq[1].y & 0x3) << 6);                           // 3
    }
    else if (compbits == 2) // QAM16 or BPSK
    {
        uint16_t *out16 = reinterpret_cast<uint16_t *>(output);
        *out16 = ((viq[0].y & 0x3) << 8)  | ((viq[0].x & 0x3) << 10) | // network order (big endian)
                 ((viq[1].y & 0x3) << 12) | ((viq[1].x & 0x3) << 14) |
                 ((viq[2].y & 0x3) << 0)  | ((viq[2].x & 0x3) << 2) |
                 ((viq[3].y & 0x3) << 4)  | ((viq[3].x & 0x3) << 6);
    }
    else if (compbits == 1) // QPSK
    {
        *output = ((viq[0].y & 0x1) << 0) | ((viq[0].x & 0x1) << 1) |
                  ((viq[1].y & 0x1) << 2) | ((viq[1].x & 0x1) << 3) |
                  ((viq[2].y & 0x1) << 4) | ((viq[2].x & 0x1) << 5) |
                  ((viq[3].y & 0x1) << 6) | ((viq[3].x & 0x1) << 7);
    }
}

// Unpack 4 REs (4 IQ pairs = 8 values) from 1, 2, 3 or 4 bytes in global memory.
__host__ __host__ __device__ inline void unpackQamInput_x4(const uint8_t *input, // Input address for the current thread
                                                           int2 viq[4],          // IQ values
                                                           int compbits)         // Compressed bits per value BPSK=2, QPSK=1, QAM16=2, QAM64=3, QAM256=4
{
    // While rebuilding the viq values, no need to mask out the higher bits
    // they will disappear in the next step with the left shift.
    if (compbits == 4) // QAM256
    {
        uint32_t vin = *reinterpret_cast<const uint32_t *>(input);
        viq[0].y = (vin >> 24);
        viq[0].x = (vin >> 28);
        viq[1].y = (vin >> 16);
        viq[1].x = (vin >> 20);
        viq[2].y = (vin >> 8);
        viq[2].x = (vin >> 12);
        viq[3].y = (vin >> 0);
        viq[3].x = (vin >> 4);
    }
    else if (compbits == 3) // QAM64
    {
        uint8_t vin0 = input[0];
        uint8_t vin1 = input[1];
        uint8_t vin2 = input[2];
        viq[0].y = (vin0 >> 0);                 // 3, remains  5
        viq[0].x = (vin0 >> 3);                 // 3, remains 2
        viq[1].y = ((vin0 >> 6) | (vin1 << 2)); // 2 + 1, remains 7
        viq[1].x = (vin1 >> 1);                 // 3, remains 4
        viq[2].y = (vin1 >> 4);                 // 3, remains 1
        viq[2].x = ((vin1 >> 7) | (vin2 << 1)); // 1 + 2, remains 6
        viq[3].y = (vin2 >> 2);                 // 3, remains 3
        viq[3].x = (vin2 >> 5);                 // 3, remains 0
    }
    else if (compbits == 2) // QAM16 or BPSK
    {
        uint16_t vin = *reinterpret_cast<const uint16_t *>(input);
        viq[0].y = (vin >> 8);
        viq[0].x = (vin >> 10);
        viq[1].y = (vin >> 12);
        viq[1].x = (vin >> 14);
        viq[2].y = (vin >> 0);
        viq[2].x = (vin >> 2);
        viq[3].y = (vin >> 4);
        viq[3].x = (vin >> 6);
    }
    else if (compbits == 1) // QPSK
    {
        uint8_t vin = *reinterpret_cast<const uint8_t *>(input);
        viq[0].y = (vin >> 0);
        viq[0].x = (vin >> 1);
        viq[1].y = (vin >> 2);
        viq[1].x = (vin >> 3);
        viq[2].y = (vin >> 4);
        viq[2].x = (vin >> 5);
        viq[3].y = (vin >> 6);
        viq[3].x = (vin >> 7);
    }
    // Expand the values back to 32-bit integers
    // shift left first then right to propagate the sign bits
    for (int i = 0; i < 4; i++)
    {
        viq[i].x = (viq[i].x << (32 - compbits)) >> (32 - compbits);
        viq[i].y = (viq[i].y << (32 - compbits)) >> (32 - compbits);
    }
}
