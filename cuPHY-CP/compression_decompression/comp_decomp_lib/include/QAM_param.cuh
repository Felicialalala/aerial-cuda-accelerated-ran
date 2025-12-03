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

// Qam parameters shared by all the PRBs in the same list
struct QamListParam
{
    // Enum that fits on 4 bits, and for which the value modulo 8 gives the number of bits per I or Q element
    // It also allows to compute -1/2^(N) and get the proper shift (except for BPSK which is never shifted)
    typedef enum
    {
        MODCOMP_UNUSED = 0,
        MODCOMP_QPSK = 1,
        MODCOMP_QAM16 = 2,
        MODCOMP_QAM64 = 3,
        MODCOMP_QAM256 = 4,
        MODCOMP_BPSK = 10
    } qamwidth;

    // bits [0:0] = CSF0
    // bits [1:1] = CSF1
    // bits [2:5] = qamwidth
    uint8_t bits;

    __host__ __device__ QamListParam() : bits(0) {}

    // Set
    __host__ __device__ inline void set(const qamwidth width,
                                        const uint8_t &csf0,
                                        const uint8_t &csf1)
    {
        bits = csf0 | (csf1 << 1) | (width << 2);
    }

    // Return the QAM width
    __host__ __device__ inline qamwidth get_qam_width() const
    {
        return static_cast<qamwidth>(bits >> 2);
    }

    // Return the number of bits per I or Q value
    __host__ __device__ inline uint32_t get_bits_per_value() const
    {
        return (get_qam_width() & 7);
    }

    // Return the number of bits per I or Q value for a given width
    __host__ __device__ static inline uint32_t get_bits_per_value(qamwidth width)
    {
        return (width & 7);
    }

    // Return the number of bits per RE = IQ pair
    __host__ __device__ inline uint32_t get_bits_per_element() const
    {
        return (2 * get_bits_per_value());
    }

    // Return the floating point value to use for the shift (Do not use the value for BPSK)
    __host__ __device__ inline float get_shift() const
    {
        // BPSK = invalid, QPSK = 1/2, QAM16 = 1/4, QAM64 = 1/8. QAM256 = 1/16
        union
        {
            uint32_t i;
            float f;
        } v;
        // Build -1/2^(udIqWidth)
        v.i = (127 - get_qam_width()) << 23;
        return v.f;
    }

    // Return the factor to apply to converting from float to a signed int
    __host__ __device__ inline float get_f2i_fact() const
    {
        // BPSK = 2.0, QPSK = 1.0, QAM16 = 2.0, QAM64 = 4.0, QAM256 = 8.0
        union
        {
            uint32_t i;
            float f;
        } v;
        // Build 2^(udIqWidth)
        v.i = (126 + get_bits_per_value()) << 23;
        return v.f;
    }

    // Return the factor to apply to converting from a signed int to a float
    __host__ __device__ inline float get_i2f_fact() const
    {
        // BPSK = 0.5, QPSK = 1.0, QAM16 = 0.5, QAM64 = 0.25, QAM256 = 0.125
        union
        {
            uint32_t i;
            float f;
        } v;
        // Build 2^(udIqWidth)
        v.i = (128 - get_bits_per_value()) << 23;
        return v.f;
    }

    // Returns if the values covered by mask[num] need to be shifted
    template <int num>
    __host__ __device__ inline bool need_shift() const
    {
        return ((bits >> (num ? 1 : 0)) & 1);
    }

    // Based on 2 input RE masks, return the combined mask of which REs need to be shifted
    __host__ __device__ inline uint32_t get_shift_mask(const uint32_t &remask0, const uint32_t &remask1) const
    {
        return ((remask0 * (bits & 1)) | (remask1 * ((bits >> 1) & 1)));
    }
};

struct QamPrbParam
{
    // bits [ 0:11] = remask0
    // bits [16:27] = remask1
    uint32_t bits;

    __host__ __device__ QamPrbParam() : bits(0) {}

    // Initialize with 2 x 12-bit masks
    __host__ __device__ inline void set(const uint32_t &remask0, const uint32_t &remask1)
    {
        bits = remask0 | (remask1 << 16);
    }

    // Return the number of bytes for this compressed PRB
    // Number of REs x bits per RE rounded to full bytes
    template <bool selective_sending = false>
    __host__ __device__ inline uint32_t comp_bytes(uint32_t bits_per_element) const
    {
        if constexpr (selective_sending)
            return (((bits_per_element * __builtin_popcount(bits)) + 7) / 8);
        else
            return ((bits_per_element * 12) / 8);
    }

    // Return true if the bit 'i' is set in mask[num], false otherwise
    // i must be in the range [0:11]
    template <int num>
    __host__ __device__ inline bool mask_on(uint32_t i) const
    {
        return bits & ((num ? 0x10000 : 0x1) << (11- i));
    }

    // Get mask[num], (bitmap of 12 bits)
    template <int num>
    __host__ __device__ inline uint32_t get_mask() const
    {
        if constexpr (num)
            return (bits >> 16); // Higher 4 bits not used, no need to msk
        else
            return (bits & 0xfff);
    }

    // Get complete mask (bit set to 1 if RE is used in one of the 2 masks)
    __host__ __device__ inline uint32_t get_mask() const
    {
        return ((bits | (bits >> 16)) & 0xfff);
    }
};
