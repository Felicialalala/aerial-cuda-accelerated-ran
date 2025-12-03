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

#if !defined(SEQUENCE_HPP_INCLUDED_)
#define SEQUENCE_HPP_INCLUDED_
#include <stdint.h>

#define MAX_INPUT_TB_SIZE_IN_WORDS ((MAX_ENCODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED) / 32)

namespace descrambling
{
constexpr uint32_t     BITS_PROCESSED_PER_LUT_ENTRY      = 32;
constexpr uint32_t     BITS_PROCESSED_PER_LUT_ENTRY_MASK = (1UL << BITS_PROCESSED_PER_LUT_ENTRY) - 1;
constexpr uint32_t     POLY_1                            = 0x80000009;
constexpr uint32_t     POLY_2                            = 0x8000000F;
constexpr uint32_t     POLY_1_GMASK                      = 0x00000009;
constexpr uint32_t     POLY_2_GMASK                      = 0x0000000F;
constexpr uint32_t     GLOBAL_BLOCK_SIZE                 = 512;
constexpr uint32_t     WARP_SIZE                         = 32;
constexpr uint32_t     WORD_SIZE                         = sizeof(uint32_t) * 8;
// Nc value from 5G spec: number of bits skipped for both LFSRs
constexpr uint32_t NC = 1600;
} // namespace descrambling
#endif
