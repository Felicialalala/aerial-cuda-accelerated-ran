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

#ifndef BLOCKFP_TEST_KERNELS_H
#define BLOCKFP_TEST_KERNELS_H

#include <string>
#include <cstdint>

// N_ANT and N_LAYERS are fixed compile-time constants in the .cu file due to their
// use as template parameters in CUDA kernels.
// This wrapper function will execute the main logic of the BlockFP test.
void execute_blockfp_processing(
    uint32_t nfreqs,
    uint8_t compbits,
    float beta,
    int16_t beam_id_offset,
    const std::string& filename, // HDF5 input file, empty if generating random data
    bool verbose
);

#endif // BLOCKFP_TEST_KERNELS_H 