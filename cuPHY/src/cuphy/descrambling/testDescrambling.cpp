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

#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "descrambling.cuh"
#include "descrambling.hpp"

using namespace descrambling;

uint32_t reverse(uint32_t x, int bits)
{
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);   // Swap _<>_
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);   // Swap __<>__
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);   // Swap ____<>____
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);   // Swap ...
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16); // Swap ...
    return x >> (32 - bits);
}

int testDescrambling(uint32_t nTBs, uint32_t TBSize)
{
    // test result
    int      res  = 0;
    uint32_t size = nTBs * TBSize;

    std::vector<float>    llrs(size);
    std::vector<float>    gpuOut(size);
    std::vector<uint32_t> cinitArray(nTBs);
    std::vector<uint32_t> cpuSeq(size);
    std::vector<uint32_t> tbBoundaryArray(nTBs + 1);

    // populate input parameters

    for(int i = 0; i < size; i++) llrs[i] = i;

    for(int i = 0; i < nTBs; i++) cinitArray[i] = 1507348;

    tbBoundaryArray[0] = 0;
    for(int i = 1; i <= nTBs; i++)
        tbBoundaryArray[i] = tbBoundaryArray[i - 1] + TBSize;

    uint32_t maxNCodeBlocks = TBSize / (GLOBAL_BLOCK_SIZE);

    // generate CPU scrambling sequence

    std::vector<uint32_t> x1(NC + size + 31);
    std::vector<uint32_t> x2(NC + size + 31);

    for(int i = 0; i < nTBs; i++)
    {
        for(int n = 0; n < 31; n++)
        {
            x2[n] = (cinitArray[i] >> n) & 0x1;
        }

        x1[0] = 1;

        for(int j = 0; j < NC + tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            x1[j + 31] = (x1[j + 3] + x1[j]) & 0x1;

            x2[j + 31] = (x2[j + 3] + x2[j + 2] + +x2[j + 1] + x2[j]) & 0x1;
        }
        /*
     for (n = 0; n < len; n++) printf("%d, ", x1[n]);
    */

        for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            cpuSeq[tbBoundaryArray[i] + j] = (x1[j + NC] + x2[j + NC]) & 0x1;
        }
    }

    void* descrambleEnv[1];
    cuphyDescrambleInit(descrambleEnv);

    cuphyStatus_t status;

    status = cuphyDescrambleLoadParams(descrambleEnv,
                                       nTBs,
                                       maxNCodeBlocks,
                                       tbBoundaryArray.data(),
                                       cinitArray.data());

    status = cuphyDescrambleLoadInput(descrambleEnv, llrs.data());

    status = cuphyDescramble(descrambleEnv, nullptr, true, 1000, 0);
    status = cuphyDescrambleStoreOutput(descrambleEnv, gpuOut.data());

    // CPU computation
    for(int i = 0; i < nTBs; i++)
    {
        for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            if(cpuSeq[tbBoundaryArray[i] + j])
            {
                llrs[tbBoundaryArray[i] + j] = -llrs[tbBoundaryArray[i] + j];
            }
        }
    }

    res = 1;
    for(int i = 0; i < size; i++)
    {
        if(gpuOut[i] != llrs[i])
        {
            std::cout << "Error: not equal at " << i << "(" << size << ") "
                      << gpuOut[i] << " " << llrs[i] << "\n";
            res = 0;
        }
    }
    return res;
}

int DESCRAMBLE_TEST()
{
    int tbSize = 117504;
    int nTBs   = 8;

    int res = testDescrambling(nTBs, tbSize);

    return res;
}

int FIBONACCI_LFSR1_OPT_TEST()
{
    // Test all possible u32 bit patterns
    for (uint64_t i = 0; i <= 0xFFFFFFFF; i++) {
        uint32_t init_state = i & 0xFFFFFFFF;
        uint32_t ref_state = init_state, test_state = init_state;
        const uint32_t ref = fibonacciLFSR1(ref_state, 32);
        const uint32_t test = fibonacciLFSR1_n32(test_state);
        if (ref != test) {
            printf("Error on input 0x%x: expected result 0x%x, got 0x%x\n", init_state, ref, test);
            return 0;
        }
        if (ref_state != test_state) {
            printf("Error on input 0x%x: expected state 0x%x, got 0x%x\n", init_state, ref_state, test_state);
            return 0;
        }
    }
    return 1;
}

int FIBONACCI_LFSR2_OPT_TEST()
{
    // Test all possible u32 bit patterns
    for (uint64_t i = 0; i <= 0xFFFFFFFF; i++) {
        uint32_t init_state = i & 0xFFFFFFFF;
        uint32_t ref_state = init_state, test_state = init_state;
        const uint32_t ref = fibonacciLFSR2(ref_state, 32);
        const uint32_t test = fibonacciLFSR2_n32(test_state);
        if (ref != test) {
            printf("Error on input 0x%x: expected result 0x%x, got 0x%x\n", init_state, ref, test);
            return 0;
        }
        if (ref_state != test_state) {
            printf("Error on input 0x%x: expected state 0x%x, got 0x%x\n", init_state, ref_state, test_state);
            return 0;
        }
    }
    return 1;
}

// __host__ adaptation of the __device__ version in descrambling.cuh
inline uint32_t galois31MaskLFSRWordHost(uint32_t state)
{
    const uint32_t rev_state = reverse(state, 32);
    const uint32_t res =
        ((rev_state >> 1) & 0xFFFFFFF) |
        // bit 28 - s(2,0) ^ s(30,0)
        (((state & 0x4) << 26) ^ ((state & 0x40000000) >> 2)) |
        // bit 29 - s(1,0) ^ s(30,0) ^ s(29,0)
        (((state & 0x2) << 28) ^ ((state & 0x40000000) >> 1) ^ (state & 0x20000000)) |
        // bit 30 - s(0,0) ^ s(30,0) ^ s(29,0) ^ s(28,0)
        (((state & 0x1) << 30) ^ ((state & 0x40000000) >> 0) ^ ((state & 0x20000000) << 1) ^ ((state & 0x10000000) << 2));
        // bit 31 is 0
    return res;
}

int GALOIS31_LFSRWORD_OPT_TEST()
{
    // Test all possible u32 bit patterns
    for (uint64_t i = 0; i <= 0xFFFFFFFF; i++) {
        uint32_t state = i & 0xFFFFFFFF;
        const uint32_t ref = galois31LFSRWord(state, 0xF, 31);
        const uint32_t test = galois31MaskLFSRWordHost(state);
        if (ref != test) {
            printf("Error on input 0x%x: expected result 0x%x, got 0x%x\n", state, ref, test);
            return 0;
        }
    }
    return 1;
}

TEST(DESCRAMBLE, SAME_SIZE_TBS) { EXPECT_EQ(DESCRAMBLE_TEST(), 1); }
TEST(DESCRAMBLE, FIBONACCI_LFSR1_OPT) { EXPECT_EQ(FIBONACCI_LFSR1_OPT_TEST(), 1); }
TEST(DESCRAMBLE, FIBONACCI_LFSR2_OPT) { EXPECT_EQ(FIBONACCI_LFSR2_OPT_TEST(), 1); }
TEST(DESCRAMBLE, GALOIS31_LFSRWORD_OPT) { EXPECT_EQ(GALOIS31_LFSRWORD_OPT_TEST(), 1); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}
