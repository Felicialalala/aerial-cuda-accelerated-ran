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
#include "cuphy.hpp"


// Size 1 byte (0 padding bytes), Alignment: 1 byte
typedef struct
{
  uint8_t   u8;
} testDescrType1;

// Size 16 bytes (5 padding bytes), Alignment: 8 bytes
typedef struct
{
  uint16_t  u16;
  uint8_t   u8;
  uint64_t  u64;
} testDescrType2;
 
// Size 8 bytes (3 padding bytes), Alignment: 4 bytes
typedef struct
{
  uint8_t   u8;
  uint32_t  u32;
} testDescrType3;

// Size 8 bytes (0 padding bytes), Alignment: 8 bytes
typedef struct
{
  double    d;
} testDescrType4;

////////////////////////////////////////////////////////////////////////
// Test descriptor allocation with synthetic sizes and alignments
TEST(Descriptor, Synthetic)
{
    static constexpr uint32_t N_DESCR = 3;
    std::array<size_t, N_DESCR> descrSizeBytes{3,2,20};
    std::array<size_t, N_DESCR> descrAlignBytes{1,1,4};

    cuphy::kernelDescrs<N_DESCR> testKernelDescr("testDescriptor");

    testKernelDescr.alloc(descrSizeBytes, descrAlignBytes);
    testKernelDescr.displayDescrSizes();

    auto cpuDescrStartAddrs = testKernelDescr.getCpuStartAddrs();
    auto gpuDescrStartAddrs = testKernelDescr.getGpuStartAddrs();

    for(uint32_t i = 0; i < N_DESCR; ++i)
    {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(cpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(gpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
    }
}

////////////////////////////////////////////////////////////////////////
// Test descriptor allocation for different sizes and alignments
TEST(Descriptor, Basic)
{
    static constexpr uint32_t N_DESCR = 5;
    std::array<size_t, N_DESCR> descrSizeBytes{sizeof(testDescrType1), sizeof(testDescrType2), sizeof(testDescrType1), sizeof(testDescrType3), sizeof(testDescrType4)};
    std::array<size_t, N_DESCR> descrAlignBytes{alignof(testDescrType1), alignof(testDescrType2), sizeof(testDescrType1), alignof(testDescrType3), alignof(testDescrType4)};

    cuphy::kernelDescrs<N_DESCR> testKernelDescr("testDescriptor");

    testKernelDescr.alloc(descrSizeBytes, descrAlignBytes);
    testKernelDescr.displayDescrSizes();

    auto cpuDescrStartAddrs = testKernelDescr.getCpuStartAddrs();
    auto gpuDescrStartAddrs = testKernelDescr.getGpuStartAddrs();

    for(uint32_t i = 0; i < N_DESCR; ++i)
    {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(cpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(gpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
    }
}

////////////////////////////////////////////////////////////////////////
// Test with a descriptor set which includes an empty descriptor
TEST(Descriptor, Empty)
{
    static constexpr uint32_t N_DESCR = 5;
    std::array<size_t, N_DESCR> descrSizeBytes{sizeof(testDescrType1), 0, sizeof(testDescrType1), sizeof(testDescrType3), sizeof(testDescrType4)};
    std::array<size_t, N_DESCR> descrAlignBytes{alignof(testDescrType1), 0, sizeof(testDescrType1), alignof(testDescrType3), alignof(testDescrType4)};

    cuphy::kernelDescrs<N_DESCR> testKernelDescr("testDescriptor");

    testKernelDescr.alloc(descrSizeBytes, descrAlignBytes);
    testKernelDescr.displayDescrSizes();

    auto cpuDescrStartAddrs = testKernelDescr.getCpuStartAddrs();
    auto gpuDescrStartAddrs = testKernelDescr.getGpuStartAddrs();

    for(uint32_t i = 0; i < N_DESCR; ++i)
    {
      if(0 != descrAlignBytes[i]) // Check only for descriptors with non-zero alignments
      {
         EXPECT_EQ(reinterpret_cast<uintptr_t>(cpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
         EXPECT_EQ(reinterpret_cast<uintptr_t>(gpuDescrStartAddrs[i]) % descrAlignBytes[i], 0);
      }
    }
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    return result;
}