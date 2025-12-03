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

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include "test_utils.hpp"

const size_t CACHE_SIZE_BYTES = 36864*1024; // Ice Lake L3 cache size
const size_t CACHE_THRASHER_SIZE = CACHE_SIZE_BYTES*4 / sizeof(uint32_t);
auto g_ptr_cache_thrasher = std::make_unique<uint32_t[]>(CACHE_THRASHER_SIZE);
auto g_ptr_cache_thrasher_indices = std::make_unique<size_t[]>(CACHE_THRASHER_SIZE);
bool g_cache_indices_valid = false;
volatile int g_cache_thrasher_sum = 0;

static const char* g_env_aerial_test_cacheinvd = std::getenv("AERIAL_TEST_CACHEINVD");

void thrash_cache(void)
{
    if (g_env_aerial_test_cacheinvd == nullptr) return;

    // Attempt to use https://github.com/batmac/wbinvd kernel module if it's installed
    // If it isn't present, thrash the cache the old fashioned way (slower)
    FILE *fp = fopen("/proc/wbinvd","r");
    if (fp != nullptr)
    {
        char wbinvd[1024];
        int ret = fscanf(fp,"%s",wbinvd);
    }
    else
    {
        // Write to region larger than L3 cache size
        memset(g_ptr_cache_thrasher.get(),12345678,CACHE_THRASHER_SIZE*sizeof(uint32_t));

        // Random read from region larger than L3 cache size
        g_cache_thrasher_sum = 0;
        if (g_cache_indices_valid == false)
        {
            std::mt19937 rgen;
            rgen.seed(1234);
            for (size_t k=0; k<CACHE_THRASHER_SIZE; k++)
            {
                size_t index = rgen() % CACHE_THRASHER_SIZE;
                g_ptr_cache_thrasher_indices[k] = index;
            }
            g_cache_indices_valid = true;
        }

        for (size_t k=0; k<CACHE_THRASHER_SIZE; k++)
        {
            size_t index = g_ptr_cache_thrasher_indices[k];
            g_cache_thrasher_sum += g_ptr_cache_thrasher[index];
        }
    }
    fclose(fp);
}

