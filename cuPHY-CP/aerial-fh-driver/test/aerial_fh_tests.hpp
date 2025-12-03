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

#ifndef AERIAL_FH_TESTS_HPP__
#define AERIAL_FH_TESTS_HPP__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <memory>

#include "gtest/gtest.h"
#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"
#include "flow.hpp"
#include "nic.hpp"
#include "peer.cpp"
#include "queue.hpp"
#include "utils.hpp"

using namespace aerial_fh;

#define UT_MTU 1514

#define ASSERT_CUDA(expr)            \
    {                                \
        cudaError_t err = (expr);    \
        ASSERT_EQ(err, cudaSuccess); \
    }

inline std::string get_nic_name()
{
    const char* name;
    if((name = getenv("UT_NIC")) != nullptr)
    {
        return name;
    }

    return "0000:b5:00.1";
}

inline uint64_t get_ns()
{
    struct timespec t = {};
    if(unlikely(clock_gettime(CLOCK_REALTIME, &t) != 0))
    {
        return 0;
    }
    return static_cast<uint64_t>(t.tv_nsec) + static_cast<uint64_t>(t.tv_sec) * 1000 * 1000 * 1000;
}

#endif //ifndef AERIAL_FH_TESTS_HPP__
