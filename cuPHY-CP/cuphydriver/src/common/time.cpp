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

#include "time.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 2) // "DRV.TIME"

uint64_t get_ns(void)
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime failed");
    }
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_ns() + ns, start_t = 0;
    while((start_t = get_ns()) < end_t)
    {
        for(int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
        {
            __asm__ __volatile__("");
        }
    }
}

Time::Time()
{
}

Time::~Time()
{
}

t_tp Time::nowTimepoint()
{
    return std::chrono::system_clock::now();
}

t_ns Time::nowNs()
{
    return std::chrono::system_clock::now().time_since_epoch();
}

t_ns Time::zeroNs()
{
    return t_ns::zero();
}

t_ns Time::getDifference(t_ns& first, t_ns& second)
{
    return first - second;
}

t_ns Time::getDifferenceNowToNs(t_ns& first)
{
    return nowNs() - first;
}

t_ns Time::getDifferenceNsToNow(t_ns& first)
{
    return first - nowNs();
}

bool Time::greater(t_ns& first, t_ns& second)
{
    return first > second;
}
bool Time::greatereq(t_ns& first, t_ns& second)
{
    return first >= second;
}

bool Time::belowThreshold(t_ns& first, t_ns& second, t_ns& threshold)
{
    return ((first - second) <= threshold);
}

bool Time::aboveThreshold(t_ns& first, t_ns& second, t_ns& threshold)
{
    return ((first - second) > threshold);
}

bool Time::waitDurationNs(t_ns& waitns)
{
    t_ns end_t = nowNs() + waitns;
    t_ns start_t;
    while((start_t = nowNs()) < end_t)
    {
        for(int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
        {
            __asm__ __volatile__("");
        }
    }

    return true;
}

bool Time::waitTimeNs(t_ns& waitns)
{
    t_ns start_t;
    while((start_t = nowNs()) < waitns)
    {
        for(int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
        {
            __asm__ __volatile__("");
        }
    }

    return true;
}

t_us Time::NsToUs(t_ns time)
{
    return std::chrono::duration_cast<t_us>(time);
}

t_ns Time::UsToNs(t_us time)
{
    return std::chrono::duration_cast<t_ns>(time);
}
