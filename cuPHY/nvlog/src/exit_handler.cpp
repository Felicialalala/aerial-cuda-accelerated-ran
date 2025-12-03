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

#include "nvlog.hpp"
#include "nvlog.h"
#include "exit_handler.hpp"
#include <string>

#define TAG (NVLOG_TAG_BASE_NVLOG + 10) // "NVLOG.EXIT_HANDLER"

// Define initial value of exit_handler::instance
exit_handler* exit_handler::instance = nullptr;

exit_handler::exit_handler()
{
    exit_handler_flag=L1_RUNNING;
}

exit_handler::~exit_handler()
{
    exit_handler_flag=L1_RUNNING;
}

void exit_handler::set_exit_handler_flag(l1_state val)
{
    exit_handler_flag=val;
}

uint32_t exit_handler::get_l1_state()
{
    return exit_handler_flag;
}

void exit_handler::test_trigger_exit(const char* file, int line, const char* info)
{
    // Note: DO NOT call NVLOGF_FMT() in this function to avoid nested calls.

    // Atomic fetch and set
    l1_state curr_state = exit_handler_flag.exchange(L1_EXIT);
    bool old_flag = (curr_state == L1_EXIT) ? true : false;
    char ts[NVLOG_TIME_STRING_LEN] = {'\0'};
    nvlog_gettimeofday_string(ts, NVLOG_TIME_STRING_LEN);

    int cpu_id = sched_getcpu();

    // Get thread name and CPU core number
    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);

    // Only one thread runs the exit callback and call exit(EXIT_FAILURE), other threads just wait.
    if(!old_flag)
    {
        // #define NVLOGE_FMT(component_id, event_level, format_fmt, ...) NVLOG_FMT_EVT(fmtlog::ERR,component_id, event_level, format_fmt, ##__VA_ARGS__)

        NVLOG_FMT_EVT(fmtlog::FAT, TAG, AERIAL_SYSTEM_API_EVENT, "FATAL exit: Thread [{}] on core {} file {} line {}: additional info: {}", thread_name, cpu_id, file, line, info);
        printf("%s FATAL exit: Thread [%s] on core %d file %s line %d: additional info: %s\n", ts, thread_name, cpu_id, file, line, info);

        // Sleep for logger to clean up
        usleep(100000);

        if (exit_cb != nullptr)
        {
            exit_cb();
        }
        exit(EXIT_FAILURE);
    }
    else
    {
        NVLOG_FMT_EVT(fmtlog::FAT, TAG, AERIAL_SYSTEM_API_EVENT, "FATAL already exiting: Thread [{}] on core {} file {} line {}: additional info: {}", thread_name, cpu_id, file, line, info);
        printf("%s FATAL already exiting: Thread [%s] on core %d file %s line %d: additional info: %s\n", ts, thread_name, cpu_id, file, line, info);
        usleep(1000 * 1000 * 20); // Sleep enough time to wait for ongoing exiting to finish
    }
}

bool exit_handler::test_exit_in_flight()
{
    return ((exit_handler_flag==L1_EXIT)?true:false);
}

void exit_handler::set_exit_handler_cb(void (*exit_hdlr_cb)())
{
    exit_cb=exit_hdlr_cb;
}