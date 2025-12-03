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

#include "signal_handler.hpp"
#include "nvlog.hpp"
#include "nv_utils.h"

using namespace std;

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 0) // "MAC"

// For signal handler
#define BT_BUF_SIZE 100
static struct sigaction sys_sa_handlers[__SIGRTMAX];

#define ENABLE_CORE_DUMP_IN_SIGNAL_HANDLER (0)

void get_gettimeofday_str(char* ts_buf)
{
    if(ts_buf == nullptr)
    {
        return;
    }

    struct timeval tv;
    struct tm      ptm;
    gettimeofday(&tv, NULL);
    if(localtime_r(&tv.tv_sec, &ptm) != NULL)
    {
        // size = 8 + 7 = 15
        size_t size = strftime(ts_buf, sizeof("00:00:00"), "%H:%M:%S", &ptm);
        size += snprintf(ts_buf + size, 8, ".%06ld", tv.tv_usec);
    }
}

// Print backtrace (can't show function name and line info)
void log_backtrace_printf(char* thread_name)
{
    int    nptrs;
    void*  buffer[BT_BUF_SIZE];
    char** strings;

    nptrs = backtrace(buffer, BT_BUF_SIZE);
    printf("thread [%s] backtrace() returned %d addresses\n", thread_name, nptrs);

    // The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO) would produce similar output to the following
    strings = backtrace_symbols(buffer, nptrs);
    if(strings == NULL)
    {
        printf("thread [%s] backtrace backtrace_symbols error\n", thread_name);
        exit(EXIT_FAILURE);
    }

    for(int j = 0; j < nptrs; j++)
    {
        printf("thread [%s] BT-%d: %s\n", thread_name, j, strings[j]);
    }

    free(strings);
}

void log_backtrace_fmt(char* thread_name)
{
    int    nptrs;
    void*  buffer[BT_BUF_SIZE];
    char** strings;

    nptrs = backtrace(buffer, BT_BUF_SIZE);
    NVLOGC_FMT(TAG, "thread [{}] backtrace() returned {} addresses", thread_name, nptrs);

    // The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO) would produce similar output to the following
    strings = backtrace_symbols(buffer, nptrs);
    if(strings == NULL)
    {
        NVLOGC_FMT(TAG, "thread [{}] backtrace backtrace_symbols error", thread_name);
        exit(EXIT_FAILURE);
    }

    for(int j = 0; j < nptrs; j++)
    {
        NVLOGC_FMT(TAG, "thread [{}] BT-{}: {}", thread_name, j, strings[j]);
    }

    free(strings);
}

// SIGNAL handler function
static void sigaction_handler(int signum)
{
    char thread_name[16];
    pthread_getname_np(pthread_self(), thread_name, 16);

    NVLOGC_FMT(TAG, "{}: thread [{}] received SIGNAL {} - {} - {}", __func__, thread_name, signum, sigabbrev_np(signum), sigdescr_np(signum));

    char ts_buf[32] = "";
    get_gettimeofday_str(ts_buf);
    printf("%s [PRINTF] %s: thread [%s] received SIGNAL %d - %s - %s\n", ts_buf, __func__, thread_name, signum, sigabbrev_np(signum), sigdescr_np(signum));

    // if(ENABLE_CORE_DUMP_IN_SIGNAL_HANDLER)
    // {
        log_backtrace_fmt(thread_name);
    // }

    usleep(1000L * 1000L);

    // Use printf after usleep, FMT logger will not be saved to file

    if(signum < 0 && signum < __SIGRTMAX)
    {
        get_gettimeofday_str(ts_buf);
        printf("%s [PRINTF] thread [%s] invalid signum: %d. Valid range: 0-%d\n", ts_buf, thread_name, signum, __SIGRTMAX);
        return;
    }

    if(ENABLE_CORE_DUMP_IN_SIGNAL_HANDLER)
    {
        // Trigger core dump
        sigaction(signum, &sys_sa_handlers[signum], NULL);
    }

    get_gettimeofday_str(ts_buf);
    printf("%s [PRINTF] %s: thread [%s] SIGNAL %d handler finished, exit\n", ts_buf, __func__, thread_name, signum);

    exit(EXIT_FAILURE);
}

// Register handler function for a system SIGNAL
static void register_signal(int signum)
{
    NVLOGC_FMT(TAG, "{}: SIGNAL {} - {} - {}", __func__, signum, sigabbrev_np(signum), sigdescr_np(signum));

    if(ENABLE_CORE_DUMP_IN_SIGNAL_HANDLER)
    {
        struct sigaction  sa  = {0};
        struct sigaction* old = &sys_sa_handlers[signum];

        sa.sa_handler = sigaction_handler;
        sigaction(signum, &sa, old);
    }
    else
    {
        signal(signum, sigaction_handler);
    }
}

// Setup SIGNALs to do FMT log clean up before exiting
void generic_signal_setup()
{
    NVLOGC_FMT(TAG, "{}: register signal handler", __func__);
    register_signal(SIGINT);
    register_signal(SIGABRT);
    register_signal(SIGKILL);
    // register_signal(SIGSEGV);
    register_signal(SIGTERM);
}