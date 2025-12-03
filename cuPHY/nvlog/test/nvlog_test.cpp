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

#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <iostream>

#include "nv_utils.h"
#include "nvlog.hpp"

#define TAG_TEST (NVLOG_TAG_BASE_NVLOG + 1) // "NVLOG.TEST"
#define ITAG (NVLOG_TAG_BASE_NVLOG + 2)     // "NVLOG.ITAG"
#define STAG "NVLOG.STAG"

#ifdef NVLOG_USE_INTEGER_TAG_IN_C
#define CTAG ITAG
#else
#define CTAG STAG
#endif

void test_basic_log_print()
{
    NVLOGE_FMT(STAG, AERIAL_NVLOG_EVENT, "This is STAG C printf style error log. level={}", NVLOG_ERROR);
    NVLOGI_FMT(STAG, "This is STAG C printf style log. level={}", NVLOG_INFO);

    // printf style log: NVLOGE_FMT, NVLOGC_FMT, NVLOGW_FMT, NVLOGI_FMT, NVLOGD_FMT, NVLOGV_FMT
    NVLOGE_FMT(ITAG, AERIAL_NVLOG_EVENT, "This is ITAG C printf style error log. level={}", NVLOG_ERROR);
    NVLOGI_FMT(ITAG, "This is ITAG C printf style log. level={}", NVLOG_INFO);

    NVLOGE_FMT(STAG, AERIAL_NVLOG_EVENT, "This is STAG C printf style error log. Error code=1");
    NVLOGE_FMT(ITAG, AERIAL_NVLOG_EVENT, "This is ITAG C printf style error log. Error code=2");

    float vals[12] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.1, 12.2};
    // nvlog_print_array_2(NVLOG_CONSOLE, "FLOAT_ARRAY", vals);
    NVLOGW_FMT_ARRAY(ITAG, "FLOAT_ARRAY", vals, 8);
    NVLOGW_FMT_ARRAY(ITAG, "FLOAT_ARRAY", vals, 12);

    // Test log levels
    NVLOGV_FMT(ITAG, "Test NVLOGV_FMT: level={}", NVLOG_VERBOSE);
    NVLOGD_FMT(ITAG, "Test NVLOGD_FMT: level={}", NVLOG_DEBUG);
    NVLOGI_FMT(ITAG, "Test NVLOGI_FMT: level={}", NVLOG_INFO);
    NVLOGW_FMT(ITAG, "Test NVLOGW_FMT: level={}", NVLOG_WARN);
    NVLOGC_FMT(ITAG, "Test NVLOGC_FMT: level={}", NVLOG_CONSOLE);
    NVLOGE_FMT(ITAG, AERIAL_NVLOG_EVENT, "Test NVLOGE_FMT: level={}", NVLOG_ERROR);
    NVLOG_FMT_EVT(fmtlog::FAT, ITAG, AERIAL_NVLOG_EVENT, "Test NVLOGF_FMT without exiting: level={}", NVLOG_FATAL);
}

__thread int nvlog_errno = 0;

// Print log to console

static void nvlog_set_errno(int no)
{
    nvlog_errno = no;
}

static int nvlog_get_errno(void)
{
    return nvlog_errno;
}

void test_multi_process_same_log_file(void) // Note: set the same log name
{
    pid_t fpid = fork();
    if(fpid < 0)
    {
        return;
    }
    else if(fpid == 0)
    {
        sleep(1); // Wait for primary to start log first
        // Later started process 2: primary=0, name="nvlog"
        NVLOGI_FMT(ITAG, "Forked Child process: PID={}", getpid());
    }
    else
    {
        // First started process 1: primary=1, name="nvlog"
        NVLOGI_FMT(ITAG, "Forked Parent process: Child PID={}", getpid());
    }
}

void test_log_performance(void)
{
    NVLOGC_FMT(ITAG, "Start test log performance ...");
    int j;
    for(j = 0; j < 10; j++)
    {
        struct timespec last, now;
        clock_gettime(CLOCK_REALTIME, &last);

        int i;
        //for(i = 0; i < 1000 * 1000; i++)
        for(i = 0; i < 5000; i++)
        {
            NVLOGI_FMT(ITAG, "test {}", i);
        }
        clock_gettime(CLOCK_REALTIME, &now);
        NVLOGC_FMT(ITAG, "Tested {} line, average time cost for one line NVLOGI_FMT is {} ns", i, nvlog_timespec_interval(&last, &now) / i);
    }
}

void* thread_func_2(void* args)
{
#ifdef NVIPC_FMTLOG_ENABLE
    fmtlog::setThreadName("nvlog_test2");
#endif
    if(pthread_setname_np(pthread_self(), "nvlog_test2") != 0)
    {
        NVLOGE_FMT(ITAG,AERIAL_NVLOG_EVENT, "{}: set thread name failed", __func__);
    }

    nvlog_set_errno(222);
    int test_count = 3;
    while(test_count-- > 0)
    {
        nvlog_set_errno(nvlog_get_errno() + 2);
        NVLOGC_FMT(ITAG, "nvlog_test2 running ... nvlog_errno={}", nvlog_get_errno());
        sleep(1);
    }

    return NULL;
}

void test_multi_thread_errno(void)
{
    pthread_t pthread_id;
    if(pthread_create(&pthread_id, NULL, thread_func_2, NULL) != 0)
    {
        NVLOGE_FMT(ITAG,AERIAL_NVLOG_EVENT, "{}: pthread_create failed", __func__);
    }

    nvlog_set_errno(111);
    int test_count = 3;
    while(test_count-- > 0)
    {
        nvlog_set_errno(nvlog_get_errno() + 1);
        NVLOGC_FMT(ITAG, "nvlog_test1 running ... nvlog_errno={}", nvlog_get_errno());
        sleep(1);
    }

    if(pthread_join(pthread_id, NULL) != 0)
    {
        NVLOGE_FMT(STAG, AERIAL_NVLOG_EVENT, "{}: pthread_join failed: pthread_id={}", __func__, pthread_id);
    }
}

int main(int argc, char* argv[])
{
    // Relative path of this process is $cuBB_SDK/build/cuPHY-CP/gt_common_libs/nvlog/test/nvlog_test
    char        yaml_file[1024];
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(yaml_file, relative_path.c_str());

    printf("Relative path: %s\n",relative_path.c_str());
    printf("Absolute path: %s\n",yaml_file);
    pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file, "nvlog_test.log",NULL);
    nvlog_fmtlog_thread_init();

    // The thread name length can be 15 at most
    if(pthread_setname_np(pthread_self(), "nvlog_test1") != 0)
    {
        NVLOGE_FMT(ITAG, AERIAL_NVLOG_EVENT, "{}: set thread name failed", __func__);
    }
#ifdef NVIPC_FMTLOG_ENABLE
    fmtlog::setThreadName("nvlog_test1");
#endif

    test_basic_log_print();


    // Test performance
    NVLOGW_FMT(ITAG, "Testing log performance with default log level");
    test_log_performance();

    NVLOGW_FMT(ITAG, "Testing log performance with INF log level");

    for (int k=0; k<NVLOG_FMTLOG_NUM_TAGS; k++)
    {
#ifdef NVIPC_FMTLOG_ENABLE
        g_nvlog_component_levels[k] = fmtlog::INF;
#endif
    }

    test_log_performance();

    usleep(2000000);
    NVLOGW_FMT(ITAG, "Done with performance test");

    test_multi_thread_errno();

    NVLOGW_FMT(ITAG, "Last log message");

    // NVLOGF_FMT(ITAG, AERIAL_NVLOG_EVENT, "Test NVLOGF_FMT: last log message with level={} - exiting", NVLOG_FATAL);

    nvlog_fmtlog_close(bg_thread_id);

    return 0;
}
