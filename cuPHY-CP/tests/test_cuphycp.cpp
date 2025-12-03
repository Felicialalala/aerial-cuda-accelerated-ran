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

#include <stdio.h>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "nvlog.hpp"

pthread_t g_bg_thread_id;
void exit_handler()
{
    nvlog_fmtlog_close(g_bg_thread_id);
    abort();
}

enum test_mode
{
    TEST = 0,
    BENCH = 1,
    HELP = 2
};

int main(int argc, char** argv)
{
    test_mode mode = test_mode::HELP;

    for (int k=1; k<argc; k++)
    {
        if (strcmp(argv[k],"--test") == 0)
        {
            mode = test_mode::TEST;
            break;
        }
        else if (strcmp(argv[k],"--bench") == 0)
        {
            mode = test_mode::BENCH;
            break;
        }
    }

    if (mode == test_mode::HELP)
    {
        printf("Usage:\n");
        printf("  Unit Tests: %s --test <other test options, -h for help>\n",argv[0]);
        printf("  Benchmarks: %s --bench <other bench options, -h for help>\n",argv[0]);
        return 1;
    }

    char        yaml_file[1024];
    std::string relative_path = std::string("../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(yaml_file, relative_path.c_str());
    g_bg_thread_id = nvlog_fmtlog_init(yaml_file, "test_cuphycp.log",exit_handler);
    nvlog_fmtlog_thread_init();
    sleep(1); // let logger thread statup

    if (mode == test_mode::TEST)
    {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    else
    {
        ::benchmark::Initialize(&argc, argv);
        return ::benchmark::RunSpecifiedBenchmarks();
    }
}