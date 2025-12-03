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
#include <sys/time.h>
#include <signal.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include "nvlog.hpp"
#include "nv_utils.h"

#include "nv_phy_utils.hpp"
#include "cumac_cp_configs.hpp"
#include "cumac_app.hpp"
#include "msg_recv.hpp"
#include "signal_handler.hpp"

using namespace std;
using namespace nv;

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 1) // "CUMCP.MAIN"

// For signal handler
#define BT_BUF_SIZE 100
static struct sigaction sys_sa_handlers[__SIGRTMAX];

////////////////////////////////////////////////////////////////////////
void usage()
{
    NVLOGC_FMT(TAG, "Usage: cumac_cp <Fxx> <xC> [DL/UL] [--channels <channel names>]");
    NVLOGC_FMT(TAG, "Example: cumac_cp F08 2C --channels PDSCH+PDCCH_DL+PDCCH_UL+PBCH");
}

// Print backtrace (can't show function name and line info)
void print_backtrace(void)
{
    int nptrs;
    void *buffer[BT_BUF_SIZE];
    char **strings;

    nptrs = backtrace(buffer, BT_BUF_SIZE);
    NVLOGC_FMT(TAG, "backtrace() returned {} addresses", nptrs);

    // The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO) would produce similar output to the following
    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL)
    {
        NVLOGC_FMT(TAG, "backtrace backtrace_symbols error");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < nptrs; j++) {
        NVLOGC_FMT(TAG, "BT-{}: {}", j, strings[j]);
    }

    free(strings);
}

// SIGNAL handler function
void sigaction_handler(int signal)
{
    NVLOGC_FMT(TAG, "{}: received SIGNAL {} - {} - {}", __func__, signal, sigabbrev_np(signal), sigdescr_np(signal));

    print_backtrace();

    usleep(1000L * 1000L);

    // Use printf after usleep, FMT logger will not be saved to file

    if (signal < 0 && signal < __SIGRTMAX)
    {
        printf("Invalid signal number: %d. Valid range: 0-%d\n", signal, __SIGRTMAX);
        return;
    }

    // Trigger core dump
    sigaction(signal, &sys_sa_handlers[signal], NULL);

    printf("%s: SIGNAL %d - exit\n", __func__, signal);
}

// Register handler function for a system SIGNAL
void register_signal(int signal)
{
    struct sigaction sa={0};
    struct sigaction* old = &sys_sa_handlers[signal];

    sa.sa_handler=sigaction_handler;
    sigaction(signal, &sa, old);
}

// Setup SIGNALs to do FMT log clean up before exiting
void signal_setup()
{
    register_signal(SIGSEGV);
    register_signal(SIGABRT);
}

int parse_integer_value(char* arg, uint64_t* mask)
{
    if(arg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: null parameter", __FUNCTION__);
        return -1;
    }
    NVLOGI_FMT(TAG, "{}: argv={}", __FUNCTION__, arg);

    char* err_ptr = NULL;
    if(strncmp(arg, "0b", 2) == 0 || strncmp(arg, "0B", 2) == 0)
    {
        *mask = strtoull(arg + 2, &err_ptr, 2); // Binary
    }
    else
    {
        *mask = strtoull(arg, &err_ptr, 0); // Octal, Decimal, Hex
    }

    if(err_ptr == NULL || *err_ptr != '\0')
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid integer parameter: {}", __FUNCTION__, arg);
        return -1;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Debug starting CPU core issue
    printf("Started cumac_cp on CPU core %d\n", sched_getcpu());

    // Print starting time to debug starting CPU core issue
    char ts_buf[32] = "";
    struct timeval tv;
    struct tm ptm;
    gettimeofday(&tv, NULL);
    if(localtime_r(&tv.tv_sec, &ptm) != NULL)
    {
        // size = 8 + 7 = 15
        size_t size = strftime(ts_buf, sizeof("00:00:00"), "%H:%M:%S", &ptm);
        size += snprintf(ts_buf + size, 8, ".%06ld", tv.tv_usec);
    }
    printf("Started cumac_cp on CPU core %d at: %s\n", sched_getcpu(), ts_buf);

    assign_thread_name("cumac_cp_init");
    try{
        // Parse input parameters
        int show_thrput = 0;
        int no_validation = 0;
        uint64_t cell_mask           = 0;
        uint32_t channel_mask        = 0;
        uint64_t mode = 0;
        string   launch_pattern_file = "launch_pattern";
        string   config_yaml = CONFIG_CUMAC_CONFIG_YAML;
        for(int i = 1; i < argc; i++)
        {
            if(strncmp(argv[i], "--channels", strlen("--channels")) == 0 && i < argc - 1)
            {
                // Parse channels
                i++;
                // if(parse_channel_mask(argv[i], &channel_mask) < 0)
                // {
                //     return -1;
                // }
            }
            else if(strncmp(argv[i], "--cells", strlen("--cells")) == 0)
            {
                // Parse cells
                i++;
                if(parse_integer_value(argv[i], &cell_mask) < 0)
                {
                    return -1;
                }
            }
            else if(strncmp(argv[i], "--mode", strlen("--mode")) == 0)
            {
                // Parse cells
                i++;
                if(parse_integer_value(argv[i], &mode) < 0)
                {
                    return -1;
                }
            }
            else if(strncmp(argv[i], "--config", strlen("--config")) == 0)
            {
                // Parse config yaml file name
                i++;
                config_yaml = std::string(argv[i]);
            }
            else if(strncmp(argv[i], "--thrput", strlen("--thrput")) == 0)
            {
                // Parse channels
                show_thrput = 1;
            }
            else if(strncmp(argv[i], "--no-validation", strlen("--no-validation")) == 0)
            {
                no_validation = 1;
            }
            else
            {
                // Parse launch pattern file name
                launch_pattern_file.append("_").append(argv[i]);
            }
        }
        // printf("Parsed args: cell_mask=0x%lX channel_mask=0x%X config=%s lp=%s.yaml\n", cell_mask, channel_mask, config_yaml.c_str(), launch_pattern_file.c_str());

        // Open cumac_cp_config.yaml and create nvlog instance
        char cumac_cp_yaml[MAX_PATH_LEN];
        get_full_path_file(cumac_cp_yaml, NULL, CONFIG_CUMAC_CONFIG_YAML, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        NVLOGC_FMT(TAG, "cumac_cp_yaml={}", cumac_cp_yaml);

        yaml::file_parser fp(cumac_cp_yaml);
        yaml::document    doc       = fp.next_document();
        yaml::node        yaml_root = doc.root();

        // Bind low-priority threads to configured core
        int low_priority_core = -1;
        if (yaml_root.has_key("low_priority_core")) {
            low_priority_core = yaml_root["low_priority_core"].as<int>();
        }
        if (low_priority_core >= 0) {
            nv_assign_thread_cpu_core(low_priority_core);
        }
        NVLOGC_FMT(TAG, "low_priority_core={}", low_priority_core);

        // Load nvlog configuration and open the logger
        std::string log_name = yaml_root["log_name"].as<std::string>();
        std::string ipc_transport = yaml_root["transport"]["type"].as<std::string>();
        // Relative path of this process is $cuBB_SDK/build/cuPHY-CP/testMAC/testMAC/
        char        yaml_file[1024];
        get_full_path_file(yaml_file, NULL, NVLOG_DEFAULT_CONFIG_FILE, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

        pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file, log_name.c_str(), NULL);
        nvlog_fmtlog_thread_init();
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

        // signal_setup();
        generic_signal_setup();

        launch_pattern_file.append(".yaml");
        NVLOGC_FMT(TAG, "Run {} {}", argv[0], launch_pattern_file.c_str());

        pthread_setname_np(pthread_self(), "cumac_cp");

        cumac_cp_configs* configs = new cumac_cp_configs(yaml_root);

        cumac_receiver receiver(yaml_root, *configs);
        receiver.start();

        // Infinite loop until exit
        receiver.join();

        nvlog_fmtlog_close(bg_thread_id);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "cumac_cp exit with exception: {}", e.what());
        sleep(1); // For logger to save to file
        return -1;
    }

    NVLOGC_FMT(TAG, "cumac_cp exit normally", __func__);
    return 0;
}
