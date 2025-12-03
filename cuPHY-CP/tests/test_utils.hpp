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

#pragma once
#include <optional>
#include <filesystem>
#include <cassert>
#include <sys/wait.h>

#include <unistd.h>
#include <signal.h>
#include "fmt/format.h"
#include <iostream>

void thrash_cache(void);

struct PerfProfiler {
public:
    PerfProfiler (std::optional<int> sampling_frequency_hz_, std::optional<std::filesystem::path> perf_data_dir_) 
    {
        if (perf_data_dir_) 
        {
            perf_data_dir = perf_data_dir_.value();
        } else
        {
            perf_data_dir = "/tmp/perf_data"; 
        }

        perf_enabled = validation_checks(perf_data_dir); 

        if (sampling_frequency_hz_) 
        {
            sampling_frequency_hz = sampling_frequency_hz_.value(); 
        } else 
        {
            sampling_frequency_hz = 999; 
        }
    }


    void start (const std::string &fname)
    {
        if (!perf_enabled) { return; }

        auto const current_pid = getpid();

        assert(perf_pid == 0);
        std::filesystem::path output_file = perf_data_dir / fname;
        perf_pid = fork();

        if (perf_pid == 0) {
            // Child process - build perf command arguments dynamically
            std::vector<char const*> args = {"perf", "record"};

            args.insert(args.end(),
                        {"--call-graph", "dwarf", 
                        "-g",
                        "-o", output_file.c_str(), 
                        "-p", std::to_string(current_pid).c_str(),
                        "-F", std::to_string(sampling_frequency_hz).c_str()
                        });
            
            args.push_back(NULL);

            execvp("perf", const_cast<char* const*>(args.data()));
            // You should never reach here if execvp succeeds.
            exit(1);
        }
    }

    void stop()
    {
        if (!perf_enabled) { return; }
        assert(perf_pid != 0);
        kill(perf_pid, SIGTERM); // Ask nicely! 
        waitpid(perf_pid, NULL, 0);
        perf_pid = 0;
    }

private:
    bool perf_enabled{}; 
    int sampling_frequency_hz{};
    std::filesystem::path perf_data_dir{}; 
    int perf_pid{0}; 

    bool validation_checks(std::filesystem::path &dir_name) 
    {
        bool status = is_perf_tool_available(); 
        if (status) 
        {
            if (std::filesystem::exists(perf_data_dir)) 
            {
                if (!std::filesystem::is_directory(dir_name)) 
                {
                    status = false; 
                    throw std::runtime_error(fmt::format("Perf data directory:{} is not a directory", dir_name.string()));
                }
            } else 
            {
                bool result = false; 
                try {
                    std::filesystem::create_directory(perf_data_dir);
                    status = true; 
                } catch (const std::filesystem::filesystem_error& e) {
                    // Handle error
                    status = false; 
                    throw std::runtime_error(fmt::format("create_directory failed. e:{}", e.what()));
                }
            }
        }
        return status;  
    }

    bool is_perf_tool_available() 
    {
        auto const result = std::system("which perf > /dev/null 2>&1"); 
        if (result != 0) 
        {
            throw std::runtime_error("perf tool not found!!"); 
            return false; 
        }

        return true; 
    }

}; 
