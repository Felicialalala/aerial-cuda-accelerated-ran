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

#ifndef __LOCAL_SIGNAL_HANDLER__
#define __LOCAL_SIGNAL_HANDLER__

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

#define ENABLE_CELL_STOP_SIGNAL_HANDLER (1)

// Setup the test_mac old SIGNAL handler
void test_mac_signal_setup();

// Setup generic SIGNAL handler to do FMT log clean up before exiting
void generic_signal_setup();

// Get a timestamp string with "1970-01-01 00:00:00" style
void get_gettimeofday_str(char* ts_buf);

// Print backtrace (can't show function name and line info)
void log_backtrace_printf(char* thread_name); // Print by printf()
void log_backtrace_fmt(char* thread_name);    // Print by nvlog FMT logger

#endif // __LOCAL_SIGNAL_HANDLER__