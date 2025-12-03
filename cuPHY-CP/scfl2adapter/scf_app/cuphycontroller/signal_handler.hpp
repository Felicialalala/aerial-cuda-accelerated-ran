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

// Register handler function for a system SIGNAL
void register_signal(int signal);

// Setup SIGNALs to do FMT log clean up before exiting
void signal_setup();

#endif // __LOCAL_SIGNAL_HANDLER__