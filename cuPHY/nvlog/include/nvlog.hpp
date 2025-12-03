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

#ifndef _NVLOG_HPP_
#define _NVLOG_HPP_

#include <stdint.h>

#include <sstream>
#include <iostream>

#include "nvlog.h"
#include "nv_utils.h"

#include "yaml-cpp/yaml.h"

#include "nvlog_fmt.hpp"


// Initialize the fmtlog for nvlog
pthread_t nvlog_fmtlog_init(const char* yaml_file, const char* name,void (*exit_hdlr_cb)());
void nvlog_fmtlog_thread_init();
void nvlog_fmtlog_thread_init(const char* name);

// close the fmtlog for nvlog
void nvlog_fmtlog_close(pthread_t bg_thread_id);

int get_root_path(char* path, int cubb_root_path_relative_num);
int get_full_path_file(char* dest_buf, const char* relative_path, const char* file_name, int cubb_root_dir_relative_num);

#endif /* _NVLOG_HPP_ */
