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

#ifndef _NV_UTILS_H_
#define _NV_UTILS_H_

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

// Convert process relative path to absolute path
int nv_get_absolute_path(char* absolute_path, const char* relative_path);

// For CPU core binding and priority setting
int nv_set_sched_fifo_priority(int priority);
int nv_assign_thread_cpu_core(int cpu_id);

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#endif /* _NV_UTILS_H_ */
