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

#ifndef _STAT_LOG_H_
#define _STAT_LOG_H_

#include <time.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define STAT_NAME_MAX_LEN (32)

// Decide the statistic period by timer or counter
typedef enum
{
    STAT_MODE_NONE    = 0,
    STAT_MODE_COUNTER = 1,
    STAT_MODE_TIMER   = 2,
    STAT_MODE_MAX     = 3
} print_mode_t;

typedef struct stat_log_t stat_log_t;
struct stat_log_t
{
    int (*init)(stat_log_t* stat);

    int (*add)(stat_log_t* stat, int64_t value);

    int (*time_interval)(stat_log_t* stat);

    int (*set_clock_source)(stat_log_t* stat, clockid_t clk_src);

    int (*set_log_level)(stat_log_t* stat, int log_level);

    int (*set_limit)(stat_log_t* stat, int64_t min, int64_t max);

    int32_t (*get_counter)(stat_log_t* stat);
    int64_t (*get_min_value)(stat_log_t* stat);
    int64_t (*get_max_value)(stat_log_t* stat);
    int64_t (*get_avg_value)(stat_log_t* stat);

    int (*print)(stat_log_t* stat);

    int (*close)(stat_log_t* stat);
};

stat_log_t* stat_log_open(const char* name, int mode, int64_t period);

int assert_time_order(struct timespec ts_old, struct timespec ts_new);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _STAT_LOG_H_ */
