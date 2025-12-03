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

#ifndef TASK_INSTRUMENTATION_H
#define TASK_INSTRUMENTATION_H

#include "time.hpp"
#include <string.h>
#include <iostream>

#include "slot_map_dl.hpp"
#include "slot_map_ul.hpp"
#include "context.hpp"
#include "worker.hpp"

#include "task_instrumentation_nested.hpp"

/**
 * @brief Example task instrumentation usage:
 * 
 * TI_INIT("example task",10)
 * 
 * TI_ADD("subtask 1")
 *  ... CPU work
 * TI_ADD("subtask 2")
 *  ... CPU work
 * TI_ADD("subtask 3")
 *  ... CPU work
 * TI_ADD("End Task")
 * 
 * TI_NVSLOGI(sfn,slot,map,cpu)
 */

/**
 * @brief Maximum length of subtask name strings (including null terminator).
 */
#define MAX_SUBTASK_CHARS 32

/**
 * @brief Maximum length of NVSLOGI output buffer for task instrumentation.
 */
#define MAX_NVSLOGI_CHARS 1024

/**
 * @brief Maximum length of PMU metrics string.
 */
#define MAX_PMU_METRICS_CHARS 100

/**
 * @brief Initializes task instrumentation for a task.
 * 
 * Sets up timing arrays, task name, CPU affinity, PMU counters, and logs start time.
 * 
 * @param slot_map_type Type of slot map (SlotMapDl or SlotMapUl)
 * @param task_name Name of the task being instrumented
 * @param max_subtasks Maximum number of subtasks to track
 */
#define TI_INIT(slot_map_type,task_name,max_subtasks) \
slot_map_type* ti_slot_map = (slot_map_type*)param; \
PhyDriverCtx* ti_pdctx = StaticConversion<PhyDriverCtx>(ti_slot_map->getPhyDriverHandler()).get(); \
int ti_cpu_task_tracing = ti_pdctx->enableCPUTaskTracing(); \
char ti_task_name[MAX_SUBTASK_CHARS]; \
int ti_max_subtasks = max_subtasks; \
snprintf(ti_task_name,MAX_SUBTASK_CHARS,"%s",task_name); \
t_ns ti_times[max_subtasks]; \
char ti_subtask_names[max_subtasks][MAX_SUBTASK_CHARS]; \
int ti_subtask_count = 0; \
uint32_t ti_cpu; \
getcpu(&ti_cpu, nullptr); \
char pmu_metrics_str[MAX_PMU_METRICS_CHARS]; \
char subtask_results[MAX_NVSLOGI_CHARS]; \
if(ti_cpu_task_tracing==2) { \
    snprintf(&subtask_results[0], MAX_NVSLOGI_CHARS, "%s:%lu,", "s", Time::nowNs()); \
    NVLOGI_FMT(TAG,"{{mTI}} <{},{},{},{},{}> {}", \
    ti_task_name, ti_slot_map->getSlot3GPP().sfn_, ti_slot_map->getSlot3GPP().slot_, ti_slot_map->getId(), ti_cpu, \
    subtask_results); \
} \
worker->readStartCounters();

/**
 * @brief Initializes task instrumentation for a downlink task.
 * 
 * @param task_name Name of the task being instrumented
 * @param max_subtasks Maximum number of subtasks to track
 */
#define TI_INIT_DL(task_name,max_subtasks) TI_INIT(SlotMapDl,task_name,max_subtasks)

/**
 * @brief Initializes task instrumentation for an uplink task.
 * 
 * @param task_name Name of the task being instrumented
 * @param max_subtasks Maximum number of subtasks to track
 */
#define TI_INIT_UL(task_name,max_subtasks) TI_INIT(SlotMapUl,task_name,max_subtasks)

/**
 * @brief Appends a subtask entry with a specific timestamp.
 * 
 * Only records if full tracing mode (ti_cpu_task_tracing==1) is enabled and
 * subtask limit has not been reached.
 * 
 * @param subtask_name Name of the subtask
 * @param time Timestamp for the subtask (t_ns type)
 */
#define TI_APPEND(subtask_name,time) \
if(ti_cpu_task_tracing==1) { \
    if(ti_subtask_count < ti_max_subtasks) { \
        strcpy(ti_subtask_names[ti_subtask_count],subtask_name); \
        ti_times[ti_subtask_count] = time; \
        ti_subtask_count += 1; \
    } \
}

/**
 * @brief Records a subtask timing checkpoint with current timestamp.
 * 
 * @param subtask_name Name of the subtask/checkpoint
 */
#define TI_ADD(subtask_name) \
    TI_APPEND(subtask_name,Time::nowNs())

/**
 * @brief Logs task instrumentation data at task completion.
 * 
 * Reads PMU counters and logs task timing data. Supports two modes:
 * - Mode 1 (ti_cpu_task_tracing==1): Full tracing with all subtask timestamps
 * - Mode 2 (ti_cpu_task_tracing==2): Tracing mode with only end timestamp
 * 
 * @param sfn System Frame Number (unused, retrieved from slot_map)
 * @param slot Slot number (unused, retrieved from slot_map)
 * @param map Slot map ID (unused, retrieved from slot_map)
 * @param cpu CPU core ID (unused, retrieved earlier)
 */
#define TI_NVSLOGI(sfn,slot,map,cpu) \
if(ti_cpu_task_tracing != 0) { \
    worker->readEndCounters(); \
    worker->formatCounterMetrics(pmu_metrics_str,MAX_PMU_METRICS_CHARS); \
} \
if(ti_cpu_task_tracing==1) { \
    int offset=0; \
    for(int ii=0; ii<ti_subtask_count; ii++) { \
        offset += snprintf(&subtask_results[offset], MAX_NVSLOGI_CHARS-offset, "%s:%lu,", ti_subtask_names[ii], ti_times[ii].count()); \
    } \
    NVLOGI_FMT(TAG,"{{TI}} <{},{},{},{},{}> <{}> {}", \
    ti_task_name, ti_slot_map->getSlot3GPP().sfn_, ti_slot_map->getSlot3GPP().slot_, ti_slot_map->getId(), ti_cpu, \
    pmu_metrics_str, \
    subtask_results); \
} else if(ti_cpu_task_tracing==2) { \
    snprintf(&subtask_results[0], MAX_NVSLOGI_CHARS, "%s:%lu,", "e", Time::nowNs()); \
    NVLOGI_FMT(TAG,"{{mTI}} <{},{},{},{},{}> <{}> {}", \
    ti_task_name, ti_slot_map->getSlot3GPP().sfn_, ti_slot_map->getSlot3GPP().slot_, ti_slot_map->getId(), ti_cpu, \
    pmu_metrics_str, \
    subtask_results); \
}

/**
 * @brief Appends a list of subtasks from a ti_subtask_info structure.
 * 
 * Iterates through all recorded subtasks in the structure and appends them
 * to the current task instrumentation.
 * 
 * @param subtask_info The ti_subtask_info structure containing nested subtask data
 */
#define TI_APPEND_LIST(subtask_info) \
for(int ii=0;ii<subtask_info.count;++ii) {\
    TI_APPEND(subtask_info.tname[ii],subtask_info.time[ii]) \
}

#endif // #ifndef TASK_INSTRUMENTATION_H
