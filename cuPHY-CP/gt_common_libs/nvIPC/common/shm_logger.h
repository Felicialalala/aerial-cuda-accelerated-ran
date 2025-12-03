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

#ifndef _SHM_LOGGER_H_
#define _SHM_LOGGER_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
// #include <stdatomic.h>
#include <semaphore.h>

#include "nv_ipc.h"
#include "nv_ipc_shm.h"

#define SHMLOG_DEFAULT_NAME "shmlog"

#define SHMLOG_NAME_MAX_LEN 100

typedef enum
{
    SHMLOG_STATE_INIT   = 0,
    SHMLOG_STATE_OPENED = 1,
    SHMLOG_STATE_CLOSED = 2,
} shmlogger_state_t;

typedef struct
{
    uint64_t max_file_size;  // Max log length for one time log API call
    int64_t  shm_cache_size; // Shared memory size, see /dev/shm/. Value will be aligned to 2^n automatically
    int32_t  save_to_file;   // Whether to start a background thread for save SHM cache to disk file.
    int32_t  file_saving_core;    // CPU core ID for the background file saving if enabled. -1 means no core binding.
    int32_t  shm_caching_core;    // CPU core ID for the background copying to shared memory if enabled. Has to be bound
    int32_t  max_msg_size;
    int32_t  max_data_size;
} shmlogger_config_t;

typedef struct {
    const char* prefix;      // Original prefix parameter
    const char* type;        // Original type parameter
    const char* path;        // Original path parameter
    int fh_collect;          // fh pcap collect mode
    const char* output_filename; // Output filename used for fh collect
    // Add any future parameters here
} shmlog_collect_params_t;

// A record includes 4 parts: record_t header, MSG part, DATA part, buf_size
typedef struct
{
    struct timeval tv;
    int32_t        buf_size;
    int32_t        msg_id;
    int32_t        data_len;
    int32_t        flags;
    char           buf[0]; // Store MSG part and DATA part. DATA part can be empty.
    // Implicitly defined an "int32_t record.buf_size" after DATA part for validation and backword address. sizeof(record.buf_size) = 4.
    // Total size = sizeof(record_t) + msg_len + min(data_len, max_data_size) + 4.
} record_t;


typedef struct shmlogger_t shmlogger_t;

shmlogger_t* shmlogger_open(int primary, const char* name, shmlogger_config_t* cfg);
int          shmlogger_close(shmlogger_t* logger);
int          shmlogger_reset(shmlogger_t *logger);

void shmlogger_save_ipc_msg(shmlogger_t* logger, nv_ipc_msg_t* msg, int32_t flags, int32_t max_data_size);
void shmlogger_save_fh_buffer(shmlogger_t* logger, const char* buffer, int32_t size, int32_t flags, uint64_t timestamp);
void shmlogger_save_buffer(shmlogger_t* logger, const char* buffer, int32_t size, int32_t flags);
int shmlogger_collect(const char* prefix, const char* type, const char* path);
int shmlogger_collect_ex(const shmlog_collect_params_t* params);

unsigned long shmlogger_get_packet_count(shmlogger_t *logger);

static inline int get_record_size(record_t* record)
{
    return sizeof(record_t) + record->buf_size + sizeof(record->buf_size);
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _SHM_LOGGER_H_ */
