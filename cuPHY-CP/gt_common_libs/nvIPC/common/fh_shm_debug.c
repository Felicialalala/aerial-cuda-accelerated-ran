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

#define _GNU_SOURCE /* See feature_test_macros(7) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <pcap/pcap.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <sched.h>
#include "fh_shm_debug.h"
#include "nvlog.h"

#define TAG (NVLOG_TAG_BASE_FH_DRIVER + 25) //"FH.DEBUG"
#define LOG_SHM_NAME_LEN 200

#define PCAP_DATA_LINK_TYPE 1

static FILE*           pcapfile;
static pthread_mutex_t pcapmutex;

// Record header for each packet. See https://wiki.wireshark.org/Development/LibpcapFileFormat
typedef struct
{
    uint32_t ts_sec;   /* timestamp seconds */
    uint32_t ts_usec;  /* timestamp microseconds */
    uint32_t incl_len; /* number of octets of packet saved in file */
    uint32_t orig_len; /* actual length of packet */
} record_header_t;

// Write common headers ahead of eCPRI payload
static int pcap_write_common_headers(record_t* record, int32_t fapi_len)
{
    int         ret = 0;

    // PCAP header: 16B
    record_header_t record_hdr;
    record_hdr.ts_sec  = record->tv.tv_sec;
    record_hdr.ts_usec = record->tv.tv_usec;
    record_hdr.incl_len = record->buf_size;
    record_hdr.orig_len = record->buf_size;
    if(fwrite(&record_hdr, sizeof(record_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }
    return ret;
}

static int pcap_write_record(record_t* record, int max_data_size)
{
    if(pcap_write_common_headers(record, record->buf_size) < 0)
    {
        return -1;
    }

    // Write packet payload
    if(fwrite(record->buf, record->buf_size, 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGD(TAG, "%s: write OK. record->buf_size=%d", __func__, record->buf_size);
        return 0;
    }
    return 0;
}

static int pcap_write_file_header()
{
    struct pcap_file_header file_header;
    file_header.magic         = 0xA1B2C3D4;
    file_header.version_major = PCAP_VERSION_MAJOR;
    file_header.version_minor = PCAP_VERSION_MINOR;
    file_header.thiszone      = 0;          /* gmt to local correction */
    file_header.sigfigs       = 0;          /* accuracy of timestamps */
    file_header.snaplen       = 0x00040000; /* max length saved portion of each pkt */
    file_header.linktype      = PCAP_DATA_LINK_TYPE;
    NVLOGI(TAG, "%s: write pcap_write_file_header size=%lu", __func__, sizeof(struct pcap_file_header));

    if(fwrite(&file_header, sizeof(struct pcap_file_header), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: write pcap_write_file_header failed", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}

static int pcap_file_open(const char* filename)
{
    // Create or lookup the semaphore
    pthread_mutex_init(&pcapmutex, NULL);
    char path[LOG_SHM_NAME_LEN * 2];
    snprintf(path, LOG_SHM_NAME_LEN * 2, "%s", filename);

    // Open a temperate file to store the logs
    if((pcapfile = fopen(path, "w")) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: failed to open file %s", __func__, path);
        return -1;
    }
    pcap_write_file_header();
    NVLOGI(TAG, "%s: opened file %s for PCAP log", __func__, path);
    return 0;
}

#define MAX_SHMLOG_BUF_SIZE (3 * 1024 * 1024)

static int convert_pcap(FILE* record_file, FILE* pcap_file, long start, long end, int32_t max_msg_size, int32_t max_data_size)
{
    int       ret    = 0;
    record_t* record = malloc(MAX_SHMLOG_BUF_SIZE);
    if (record == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: malloc failed", __func__);
        return -1;
    }
    record->buf_size = 0;
    record->flags = 0;

    NVLOGI(TAG, "%s: start=0x%lX end=0x%lX", __func__, start, end);

    int32_t max_record_size = sizeof(record_t) + max_msg_size + max_data_size + 4;

    long pos_forward = start;
    while(pos_forward + sizeof(record_t) < end)
    {
        // Set file offset
        if(fseek(record_file, pos_forward, SEEK_SET) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fseek error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        // Read the record_t header
        if(fread(record, sizeof(record_t), 1, record_file) != 1)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fread header error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }
        NVLOGV(TAG, "%s: record: dir=%d msg_id=0x%X record->buf_size=%d", __func__, record->flags >> 16, record->flags & 0xFFFF, record->buf_size);

        if(pos_forward + record->buf_size + sizeof(record->buf_size) > end)
        {
            NVLOGI(TAG, "%s: The last record was overridden, skip pos_forward=0x%lX record->buf_size=0x%X", __func__, pos_forward, record->buf_size);
            break;
        }

        // Error check
        if(record->buf_size <= 0 || record->buf_size > max_record_size - sizeof(record_t) - 4)
        {
            NVLOGI(TAG, "%s: error record: pos_forward=0x%lX record->buf_size=0x%X", __func__, pos_forward, record->buf_size);
            ret = -1;
            break;
        }

        // Read the payload
        if(fread(record->buf, record->buf_size, 1, record_file) != 1)
        {
            NVLOGI(TAG, "%s: fread payload error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        int32_t buf_size = -1;
        if(fread(&buf_size, sizeof(record->buf_size), 1, record_file) != 1)
        {
            NVLOGI(TAG, "%s: fread size error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        // For debug: error check
        if(buf_size != record->buf_size)
        {
            NVLOGI(TAG, "%s: record file format error: pos_forward=0x%lX record->buf_size=%d buf_size=%d", __func__, pos_forward, record->buf_size, buf_size);
            ret = -1;
            break;
        }

        // Write to pcap file
        pcap_write_record(record, max_data_size);

        // Check and move file offset
        int record_size = get_record_size(record);
        if(pos_forward + record_size + sizeof(record_t) >= end)
        {
            break;
        }
        else
        {
            pos_forward += record_size;
        }
    }

    if (ret < 0)
    {
        if (end - pos_forward <= max_record_size * 2)
        {
            // Ignore the last 2 records parsing failure
            ret = 0;
            NVLOGI(TAG, "%s: The last %lu bytes was not integrative. pos_forward=0x%lX record->buf_size=0x%X msg_id=0x%02X",
                    __func__, end - pos_forward, pos_forward, record->buf_size, record->flags & 0xFFFF);
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: convert error: remain_size=%lu pos_forward=0x%lX record->buf_size=0x%X msg_id=0x%02X",
                    __func__, end - pos_forward, pos_forward, record->buf_size, record->flags & 0xFFFF);
        }
    }

    free(record);
    fflush(pcapfile);
    NVLOGI(TAG, "%s: ret=%d start=0x%lX end=0x%lX converted_pos=0x%lX - %ld", __func__, ret, start, end, pos_forward, pos_forward);
    return ret;
}

int64_t fh_convert_pcap(FILE* record_file, char* pcap_filepath, long shm_cache_size, int32_t max_msg_size, int32_t max_data_size, long total_size, uint64_t break_offset)
{
    if(record_file == NULL || fseek(record_file, 0, SEEK_END) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: record_file error", __func__);
        return -1;
    }

    int64_t file_size = ftell(record_file);
    if (file_size < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: record_file size error", __func__);
        return -1;
    }

    NVLOGI(TAG, "%s: shm_cache_size=0x%lX file_size=0x%lX total_size=0x%lX", __func__, shm_cache_size, file_size, total_size);

    int ret = 0;
    pcap_file_open(pcap_filepath);

    // Log rotation enabled
    // Convert the first SHM block
    long pcap_end = break_offset == 0 ? file_size : shm_cache_size;
    ret = convert_pcap(record_file, pcapfile, 0, pcap_end, max_msg_size, max_data_size);
    if(ret != 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: convert first SHM block failed shm_cache_size=0x%lX file_size=0x%lX", __func__, shm_cache_size, file_size);
    }

    int64_t pos_break = (break_offset & ((shm_cache_size -1) >> 1)) + shm_cache_size;

    NVLOGI(TAG, "%s: converted first block size=0x%lX=%ld MB file_size=0x%lX=%ld MB pos_break=0x%lX", __func__,
            shm_cache_size, shm_cache_size >> 20, file_size, file_size >> 20, pos_break);

    if (break_offset != 0)
    {
        // Some logs may have been overwritten, find the earliest record which hasn't been overwritten
        int64_t pos_backward = file_size;
        record_t record;
    #if 0 // Below code are for debug, do not delete
        while(pos_backward > shm_cache_size)
        {
            if(fseek(record_file, pos_backward - sizeof(record.buf_size), SEEK_SET) < 0)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fseek error: pos_backward=0x%lX err=%d - %s", __func__, pos_backward, errno, strerror(errno));
                ret = -1;
                break;
            }

            // Read the record size
            if(fread(&record.buf_size, sizeof(record.buf_size), 1, record_file) != 1)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fread error: pos_backward=0x%lX err=%d - %s", __func__, pos_backward, errno, strerror(errno));
                ret = -1;
                break;
            }

            NVLOGD(TAG, "%s: buf_size=%ld - 0x%X pos_backward=0x%lX", __func__, record.buf_size, record.buf_size, pos_backward);

            if(record.buf_size <= 0)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: error buffer size: pos_backward=0x%lX=%u MB", __func__, pos_backward, pos_backward / 1024/1024);
                ret = -1;
                break;
            }

            // Move backward to previous record
            int record_size = get_record_size(&record);
            if(pos_backward - record_size < shm_cache_size + sizeof(record.buf_size))
            {
                break;
            }
            else
            {
                pos_backward -= record_size;
            }
        }

    #else
        pos_backward = pos_break;
    #endif

        NVLOGI(TAG, "%s: found shm_cache_size=%ld pos_backward=0x%lX break_offset=0x%lX file_size=%ld",
                __func__, shm_cache_size, pos_backward, break_offset, file_size);

        if (pos_backward != pos_break)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: break_offset error: shm_cache_size=%ld pos_backward=0x%lX break_offset=0x%lX file_size=%ld",
                    __func__, shm_cache_size, pos_backward, break_offset, file_size);
        }

        if(ret == 0)
        {
            ret = convert_pcap(record_file, pcapfile, pos_backward, file_size, max_msg_size, max_data_size);
        }
    }
    int64_t pcap_size = ftell(pcapfile);
    fclose(pcapfile);
    return ret == 0 ? pcap_size : -1;
}
