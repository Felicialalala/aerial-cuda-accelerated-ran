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

#ifndef _NV_IPC_DEBUG_H_
#define _NV_IPC_DEBUG_H_

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#if defined(__cplusplus)
extern "C" {
#endif

#include <time.h>
#include <stdatomic.h>

#include "nv_ipc.h"
#include "nv_ipc_shm.h"
#include "nv_ipc_ring.h"
#include "stat_log.h"
#include "shm_logger.h"

#define FAPI_DATA_SIZE_LIMIT (8096)

#define NV_IPC_MSG_ID_MAX 128

#define DEBUG_HIGH_RESOLUTION_TIME
#ifdef DEBUG_HIGH_RESOLUTION_TIME
typedef struct timespec timestamp_t;
#else
typedef struct timeval timestamp_t;
#endif

/** NVIPC command identifiers for PCAP dynamic operation commands */
typedef enum
{
    IPC_CMD_PCAP_ENABLE = 0,            //!< Enable PCAP capture
    IPC_CMD_PCAP_CONFIG_MSG_FILTER,     //!< Configure message filter
    IPC_CMD_PCAP_CONFIG_CELL_FILTER,    //!< Configure cell filter
    IPC_CMD_PCAP_CLEAN,                 //!< Clean PCAP data
    IPC_CMD_PCAP_MAX                    //!< Maximum command count
} nvipc_cmd_id_t;

/** NVIPC command structure for PCAP dynamic operation commands */
typedef struct
{
    int32_t cmd_id;    //!< Command identifier
    int32_t param;     //!< Command parameter (reserved)
} nvipc_cmd_t;

/** Message direction enumeration */
typedef enum
{
    NVIPC_SECONDARY_TO_PRIMARY = 0,  //!< Secondary to primary direction
    NVIPC_PRIMARY_TO_SECONDARY = 1,  //!< Primary to secondary direction
    NVIPC_PRIMARY_LOOPBACK = 2,      //!< Primary loopback
    NVIPC_SECONDARY_LOOPBACK = 3,    //!< Secondary loopback
} nvipc_msg_dir_t;

/** Packet information structure */
typedef struct
{
    int32_t msg_id;              //!< Message ID
    int32_t cell_id;             //!< Cell ID
    int32_t msg_len;             //!< Message length
    int32_t data_len;            //!< Data length
    int32_t data_index;          //!< Data buffer index
    int32_t data_pool;           //!< Data pool ID
    nvipc_msg_dir_t direction;   //!< Message direction (for PCAP logging)
    atomic_uint buf_ref_count;   //!< Buffer reference count
} packet_info_t;

/** Message timing information for debugging*/
typedef struct
{
    timestamp_t ts_alloc;   //!< Timestamp after allocation
    timestamp_t ts_send;    //!< Timestamp before send
    timestamp_t ts_recv;    //!< Timestamp after receive
    timestamp_t ts_free;    //!< Timestamp before free
    timestamp_t ts_fw_deq;  //!< Timestamp after forward dequeue
    timestamp_t ts_fw_free; //!< Timestamp before forward free
} msg_timing_t;

/** Synchronization timing information for debugging*/
typedef struct
{
    timestamp_t ts_post;  //!< Timestamp before post
    timestamp_t ts_wait;  //!< Timestamp after wait finishes
} sync_timing_t;

/** Debug shared memory data structure */
typedef struct
{
    atomic_ulong post_counter_m2s;   //!< Primary to secondary semaphore post counter
    atomic_ulong wait_counter_m2s;   //!< Primary to secondary semaphore wait counter
    atomic_ulong post_counter_s2m;   //!< Secondary to primary semaphore post counter
    atomic_ulong wait_counter_s2m;   //!< Secondary to primary semaphore wait counter
    atomic_ulong ipc_dumping;        //!< IPC dump in progress flag
    nv_ipc_config_t primary_configs; //!< Primary process configuration
    uint8_t debug_buffers[];         //!< Variable-length debug buffer array
} debug_shm_data_t;

/** IPC debug context structure */
typedef struct nv_ipc_debug_t nv_ipc_debug_t;
struct nv_ipc_debug_t
{
    int  primary;                         //!< Primary process flag
    int  msg_pool_len;                    //!< Message pool length
    char prefix[NV_NAME_MAX_LEN];         //!< Instance name prefix
    nv_ipc_transport_t transport;         //!< Transport type
    nv_ipc_debug_config_t debug_configs;  //!< Debug configuration
    msg_timing_t *msg_timing;             //!< Message timing array
    sync_timing_t *sync_timing_m2s;       //!< Primary to secondary semaphore sync timing
    sync_timing_t *sync_timing_s2m;       //!< Secondary to primary semaphore sync timing
    nv_ipc_t* ipc;                        //!< IPC instance pointer
    sem_t *debug_sem;                     //!< Debug thread notification semaphore pointer
    pthread_t debug_thread_id;            //!< Debug background thread ID
    nv_ipc_ring_t *cmd_ring;              //!< Command ring for dynamic commands
    nv_ipc_shm_t*     shmpool;            //!< Shared memory pool
    debug_shm_data_t* shm_data;           //!< Shared memory buffer pointer
    shmlogger_t*      shmlogger;          //!< Shared memory logger
    nv_ipc_debug_config_t* primary_debug_configs;  //!< Primary debug config pointer
    packet_info_t* packet_infos;          //!< Packet info array
    atomic_uint* p_forward_started;       //!< Forward started flag (hold to improve performance)
    uint8_t* cell_filters;                //!< Cell filter array (hold to improve performance)
    uint8_t* msg_filters;                 //!< Message filter array (hold to improve performance)
    stat_log_t* stat_msg_build;           //!< Message build statistics logger
    stat_log_t* stat_msg_transport;       //!< Message transport statistics logger
    stat_log_t* stat_msg_handle;          //!< Message handle statistics logger
    stat_log_t* stat_msg_total;           //!< Total message statistics logger
    stat_log_t* stat_wait_delay;          //!< Wait delay statistics logger
    stat_log_t* stat_post_interval;       //!< Post interval statistics logger
    int (*alloc_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index);   //!< Allocation hook
    int (*free_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index);    //!< Free hook
    int (*send_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index);    //!< Send hook
    int (*recv_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index);    //!< Receive hook
    int (*post_hook)(nv_ipc_debug_t* ipc_debug);   //!< Post hook
    int (*wait_hook)(nv_ipc_debug_t* ipc_debug);   //!< Wait hook
    int (*fw_deq_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index);  //!< Forward dequeue hook
    int (*fw_free_hook)(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index); //!< Forward free hook
    int (*close)(nv_ipc_debug_t* ipc_debug); //!< Close
};

/**
 * Get integer value from environment variable
 *
 * @param[in] name Environment variable name
 * @param[in] def Default value if not found
 * @return Environment variable value or default
 */
long get_env_long(const char* name, long def);

/**
 * Dump an IPC message for debugging
 *
 * @param[in] ipc_debug Debug context
 * @param[in] msg Message to dump
 * @param[in] buf_index Buffer index
 * @param[in] info Additional info string
 * @return 0 on success, -1 on failure
 */
int nv_ipc_dump_msg(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index, const char* info);

/**
 * Dump SHM IPC state
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int shm_ipc_dump(nv_ipc_t* ipc);

/**
 * Dump DOCA IPC state
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int doca_ipc_dump(nv_ipc_t* ipc);

/**
 * Get send timestamp for a sent message
 *
 * @param[in] ipc_debug Debug context
 * @param[in] buf_index Buffer index
 * @return Timestamp in nanoseconds, -1 on failure
 */
int64_t nv_ipc_get_buffer_ts_send(nv_ipc_debug_t* ipc_debug, int32_t buf_index);

/**
 * Get packet info array for all messages
 *
 * @param[in] ipc IPC instance
 * @return Pointer to packet info array, NULL on failure
 */
packet_info_t* nv_ipc_get_packet_infos(nv_ipc_t *ipc);

/**
 * Get packet info for a message
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message
 * @return Pointer to packet info, NULL on failure
 */
packet_info_t* nv_ipc_get_packet_info(nv_ipc_t *ipc, nv_ipc_msg_t *msg);

/**
 * Get message direction for a message
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message
 * @return Message direction
 */
nvipc_msg_dir_t nv_ipc_get_msg_direction(nv_ipc_t *ipc, nv_ipc_msg_t *msg);

/**
 * Select IPC instance by prefix
 *
 * @param[in] prefix Instance name prefix
 * @return 0 on success, -1 on failure
 */
int nv_ipc_select(const char* prefix);

/**
 * Get free buffer count in a memory pool
 *
 * @param[in] pool_id Memory pool identifier
 * @return Free buffer count, -1 on failure
 */
int nv_ipc_get_mempool_free_count(nv_ipc_mempool_id_t pool_id);

/**
 * Convert saved message records to pcap file
 *
 * @param[in] record_file Input record file
 * @param[in] pcap_filepath Output pcap file path
 * @param[in] shm_cache_size SHM cache size
 * @param[in] max_msg_size Maximum message size
 * @param[in] max_data_size Maximum data size
 * @param[in] total_size Total size
 * @param[in] break_offset Break offset
 * @return Bytes processed on success, -1 on failure
 */
int64_t nv_ipc_convert_pcap(FILE* record_file, char* pcap_filepath, long shm_cache_size, int32_t max_msg_size, int32_t max_data_size, long total_size, uint64_t break_offset);

/**
 * Open debug context
 *
 * @param[in] ipc IPC instance
 * @param[in] cfg IPC configuration
 * @return Pointer to debug context on success, NULL on failure
 */
nv_ipc_debug_t* nv_ipc_debug_open(nv_ipc_t* ipc, nv_ipc_config_t* cfg);

/**
 * Get maximum message size for PCAP
 *
 * @return Maximum message size
 */
int get_pcap_max_msg_size();

/**
 * Get maximum data size for PCAP
 *
 * @return Maximum data size
 */
int get_pcap_max_data_size();

/**
 * Check message validity for debugging
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message to verify
 * @param[in] info Additional info string for error messages
 */
void verify_msg(nv_ipc_t *ipc, nv_ipc_msg_t *msg, const char *info);

/**
 * Get message buffer index for a message
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message
 * @return Buffer index, -1 on failure
 */
int get_msg_index(nv_ipc_t *ipc, nv_ipc_msg_t *msg);

/**
 * Start PCAP capture
 *
 * @param[in] shmpool Shared memory pool
 * @param[in] count Number of messages to capture (0 = infinite)
 * @return 0 on success, -1 on failure
 */
int nvipc_pcap_start(nv_ipc_shm_t *shmpool, uint32_t count);

/**
 * Stop PCAP capture
 *
 * @param[in] shmpool Shared memory pool
 * @return 0 on success, -1 on failure
 */
int nvipc_pcap_stop(nv_ipc_shm_t *shmpool);

/**
 * Dump PCAP statistics
 *
 * @param[in] shmpool Shared memory pool
 * @return 0 on success, -1 on failure
 */
int nvipc_pcap_dump(nv_ipc_shm_t *shmpool);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_DEBUG_H_ */
