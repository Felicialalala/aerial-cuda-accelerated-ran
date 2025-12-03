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

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/epoll.h>

#include "nv_ipc_debug.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_forward.h"
#include "nv_ipc_ring.h"

#define TAG (NVLOG_TAG_BASE_NVIPC) // "NVIPC"

nv_ipc_transport_t ipc_transport = NV_IPC_TRANSPORT_SHM;
nv_ipc_module_t    module_type   = NV_IPC_MODULE_SECONDARY;
nv_ipc_t*          ipc           = NULL;

// The CUDA device ID. Can set to -1 to fall back to CPU memory IPC
int test_cuda_device_id = -1;

#define NV_NAME_SUFFIX_MAX_LEN 16

#define NV_PATH_MAX_LEN 1024
static char dest_path[NV_PATH_MAX_LEN + NVLOG_NAME_MAX_LEN] = ".";

static char logger_name[NVLOG_NAME_MAX_LEN] = "pcap";

static char nvipc_prefix[NVLOG_NAME_MAX_LEN] = "nvipc";

typedef enum
{
    PCAP_CMD_START = 0,
    PCAP_CMD_STOP,
    PCAP_CMD_CONFIG,
    PCAP_CMD_COLLECT,
    PCAP_CMD_CLEAN,
    PCAP_CMD_DUMP,
} pcap_task_t;

void print_usage()
{
    printf("Usage: sudo pcap <start|stop|config|clean|dump> [-p <prefix>] [OPTIONS]\n");
    printf("\n");
    printf("    -p, --prefix         NVIPC instance prefix. Default is 'nvipc' if not provided\n");
    printf("    -m, --msg-filter     PCAP msg_filter. Example: -m \"0x81,0x82,0x85,0x86\"\n");
    printf("    -c, --cell-filter    PCAP cell_filter. Example: -c \"0,1,3,5\"\n");
}

int parse_integer_value(char *nptr, char **endptr, int64_t *value)
{
    if (nptr == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: null parameter", __FUNCTION__);
        return -1;
    }
    NVLOGI(TAG, "%s: nptr=%s", __FUNCTION__, nptr);

    if (strncmp(nptr, "0b", 2) == 0 || strncmp(nptr, "0B", 2) == 0)
    {
        *value = strtoll(nptr + 2, endptr, 2); // Binary
    }
    else
    {
        *value = strtoll(nptr, endptr, 0); // Octal, Decimal, Hex
    }

    // if (*endptr == NULL || **endptr != '\0')
    if (*endptr == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid integer parameter: %s", __FUNCTION__, nptr);
        return -1;
    }

    return 0;
}

int parse_filter(char* hex_string, uint8_t vals[], int max_count) {
    char *end_ptr;
    int64_t value;

    if (strncmp(hex_string, "all", strlen("all")) == 0)
    {
        // Enable all
        for (int i = 0; i < max_count; i++)
        {
            vals[i] = 1;
        }
        return 0;
    }

    while (*hex_string != '\0')
    {
        NVLOGI(TAG, "Parse integer values: [%s]\n", hex_string);
        if (parse_integer_value(hex_string, &end_ptr, &value) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid filter format: %s", __FUNCTION__, hex_string);
            return -1;
        }

        if (value >= max_count)
        {
            NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid filter value: %s", __FUNCTION__, hex_string);
        }

        vals[value] = 1;
        if (*end_ptr == '\0')
        {
            NVLOGI(TAG, "%s: parse ended: %s", __FUNCTION__, hex_string);
            break;
        }

        if (end_ptr != hex_string)
        {
            // printf("%ld\n", value);
            hex_string = end_ptr;
        }

        // Move to the next number
        while (*hex_string && (*hex_string == ',' || *hex_string == ' '))
        {
            hex_string++;
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    int ret = 0;
    int uid = getuid();

    // nvlog_c_init("/var/log/aerial/pcap.log");
    NVLOGI(TAG, "%s: nvlog [%s] opened. uid=%d", __func__, logger_name, uid);

    NVLOGI(TAG, "%s: argc=%d argv[0]=[%s]", __func__, argc, argv[0]);
    if(argc < 2)
    {
        print_usage();
        return 0;
    }

    uint8_t msg_filter[NVIPC_MAX_MSG_ID];
    uint8_t cell_filter[NVIPC_MAX_CELL_ID];
    memset(msg_filter, 0, NVIPC_MAX_MSG_ID);
    memset(cell_filter, 0, NVIPC_MAX_CELL_ID);

    int pcap_cmd = -1;
    int config_msg_filter = 0;
    int config_cell_filter = 0;

    for (int i = 1; i < argc; i++)
    {
        NVLOGI(TAG, "%s: argv[%d-%d]=[%s]", __func__, argc, i, argv[i]);
        if (strncmp(argv[i], "-h", 2) == 0 || strncmp(argv[i], "--help", strlen("--help")) == 0 || strncmp(argv[i], "help", strlen("help")) == 0)
        {
            print_usage();
            return 0;
        }
        else if (strncmp(argv[i], "-p", 2) == 0 || strncmp(argv[i], "--prefix", strlen("--prefix")) == 0)
        {
            if (i >= argc - 1)
            {
                NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: option [%s] requires an prefix argument", __func__, argv[i]);
                return -1;
            }
            i++;
            nvlog_safe_strncpy(nvipc_prefix, argv[i], NV_NAME_MAX_LEN);
        }
        else if (strncmp(argv[i], "-m", 2) == 0 || strncmp(argv[i], "--msg-filter", strlen("--msg-filter")) == 0)
        {
            if (i >= argc - 1)
            {
                NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: option [%s] requires an msg_filter argument", __func__, argv[i]);
                return -1;
            }
            i++;
            if (parse_filter(argv[i], msg_filter, NVIPC_MAX_MSG_ID) < 0)
            {
                return -1;
            }
            config_msg_filter = 1;
        }
        else if (strncmp(argv[i], "-c", 2) == 0 || strncmp(argv[i], "--cell-filter", strlen("--cell-filter")) == 0)
        {
            if (i >= argc - 1)
            {
                NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: option [%s] requires an cell_filter argument", __func__, argv[i]);
                return -1;
            }
            i++;
            if (parse_filter(argv[i], cell_filter, NVIPC_MAX_CELL_ID) < 0)
            {
                return -1;
            }
            config_cell_filter = 1;
        }
        else if (strncmp(argv[i], "start", strlen("start")) == 0)
        {
            pcap_cmd = PCAP_CMD_START;
        }
        else if (strncmp(argv[i], "stop", strlen("stop")) == 0)
        {
            pcap_cmd = PCAP_CMD_STOP;
        }
        else if (strncmp(argv[i], "collect", strlen("collect")) == 0)
        {
            pcap_cmd = PCAP_CMD_COLLECT;
        }
        else if (strncmp(argv[i], "clean", strlen("clean")) == 0)
        {
            pcap_cmd = PCAP_CMD_CLEAN;
        }
        else if (strncmp(argv[i], "dump", strlen("dump")) == 0)
        {
            pcap_cmd = PCAP_CMD_DUMP;
        }
        else if(strncmp(argv[i], "config", strlen("config")) == 0)
        {
            pcap_cmd = PCAP_CMD_CONFIG;
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: invalid parameter: argv[%d]=%s", __func__, i, argv[i]);
        }
    }

    if (pcap_cmd < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: no pcap_cmd", __func__);
        return -1;
    }

    NVLOGI(TAG, "%s: pcap_cmd=%d", __func__, pcap_cmd);

    // Get nvipc configuration
    nv_ipc_config_t config;
    if (nv_ipc_lookup_config(&config, nvipc_prefix, NV_IPC_MODULE_SECONDARY) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NVIPC instance [%s] not exist", __func__, nvipc_prefix);
        return -1;
    }

    if (config.ipc_transport != NV_IPC_TRANSPORT_SHM)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: unsupported transport type: %d", __func__, ipc_transport);
        return -1;
    }
    config.module_type = NV_IPC_MODULE_IPC_DUMP;

    // Open "<prefix>_debug" shared memory pool
    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    nvlog_safe_strncpy(name, nvipc_prefix, NV_NAME_MAX_LEN);
    strncat(name, "_debug", NV_NAME_SUFFIX_MAX_LEN);
    nv_ipc_shm_t *debug_shmpool = nv_ipc_shm_open(0, name, 0);
    if(debug_shmpool == NULL)
    {
        ret = -1;
        goto app_exit;
    }
    debug_shm_data_t* debug_shm_data = (debug_shm_data_t*)debug_shmpool->get_mapped_addr(debug_shmpool);
    nv_ipc_debug_config_t* nvipc_debug_config = &debug_shm_data->primary_configs.debug_configs;

    // Lookup the shared memory pool of the forwarder
    nvlog_safe_strncpy(name, nvipc_prefix, NV_NAME_MAX_LEN);
    strncat(name, "_fw", NV_NAME_SUFFIX_MAX_LEN);
    nv_ipc_shm_t *fw_shmpool = nv_ipc_shm_open(0, name, 0);
    if (fw_shmpool == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NVIPC FW pool was not found", __func__);
        ret = -1;
        goto app_exit;
    }

    // Open the dynamic nvipc command ring
    nvlog_safe_strncpy(name, nvipc_prefix, NV_NAME_MAX_LEN);
    strncat(name, "_cmd_ring", NV_NAME_SUFFIX_MAX_LEN);
    nv_ipc_ring_t *cmd_ring = nv_ipc_ring_open(RING_TYPE_SHM_SECONDARY, name, 64, sizeof(nvipc_cmd_t));
    if (cmd_ring == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "nv_ipc_ring_open failed: name=%s", name);
        ret = -1;
        goto app_exit;
    }

    // Open the debug semaphore
    nvlog_safe_strncpy(name, nvipc_prefix, NV_NAME_MAX_LEN);
    strncat(name, "_dbg_sem", NV_NAME_SUFFIX_MAX_LEN);
    sem_t* debug_sem;
    if ((debug_sem = sem_open(name, O_CREAT, 0600, 0)) == SEM_FAILED)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "sem_open failed: name=%s", name);
        ret = -1;
        goto app_exit;
    }

    // Config msg_filter and cell_filter first if provided in "pcap start" or "pcap config"
    nvipc_cmd_t nvipc_cmd;
    if (pcap_cmd == PCAP_CMD_START || pcap_cmd == PCAP_CMD_CONFIG)
    {
        if (config_msg_filter)
        {
            NVLOGC(TAG, "[%s]: config msg_filter", nvipc_prefix);
            memcpy(nvipc_debug_config->msg_filter, msg_filter, NVIPC_MAX_MSG_ID * sizeof(uint8_t));
            nvipc_cmd.cmd_id = IPC_CMD_PCAP_CONFIG_MSG_FILTER;
            cmd_ring->enqueue(cmd_ring, &nvipc_cmd);
            sem_post(debug_sem);
        }
        if (config_cell_filter)
        {
            NVLOGC(TAG, "[%s]: config cell_filter", nvipc_prefix);
            memcpy(nvipc_debug_config->cell_filter, cell_filter, NVIPC_MAX_CELL_ID * sizeof(uint8_t));
            nvipc_cmd.cmd_id = IPC_CMD_PCAP_CONFIG_CELL_FILTER;
            cmd_ring->enqueue(cmd_ring, &nvipc_cmd);
            sem_post(debug_sem);
        }
    }

    // PCAP commands
    switch (pcap_cmd)
    {
    case PCAP_CMD_START:
        if (nvipc_debug_config->pcap_enable == 0)
        {
            // Send a pcap_cmd to L1 to initiate the PCAP module
            NVLOGC(TAG, "[%s]: initiate pcap logger and start capturing", nvipc_prefix);
            nvipc_cmd.cmd_id = IPC_CMD_PCAP_ENABLE;
            cmd_ring->enqueue(cmd_ring, &nvipc_cmd);
            sem_post(debug_sem);
        }
        else
        {
            // L1 PCAP module had been enabled at initial, just need turn on the atomic flag "forward_started"
            NVLOGC(TAG, "[%s]: start pcap capturing", nvipc_prefix);
            nvipc_pcap_start(fw_shmpool, 0);
        }
        break;
    case PCAP_CMD_STOP:
        // Turn off the atomic flag "forward_started"
        NVLOGC(TAG, "[%s]: stop pcap capturing", nvipc_prefix);
        nvipc_pcap_stop(fw_shmpool);
        break;
    case PCAP_CMD_CONFIG:
        // Processed above, skip
        break;
    case PCAP_CMD_DUMP:
        nv_ipc_dump_config(&debug_shm_data->primary_configs);
        nvipc_pcap_dump(fw_shmpool);
        break;
    case PCAP_CMD_COLLECT:
        NVLOGC(TAG, "Collect pcap logs: %s name=%s dest_path=%s\n", argv[0], logger_name, dest_path);
        shmlogger_collect(nvipc_prefix, "pcap", dest_path);
        break;
    case PCAP_CMD_CLEAN:
        NVLOGC(TAG, "[%s]: clean cached logs", nvipc_prefix);
        nvipc_cmd.cmd_id = IPC_CMD_PCAP_CLEAN;
        cmd_ring->enqueue(cmd_ring, &nvipc_cmd);
        sem_post(debug_sem);
        break;
    default:
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: unsupported pcap_cmd: %d", __func__, pcap_cmd);
        break;
    }

app_exit:

    if (debug_shmpool != NULL)
    {
        debug_shmpool->close(debug_shmpool);
    }

    if (fw_shmpool != NULL)
    {
        fw_shmpool->close(fw_shmpool);
    }

    if (cmd_ring != NULL)
    {
        cmd_ring->close(cmd_ring);
    }

    if (debug_sem != NULL)
    {
        sem_close(debug_sem);
    }

    nvlog_c_close();

    return ret;
}
