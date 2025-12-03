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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stddef.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sched.h>

#include "fd_share.h"
#include "nv_ipc_efd.h"
#include "nv_ipc_utils.h"
#include "nv_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 11) //"NVIPC.EFD"

#define ENABLE_EFD_SEMAPHORE_FLAG 1

// #define CONFIG_NON_BLOCKING_INIT 0

#define NV_NAME_SUFFIX_MAX_LEN 16
#define UNIX_SOCK_PATH "/dev/shm/"
#define UNIX_SOCK_NAME_SUFFIX_SERVER "_un_sock_server"
#define UNIX_SOCK_NAME_SUFFIX_CLIENT "_un_sock_client"

typedef struct
{
    // Primary create Unix-domain server socket, client connect to it.
    int primary;

    /**
     * Event FD for TX - main thread reads for write(), background thread writes on (re)connection.
     * Atomics prevent data races, ensure visibility, and provide memory ordering.
     * Without atomics: main thread may write() to wrong/stale fd.
     */
    int efd_tx;

    /**
     * Event FD for RX - created before thread starts, then read-only by both threads.
     * No atomics needed: write happens-before pthread_create(), concurrent reads are safe.
     */
    int efd_rx;

    pthread_t thread_id;

    char prefix[NV_EFD_NAME_MAX_LEN];

    int (*reconnect_callback)(void *args);
    void *reconnect_callback_args;

    /**
     * Shutdown flag - accessed concurrently by main thread (writes 1) and background thread (reads).
     * Atomics prevent data races, ensure visibility, and provide memory ordering.
     * Without atomics: background thread may miss shutdown signal, preventing clean exit.
     */
    int shutdown_requested;

    /**
     * Server socket fd - main thread reads for shutdown(), background thread writes fd/-1.
     * Atomics prevent data races, ensure visibility, and provide memory ordering.
     * Without atomics: main thread may call shutdown() on wrong/invalid fd during cleanup.
     */
    int server_socket_fd;

} priv_data_t;

typedef struct
{
    priv_data_t*       priv_data;
    struct sockaddr_un server_addr;
    struct sockaddr_un client_addr;
} thread_args_t;

static inline priv_data_t* get_private_data(nv_ipc_efd_t* ipc_efd)
{
    return (priv_data_t*)((int8_t*)ipc_efd + sizeof(nv_ipc_efd_t));
}

/**
 * Shutdown thread and wait for it to finish
 *
 * @param[in] priv_data Private data structure
 * @return 0 on success, -1 on error
 */
static int shutdown_and_join_thread(priv_data_t* priv_data)
{
    int ret = 0;

    // Signal thread to exit (atomic write ensures visibility to background thread)
    __atomic_store_n(&priv_data->shutdown_requested, 1, __ATOMIC_RELEASE);

    // Shutdown server socket to interrupt blocking accept() call
    int server_socket_local = __atomic_load_n(&priv_data->server_socket_fd, __ATOMIC_ACQUIRE);
    if(server_socket_local >= 0)
    {
        shutdown(server_socket_local, SHUT_RDWR);
    }

    // Unlink Unix socket files
    char server_sock_path[NV_UNIX_SOCKET_PATH_MAX_LEN + NV_EFD_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    snprintf(server_sock_path, sizeof(server_sock_path), "%s%s%s", 
             UNIX_SOCK_PATH, priv_data->prefix, UNIX_SOCK_NAME_SUFFIX_SERVER);
    unlink(server_sock_path);

    char client_sock_path[NV_UNIX_SOCKET_PATH_MAX_LEN + NV_EFD_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    snprintf(client_sock_path, sizeof(client_sock_path), "%s%s%s",
             UNIX_SOCK_PATH, priv_data->prefix, UNIX_SOCK_NAME_SUFFIX_CLIENT);
    unlink(client_sock_path);

    // Wait for thread to complete
    void* thread_ret = NULL;
    int join_ret = pthread_join(priv_data->thread_id, &thread_ret);
    if(join_ret != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: pthread_join failed with error %d", __func__, join_ret);
        ret = -1;
    }
    else
    {
        NVLOGC(TAG, "%s: thread joined successfully", __func__);
    }

    return ret;
}

static int ipc_efd_close(nv_ipc_efd_t* ipc_efd)
{
    if(ipc_efd == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc_efd);
    int          ret       = 0;

    // Signal thread to shutdown and wait for it to finish before closing resources
    // This prevents use-after-free when thread accesses priv_data after memory is freed
    if(priv_data->thread_id != 0)
    {
        ret = shutdown_and_join_thread(priv_data);
    }

    int efd_tx_local = __atomic_load_n(&priv_data->efd_tx, __ATOMIC_ACQUIRE);
    
    if(efd_tx_local >= 0 && close(efd_tx_local) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: close efd_tx failed", __func__);
        ret = -1;
    }

    if(priv_data->efd_rx >= 0 && close(priv_data->efd_rx) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: close efd_rx failed", __func__);
        ret = -1;
    }
    free(ipc_efd);

    if(ret == 0)
    {
        NVLOGI(TAG, "%s: OK", __func__);
    }
    return ret;
}

static int ipc_efd_get_rx_fd(nv_ipc_efd_t* ipc_efd)
{
    if(ipc_efd == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc_efd);
    return priv_data->efd_rx;
}

static int ipc_efd_tx_write(nv_ipc_efd_t* ipc_efd, int value)
{
    if(ipc_efd == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc_efd);
    uint64_t     efd_value = value;

    int efd_tx_local = __atomic_load_n(&priv_data->efd_tx, __ATOMIC_ACQUIRE);
    
    if(efd_tx_local < 0)
    {
        NVLOGW(TAG, "%s: remote party is not connected", __func__);
        return -1;
    }

    NVLOGV(TAG, "%s: count=%d", __func__, value);

    if(ENABLE_EFD_SEMAPHORE_FLAG)
    {
        efd_value = 1;
    }

    ssize_t size = write(efd_tx_local, &efd_value, sizeof(uint64_t));
    
    if(size < 0)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

static int ipc_efd_rx_read(nv_ipc_efd_t* ipc_efd)
{
    if(ipc_efd == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: instance not exist", __func__);
        return -1;
    }

    priv_data_t* priv_data = get_private_data(ipc_efd);
    uint64_t     efd_value;

    ssize_t size  = read(priv_data->efd_rx, &efd_value, sizeof(uint64_t));
    int     value = (int)efd_value;
    NVLOGV(TAG, "%s: count=%d", __func__, value);

    if(size < 0)
    {
        return -1;
    }
    else
    {
        return value;
    }
}

static void unix_socket_server(thread_args_t* args)
{
    priv_data_t* priv_data = args->priv_data;

    int server_socket, client_socket;
    if((server_socket = unix_sock_create(&args->server_addr)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Create server socket failed: %d", server_socket);
        return;
    }

    // Store server socket so it can be shutdown during cleanup
    __atomic_store_n(&priv_data->server_socket_fd, server_socket, __ATOMIC_RELEASE);

    uint32_t is_reconnect = 0;
    while (!__atomic_load_n(&priv_data->shutdown_requested, __ATOMIC_ACQUIRE))
    {
        NVLOGC(TAG, "[%s] Listening for nvipc client to connect ...", priv_data->prefix);
        if ((client_socket = unix_sock_listen_and_accept(server_socket, &args->client_addr)) < 0)
        {
            NVLOGC(TAG, "Socket accept failed or interrupted, shutting down");
            close(server_socket);
            __atomic_store_n(&priv_data->server_socket_fd, -1, __ATOMIC_RELEASE);
            return;
        }

        NVLOGC(TAG, "[%s] nvipc unix socket server connected", priv_data->prefix);

        // Skip reset callback for the first connection
        if (is_reconnect)
        {
            if (priv_data->reconnect_callback != NULL)
            {
                int ret = priv_data->reconnect_callback(priv_data->reconnect_callback_args);
                if (ret < 0)
                {
                    NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "[%s] [%s]: callback returned error: ret=%d", priv_data->prefix, __func__, ret);
                }
            }
            else
            {
                NVLOGC(TAG, "[%s] [%s]: reconnect_callback was not set", priv_data->prefix, __func__);
            }
        }
        is_reconnect = 1;

        // Send efd_rx and receive efd_tx
        if (send_fd(client_socket, priv_data->efd_rx))
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Send efd_rx failed");
        }
        
        int efd_tx_new = recv_fd(client_socket);
        __atomic_store_n(&priv_data->efd_tx, efd_tx_new, __ATOMIC_RELEASE);

        NVLOGC(TAG, "[%s] Received peer event_fd: %d", priv_data->prefix, efd_tx_new);

        usleep(1000 * 200);

        close(client_socket);

        if (efd_tx_new < 0)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Share event_fd failed: efd_tx=%d, efd_rx=%d", efd_tx_new, priv_data->efd_rx);
        }
        else
        {
            NVLOGC(TAG, "Share event_fd succeed: efd_tx=%d, efd_rx=%d", efd_tx_new, priv_data->efd_rx);
        }
    }
    close(server_socket);
    __atomic_store_n(&priv_data->server_socket_fd, -1, __ATOMIC_RELEASE);
    return;
}

static void unix_socket_client(thread_args_t* args)
{
    priv_data_t* priv_data = args->priv_data;

    int client_socket;
    if((client_socket = unix_sock_create(&args->client_addr)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s create client socket failed: %d", __func__, client_socket);
        return;
    }

    // Wait a server socket to be ready and connect
    while(unix_sock_connect(client_socket, &args->server_addr) != 0)
    {
        NVLOGC(TAG, "[%s] Waiting for nvipc server to start ...", priv_data->prefix);
        sleep(1);
    }

    NVLOGC(TAG, "[%s] nvipc unix socket client connected", priv_data->prefix);

    // Send efd_rx and receive efd_tx
    if(send_fd(client_socket, priv_data->efd_rx))
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Send efd_rx failed");
    }
    
    int efd_tx_new = recv_fd(client_socket);
    __atomic_store_n(&priv_data->efd_tx, efd_tx_new, __ATOMIC_RELEASE);

    NVLOGC(TAG, "[%s] Received peer event_fd: %d", priv_data->prefix, efd_tx_new);

    close(client_socket);
}

void* event_fd_share_task(void* args)
{
    priv_data_t* priv_data = ((thread_args_t*)args)->priv_data;

    struct sched_param schedprm = {.sched_priority = -1};
    int sched_policy = -1;
    if (pthread_getschedparam(pthread_self(), &sched_policy, &schedprm) != 0)
    {
        NVLOGC(TAG, "Could not get thread scheduling info");
    }
    NVLOGC(TAG, "[%s][core %02d ] share event_fd thread info: sched_policy=%d sched_priority=%d", priv_data->prefix, sched_getcpu(), sched_policy, schedprm.sched_priority);

    if(priv_data->primary)
    {
        // nv_assign_thread_cpu_core(1);
        char thread_name[16] = "efd_";
        int offset = strlen(thread_name);
        nvlog_safe_strncpy(thread_name + offset, priv_data->prefix, 16 - offset);
        pthread_setname_np(pthread_self(), thread_name);

        unix_socket_server(args);
    }
    else
    {
        unix_socket_client(args);
    }

    int efd_tx_final = __atomic_load_n(&priv_data->efd_tx, __ATOMIC_ACQUIRE);

    if(efd_tx_final < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "Share event_fd failed: efd_tx=%d, efd_rx=%d", efd_tx_final, priv_data->efd_rx);
    }
    else
    {
        NVLOGC(TAG, "Share event_fd succeed: efd_tx=%d, efd_rx=%d", efd_tx_final, priv_data->efd_rx);
    }

    free(args);

    NVLOGC(TAG, "[%s][core %02d ] nvipc unix socket exit", priv_data->prefix, sched_getcpu());
    return NULL;
}

int nv_ipc_efd_set_reconnect_callback(nv_ipc_efd_t *ipc_efd, int (*callback)(void *), void *callback_args)
{
    if (ipc_efd == NULL || callback == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: NULL pointer", __func__);
        return -1;
    }

    priv_data_t *priv_data = get_private_data(ipc_efd);
    priv_data->reconnect_callback = callback;
    priv_data->reconnect_callback_args = callback_args;
    return 0;
}

static int ipc_efd_open(nv_ipc_efd_t* ipc_efd, const char* prefix)
{
    priv_data_t* priv_data = get_private_data(ipc_efd);

    // Create efd_rx locally and get efd_tx from remote party
    __atomic_store_n(&priv_data->efd_tx, -1, __ATOMIC_RELEASE);
    __atomic_store_n(&priv_data->server_socket_fd, -1, __ATOMIC_RELEASE);
    int flag          = ENABLE_EFD_SEMAPHORE_FLAG ? EFD_SEMAPHORE : 0;
    if((priv_data->efd_rx = eventfd(0, flag)) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: create efd_rx failed", __func__);
        return -1;
    }

    // Create parameters structure for thread argument
    thread_args_t* args = malloc(sizeof(thread_args_t));
    if(args == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: memory malloc failed", __func__);
        return -1;
    }

    args->priv_data = priv_data;
    nvlog_safe_strncpy(priv_data->prefix, prefix, NV_EFD_NAME_MAX_LEN);

    // Unix server socket address
    args->server_addr.sun_family = AF_UNIX;
    nvlog_safe_strncpy(args->server_addr.sun_path, UNIX_SOCK_PATH, NV_UNIX_SOCKET_PATH_MAX_LEN);
    strncat(args->server_addr.sun_path, prefix, NV_EFD_NAME_MAX_LEN);
    strncat(args->server_addr.sun_path, UNIX_SOCK_NAME_SUFFIX_SERVER, NV_NAME_SUFFIX_MAX_LEN);

    // Unix client socket address
    args->client_addr.sun_family = AF_UNIX;
    nvlog_safe_strncpy(args->client_addr.sun_path, UNIX_SOCK_PATH, NV_UNIX_SOCKET_PATH_MAX_LEN);
    strncat(args->client_addr.sun_path, prefix, NV_EFD_NAME_MAX_LEN);
    strncat(args->client_addr.sun_path, UNIX_SOCK_NAME_SUFFIX_CLIENT, NV_NAME_SUFFIX_MAX_LEN);

    int CONFIG_NON_BLOCKING_INIT = priv_data->primary ? 1 : 0;
    if(CONFIG_NON_BLOCKING_INIT)
    {
        if(pthread_create(&priv_data->thread_id, NULL, event_fd_share_task, args) != 0)
        {
            NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: create Unix socket listener thread failed", __func__);
            return -1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        event_fd_share_task(args);
        
        int efd_tx_check = __atomic_load_n(&priv_data->efd_tx, __ATOMIC_ACQUIRE);
        
        if(efd_tx_check < 0)
        {
            return -1;
        }
        else
        {
            return 0;
        }
    }
}

nv_ipc_efd_t* nv_ipc_efd_open(int primary, const char* prefix)
{
    if(prefix == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: configuration is NULL", __func__);
        return NULL;
    }

    int           size    = sizeof(nv_ipc_efd_t) + sizeof(priv_data_t);
    nv_ipc_efd_t* ipc_efd = malloc(size);
    if(ipc_efd == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }

    memset(ipc_efd, 0, size);
    priv_data_t* priv_data = get_private_data(ipc_efd);
    priv_data->primary     = primary;

    ipc_efd->get_fd    = ipc_efd_get_rx_fd;
    ipc_efd->notify    = ipc_efd_tx_write;
    ipc_efd->get_value = ipc_efd_rx_read;

    ipc_efd->close = ipc_efd_close;

    // Event FDs initiate
    if(ipc_efd_open(ipc_efd, prefix) < 0)
    {
        ipc_efd_close(ipc_efd);
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: Failed", __func__);
        return NULL;
    }
    else
    {
        return ipc_efd;
    }
}
