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

#ifndef _NV_IPC_EFD_H_
#define _NV_IPC_EFD_H_

#if defined(__cplusplus)
extern "C" {
#endif

/** Unix domain socket path maximum length (see sys/un.h) */
#define NV_UNIX_SOCKET_PATH_MAX_LEN 108

/** Event FD name maximum length */
#define NV_EFD_NAME_MAX_LEN 32

/**
 * Event file descriptor interface
 *
 * Provides event notification mechanism for IPC
 */
typedef struct nv_ipc_efd_t nv_ipc_efd_t;
struct nv_ipc_efd_t
{
    int (*get_fd)(nv_ipc_efd_t* ipc_efd);      //!< Get FD for select/poll/epoll
    int (*notify)(nv_ipc_efd_t* ipc_efd, int value);   //!< Notify event with value
    int (*get_value)(nv_ipc_efd_t* ipc_efd);   //!< Get and clear event value
    int (*close)(nv_ipc_efd_t* ipc_efd);       //!< Close event FD
};

/**
 * Open event file descriptor
 *
 * @param[in] primary Primary process flag (creates/initiates the event FD)
 * @param[in] prefix Instance name prefix
 * @return Pointer to event FD interface on success, NULL on failure
 */
nv_ipc_efd_t* nv_ipc_efd_open(int primary, const char* prefix);

/**
 * Set reconnection callback
 *
 * @param[in] ipc_efd Event FD instance
 * @param[in] callback Callback function to invoke on reconnect
 * @param[in] callback_args Arguments to pass to callback
 * @return 0 on success, -1 on failure
 */
int nv_ipc_efd_set_reconnect_callback(nv_ipc_efd_t* ipc_efd, int (*callback)(void *), void *callback_args);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_EFD_H_ */
