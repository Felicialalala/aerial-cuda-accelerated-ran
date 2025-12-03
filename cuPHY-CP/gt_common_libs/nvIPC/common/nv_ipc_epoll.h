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

#ifndef _NV_IPC_EPOLL_H_
#define _NV_IPC_EPOLL_H_

#include <sys/epoll.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct
{
    int                epoll_fd;
    int                target_fd;
    int                max_events;
    struct epoll_event events[];
} nv_ipc_epoll_t;

nv_ipc_epoll_t* ipc_epoll_create(int max_events, int target_fd);
int             ipc_epoll_wait(nv_ipc_epoll_t* ipc_epoll);
int             ipc_epoll_destroy(nv_ipc_epoll_t* ipc_epoll);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_EPOLL_H_ */
