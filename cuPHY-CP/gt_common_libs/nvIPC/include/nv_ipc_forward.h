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

#ifndef NV_IPC_FORWARD_H_INCLUDED_
#define NV_IPC_FORWARD_H_INCLUDED_

#include <time.h>
#include "nv_ipc.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start message forwarding
 *
 * @param[in] ipc IPC instance
 * @param[in] count Number of messages to forward (0 = infinite)
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_start(nv_ipc_t* ipc, uint32_t count);

/**
 * Stop message forwarding
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_stop(nv_ipc_t* ipc);

/**
 * Wait for forwarding semaphore with timeout
 *
 * Use clock_gettime(CLOCK_REALTIME, ts_abs) to get current timestamp and add timeout value
 *
 * @param[in] ipc IPC instance
 * @param[in] ts_abs Absolute timeout timestamp
 * @return 0 on success, -1 on timeout/failure
 */
int nvipc_fw_sem_timedwait(nv_ipc_t* ipc, const struct timespec* ts_abs);

/**
 * Wait for forwarding semaphore
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_sem_wait(nv_ipc_t* ipc);

/**
 * Enqueue message for forwarding
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message to enqueue
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_enqueue(nv_ipc_t *ipc, nv_ipc_msg_t *msg);

/**
 * Dequeue forwarded message
 *
 * @param[in] ipc IPC instance
 * @param[out] msg Buffer to store dequeued message
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_dequeue(nv_ipc_t* ipc, nv_ipc_msg_t* msg);

/**
 * Free forwarded message buffer (after dequeue)
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message to free
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_free(nv_ipc_t* ipc, nv_ipc_msg_t* msg);

/**
 * Reset forwarding queue
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int nvipc_fw_reset(nv_ipc_t *ipc);

/**
 * Get forwarding status
 *
 * @param[in] ipc IPC instance
 * @return 1 if started, 0 if stopped
 */
int nvipc_fw_get_started(nv_ipc_t* ipc);

/**
 * Get lost message count
 *
 * Returns count of messages lost due to queue full during forwarding
 *
 * @param[in] ipc IPC instance
 * @return Number of lost messages
 */
uint32_t nvipc_fw_get_lost(nv_ipc_t* ipc);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* NV_IPC_FORWARD_H_INCLUDED_ */
