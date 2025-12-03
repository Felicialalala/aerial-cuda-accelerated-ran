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

#ifndef NV_IPC_DPDK_UTILS_H
#define NV_IPC_DPDK_UTILS_H

#include <stdint.h>
#include <stddef.h>

#include "nv_ipc.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if 0
int memcpy_from_nvipc(const nv_ipc_msg_t* src, uint8_t* dst, uint32_t size);
int memcpy_to_nvipc(nv_ipc_msg_t* dst, const uint8_t* src, uint32_t size);
#endif

/**
 * Create DPDK task on specific lcore
 *
 * @param[in] func Task function to execute
 * @param[in] arg Arguments to pass to task function
 * @param[in] lcore_id Logical core ID to run task on
 * @return 0 on success, -1 on failure
 */
int  create_dpdk_task(int (*func)(void* arg), void* arg, uint16_t lcore_id);

/**
 * Initialize DPDK environment
 *
 * @param[in] argv0 Program name (argv[0])
 * @param[in] cfg DPDK configuration
 * @return 0 on success, -1 on failure
 */
int  nv_ipc_dpdk_init(const char* argv0, const nv_ipc_config_dpdk_t* cfg);

/**
 * Print current lcore information
 *
 * @param[in] info Prefix string for output
 */
void dpdk_print_lcore(const char* info);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* NV_IPC_DPDK_UTILS_H */
