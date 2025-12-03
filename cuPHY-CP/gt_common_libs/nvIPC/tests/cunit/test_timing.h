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

#ifndef _TEST_TIMING_H_
#define _TEST_TIMING_H_

#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>
#include <CUnit/Automated.h>
#include <CUnit/Console.h>

#include "array_queue.h"
#include "nv_ipc.h"
#include "nv_ipc_shm.h"
#include "nv_ipc_efd.h"
#include "nv_ipc_sem.h"
#include "nv_ipc_epoll.h"
#include "nv_ipc_ring.h"
#include "nv_ipc_mempool.h"
#include "nv_ipc_cudapool.h"

#include "test_cuda.h"
#include "stat_log.h"
#include "nv_ipc_utils.h"

#if defined(__cplusplus)
extern "C" {
#endif

int assin_cpu_for_thread(int cpu_id);
int assign_cpu_for_process(int cpu);

void test_stat_log(void);
void test_timing(void);

#if defined(__cplusplus)
}
#endif

#endif /* _TEST_TIMING_H_ */
