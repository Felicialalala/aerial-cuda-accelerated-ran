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

#ifndef _NV_IPC_GPUDATAPOOL_H_
#define _NV_IPC_GPUDATAPOOL_H_

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define GPUDATA_INFO_SIZE 512
#define ROUND_UP_GDR(x, n)     (((x) + ((n) - 1)) & ~((n) - 1))

typedef struct nv_ipc_gpudatapool_t nv_ipc_gpudatapool_t;
struct nv_ipc_gpudatapool_t
{
    void* (*get_gpudatapool_addr)(nv_ipc_gpudatapool_t* gpudatapool);

    int (*memcpy_to_host)(nv_ipc_gpudatapool_t* gpudatapool, void* host, const void* device, size_t size);

    int (*memcpy_to_device)(nv_ipc_gpudatapool_t* gpudatapool, void* device, const void* host, size_t size);

    int (*close)(nv_ipc_gpudatapool_t* gpudatapool);
};

nv_ipc_gpudatapool_t* nv_ipc_gpudatapool_open(int primary, void* shm, size_t size, int deviceId);
int nv_ipc_gpudatapool_reInit(nv_ipc_gpudatapool_t* pGpuDataPool);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_GPUDATAPOOL_H_ */
