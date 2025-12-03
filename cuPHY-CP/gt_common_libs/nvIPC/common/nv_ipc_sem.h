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

#ifndef _NV_IPC_SEM_H_
#define _NV_IPC_SEM_H_

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/** Semaphore name maximum length */
#define NV_SEM_NAME_MAX_LEN 32

/**
 * IPC semaphore interface
 *
 * Provides semaphore-based synchronization for IPC
 */
typedef struct nv_ipc_sem_t nv_ipc_sem_t;
struct nv_ipc_sem_t
{
    int (*sem_post)(nv_ipc_sem_t* ipc_sem);    //!< Post semaphore (increment)
    int (*sem_wait)(nv_ipc_sem_t* ipc_sem);    //!< Wait on semaphore (decrement, blocks if zero)
    int (*get_value)(nv_ipc_sem_t* ipc_sem, int* value);  //!< Get current semaphore value
    int (*close)(nv_ipc_sem_t* ipc_sem);       //!< Close semaphore
};

/**
 * Open IPC semaphore
 *
 * @param[in] primary Primary process flag
 * @param[in] prefix Instance name prefix
 * @return Pointer to semaphore interface on success, NULL on failure
 */
nv_ipc_sem_t* nv_ipc_sem_open(int primary, const char* prefix);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_SEM_H_ */
