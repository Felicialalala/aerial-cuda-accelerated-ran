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

#ifndef _AERIAL_EVENT_CODE_H_
#define _AERIAL_EVENT_CODE_H_

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

typedef enum
{
    AERIAL_SUCCESS             = 0,
    AERIAL_INVALID_PARAM_EVENT = 1,
    AERIAL_INTERNAL_EVENT      = 2,
    AERIAL_CUDA_API_EVENT      = 3,
    AERIAL_DPDK_API_EVENT      = 4,
    AERIAL_THREAD_API_EVENT    = 5,
    AERIAL_CLOCK_API_EVENT     = 6,
    AERIAL_NVIPC_API_EVENT     = 7,
    AERIAL_ORAN_FH_EVENT       = 8,
    AERIAL_CUPHYDRV_API_EVENT  = 9,
    AERIAL_INPUT_OUTPUT_EVENT  = 10,
    AERIAL_MEMORY_EVENT        = 11,
    AERIAL_YAML_PARSER_EVENT   = 12,
    AERIAL_NVLOG_EVENT         = 13,
    AERIAL_CONFIG_EVENT        = 14,
    AERIAL_FAPI_EVENT          = 15,
    AERIAL_NO_SUPPORT_EVENT    = 16,
    AERIAL_SYSTEM_API_EVENT    = 17,
    AERIAL_L2ADAPTER_EVENT     = 18,
    AERIAL_RU_EMULATOR_EVENT   = 19,
    AERIAL_CUDA_KERNEL_EVENT   = 20,
    AERIAL_CUPHY_API_EVENT     = 21,
    AERIAL_DOCA_API_EVENT      = 22,
    AERIAL_CUPHY_EVENT         = 23,
    AERIAL_CUPHYOAM_EVENT      = 24,
    AERIAL_CUMAC_EVENT         = 25,
    AERIAL_TEST_CUMAC_EVENT    = 26,
    AERIAL_TESTBENCH_EVENT     = 27,
    AERIAL_CUMAC_CP_EVENT      = 28,
    AERIAL_TEST_MAC_EVENT      = 29,
    AERIAL_PYAERIAL_EVENT      = 30,
    AERIAL_PTP_ERROR_EVENT     = 31,
} aerial_event_code_t;

char* nvlog_strerror(int code);

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#endif /* _AERIAL_EVENT_CODE_H_ */
