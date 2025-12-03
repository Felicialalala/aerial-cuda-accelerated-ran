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

#ifndef _FH_SHM_DEBUG_H_
#define _FH_SHM_DEBUG_H_

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

#include <time.h>
#include <stdatomic.h>
#include "shm_logger.h"

int64_t fh_convert_pcap(FILE* record_file, char* pcap_filepath, long shm_cache_size, int32_t max_msg_size, int32_t max_data_size, long total_size, uint64_t break_offset);

#if defined(__cplusplus)
}
#endif

#endif /* _FH_SHM_DEBUG_H_ */