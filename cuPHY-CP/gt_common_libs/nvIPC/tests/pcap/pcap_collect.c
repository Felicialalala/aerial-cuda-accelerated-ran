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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "nvlog.h"
#include "shm_logger.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 31) // "NVIPC.PCAP"

#define NV_PATH_MAX_LEN 512

char logger_name[NVLOG_NAME_MAX_LEN]                 = "nvipc";
char dest_path[NV_PATH_MAX_LEN + NVLOG_NAME_MAX_LEN] = ".";

int main(int argc, char** argv)
{
    NVLOGC(TAG, "Usage: %s <name> <destination path>\n", argv[0]);

    nvlog_set_log_level(NVLOG_INFO);

    int offset = 0;

    if(argc >= 2)
    {
        if(strnlen(argv[1], NVLOG_NAME_MAX_LEN) > 0)
        {
            offset += snprintf(logger_name, NVLOG_NAME_MAX_LEN, "%s", argv[1]);
        }
    }

    if(argc >= 3)
    {
        if(strnlen(argv[2], NV_PATH_MAX_LEN) > 0)
        {
            snprintf(dest_path, NV_PATH_MAX_LEN, "%s", argv[2]);
        }
    }

    NVLOGC(TAG, "Current run: %s name=%s dest_path=%s\n", argv[0], logger_name, dest_path);

    shmlogger_collect(logger_name, "pcap", dest_path);

    return 0;
}
