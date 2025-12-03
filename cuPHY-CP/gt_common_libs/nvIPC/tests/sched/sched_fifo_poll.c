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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <pthread.h>

int nv_set_sched_fifo_priority(int priority) {
    struct sched_param param;
    param.__sched_priority = priority;
    pthread_t thread_me = pthread_self();
    if (pthread_setschedparam(thread_me, SCHED_FIFO, &param) != 0) {
        printf("%s: line %d errno=%d: %s\n", __func__, __LINE__, errno, strerror(errno));
        return -1;
    } else {
        printf("%s: OK: thread=%ld priority=%d\n", __func__, thread_me, priority);
        return 0;
    }
}

int nv_assign_thread_cpu_core(int cpu_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    int ret;
    if ((ret = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask)) != 0) {
        printf("%s: line %d ret=%d errno=%d: %s\n", __func__,
        __LINE__, ret, errno, strerror(errno));
        return -1;
    } else {
        printf("%s: OK: thread=%ld cpu_id=%d\n", __func__, pthread_self(), cpu_id);
        return 0;
    }
}

int main(int argc, char **argv) {

    printf("%s: started on CPU core %d\n", __func__, sched_getcpu());

    nv_assign_thread_cpu_core(3);

    nv_set_sched_fifo_priority(90);

    printf("%s: run SCHED_FIFO while(1) loop on core %d ...\n", __func__, sched_getcpu());

    int ret = 0;

    while (1) {
        ret += 1;
    }

    // printf("Never run to here\n");

    return ret;
}
