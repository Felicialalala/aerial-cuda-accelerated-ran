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
#include <cuda.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

#include "nvlog.hpp"
#include "test_kernel.h"
#include "cupti_helper.hpp"
#include "cuphy_pti.hpp"
#include "util.hpp"

#define TAG "CUPHY.CUPTI"

#define CU_CHECK_PHYDRIVER(stmt)                   \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            printf("[%s:%d] cu failed with %d\n",  \
                   __FILE__,                       \
                   __LINE__,                       \
                   result);                        \
        }                                          \
        assert(CUDA_SUCCESS == result);            \
    } while(0)

uint64_t get_cpu_ns()
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
    {
        printf("clock_gettime fail: %d\n",ret);
        exit(1);
    }
    return static_cast<uint64_t>(t.tv_nsec) + static_cast<uint64_t>(t.tv_sec) * 1000000000ULL;
}

void usage(const char* progname)
{
    printf("Usage: %s <NIC PCI Address>\n",progname);
    printf("\n  example: %s 0000:cc:00.1\n",progname);
    exit(1);
}

int main(int argc, char** argv)
{

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "cuphy_cupti_correlation_example.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(), NULL);
    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG,"Starting cuphy_cupti_correlation_example");

    CHECK_CUDA(cudaSetDevice(0));

    if (argc != 2) usage(argv[0]);
    cuphy_pti_init(argv[1]);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cuphy_cupti_helper_init();

    for (uint64_t id=0; id<1000; id++)
    {
        cuphy_cupti_helper_push_external_id(id);
        NVLOGC_FMT(TAG,"Launch loop {}: {}",id,get_cpu_ns());
        //launch_test_kernel(stream, id, 1000000, get_cpu_ns());
        cuphy_pti_calibrate_gpu_timer(stream,id);
        cuphy_cupti_helper_pop_external_id();
        usleep(1000000);

        //if ((id >= 500) && ((id % 10) == 0))
        if ((id == 500) || (id == 510))
        {
            cuphy_cupti_helper_flush();
            //usleep(2000000);
        }
        CHECK_CUDA(cudaGetLastError());
        usleep(1000000);
    }

    cuphy_cupti_helper_stop();

    usleep(5000000);
}
