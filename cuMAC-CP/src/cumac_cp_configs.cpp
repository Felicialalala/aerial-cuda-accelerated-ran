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

#include "nvlog.hpp"

#include "cumac_cp_configs.hpp"

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 2) // "CUMCP.CFG"

using namespace std;

cumac_cp_configs::cumac_cp_configs(yaml::node config_node) : yaml_root(config_node)
{
    max_msg_size  = config_node["transport"]["shm_config"]["mempool_size"]["cpu_msg"]["buf_size"].as<int>();
    max_data_size = config_node["transport"]["shm_config"]["mempool_size"]["cpu_data"]["buf_size"].as<int>();

    recv_thread_config.name           = config_node["recv_thread_config"]["name"].as<std::string>();
    recv_thread_config.cpu_affinity   = config_node["recv_thread_config"]["cpu_affinity"].as<int>();
    recv_thread_config.sched_priority = config_node["recv_thread_config"]["sched_priority"].as<int>();

    thread_num_per_core = config_node["thread_num_per_core"].as<int>();
    cell_num = config_node["cell_num"].as<uint32_t>();
    task_ring_len = config_node["task_ring_len"].as<uint32_t>();
    run_in_cpu = config_node["run_in_cpu"].as<uint32_t>();
    debug_option = config_node["debug_option"].as<int>();

    cumac_group_tv_file = config_node["cumac_group_tv_file"].as<std::string>();

    gpu_id = config_node["gpu_id"].as<uint32_t>();

    cuda_block_num = config_node["cuda_block_num"].as<uint32_t>();

    // Performance tuning parameters
    group_buffer_enable = config_node["group_buffer_enable"].as<uint32_t>();
    multi_stream_enable = config_node["multi_stream_enable"].as<uint32_t>();
    slot_concurrent_enable = config_node["slot_concurrent_enable"].as<uint32_t>();

    yaml::node worker_cores_node = config_node["worker_cores"];
    if (worker_cores_node.length() > 0)
    {
        worker_cores.resize(worker_cores_node.length());
        for (int i = 0; i < worker_cores.size(); i ++)
        {
            worker_cores[i] = worker_cores_node[i].as<int>();
        }
    }
    else
    {
        // Use the recv_thread_config CPU core by default
        worker_cores.resize(1);
        worker_cores[0] = recv_thread_config.cpu_affinity;
    }
}

cumac_cp_configs::~cumac_cp_configs()
{
}
