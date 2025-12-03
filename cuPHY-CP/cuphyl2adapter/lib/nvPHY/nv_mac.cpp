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

#include "nv_mac.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 1) // "L2A.MAC

namespace nv
{
/**
 * @brief Constructor for MAC instance
 *
 * Initializes the transport layer using configuration from YAML.
 *
 * @param node_config YAML configuration node containing transport settings
 * @param cell_num Cell number for this MAC instance
 */
mac::mac(yaml::node node_config, uint32_t cell_num) :
    transport_(node_config["transport"], NV_IPC_MODULE_MAC, cell_num)
{
    NVLOGC_FMT(TAG, "{} construct", __func__);
}

/**
 * @brief Start the MAC thread
 *
 * Creates a thread that will invoke the virtual thread_func() method.
 */
void mac::start()
{
    // Create a thread that will invoke the virtual thread_func()
    std::thread t(&mac::thread_func, this);
    thread_.swap(t);
}

/**
 * @brief Default MAC thread function
 *
 * Base implementation that can be overridden by derived classes.
 */
void mac::thread_func()
{
    NVLOGI_FMT(TAG, "mac::thread_func()");
}

/**
 * @brief Join the MAC thread
 *
 * Blocks until the MAC thread completes execution.
 */
void mac::join()
{
    thread_.join();
}
    
} // namespace nv
