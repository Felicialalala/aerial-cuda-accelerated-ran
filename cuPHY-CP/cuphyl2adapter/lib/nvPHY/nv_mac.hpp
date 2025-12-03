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

#if !defined(NV_MAC_HPP_INCLUDED_)
#define NV_MAC_HPP_INCLUDED_

#include "nv_phy_mac_transport.hpp"
#include <thread>

namespace nv
{

/**
 * @brief Base class for MAC instance
 *
 * Provides foundation for MAC layer simulation, allowing different MAC
 * implementations to run in separate threads and communicate with PHY
 * via transport layer.
 */
class mac
{
public:
    /**
     * @brief Constructor
     *
     * Initializes member transport variable using the YAML "transport" node.
     *
     * @param node_config YAML configuration node
     * @param cell_num Cell number for this MAC instance
     */
    mac(yaml::node node_config, uint32_t cell_num);
    
    virtual ~mac() {};

    /**
     * @brief Thread function to be overridden by derived classes
     *
     * This virtual function is invoked in a separate thread when start() is called.
     */
    virtual void thread_func();
    
    /**
     * @brief Start the MAC thread
     *
     * Creates a new thread that executes the thread_func() method.
     */
    void start();
    
    /**
     * @brief Join the MAC thread
     *
     * Waits for the MAC thread to complete execution.
     */
    void join();
protected:
    /**
     * @brief Get reference to the transport layer
     * @return Reference to the phy_mac_transport object
     */
    phy_mac_transport& transport() { return transport_; }
private:
    phy_mac_transport transport_; ///< MAC-PHY transport object
    std::thread       thread_;    ///< MAC thread handle
};

} // namespace nv

#endif // !defined(NV_MAC_HPP_INCLUDED_)
