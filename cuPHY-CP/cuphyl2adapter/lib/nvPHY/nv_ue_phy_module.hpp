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

#if !defined(NV_UE_PHY_MODULE_HPP_INCLUDED_)
#define NV_UE_PHY_MODULE_HPP_INCLUDED_

#include "yaml.hpp"
#include "nv_phy_mac_transport.hpp"

#include <thread>
#include <utility>

namespace nv {

    /**
     * @brief UE PHY Module base class
     *
     * Provides the foundation for UE-side PHY processing, managing
     * communication with the MAC layer via transport and running in
     * a dedicated thread.
     */
    class UEPHY_Module {
    
        public:
        /**
         * @brief Constructor
         * @param config YAML configuration node
         * @param cell_num Cell number for this UE PHY module
         */
        UEPHY_Module(yaml::node config, uint32_t cell_num);
        
        UEPHY_Module(const UEPHY_Module& other) = delete;
        UEPHY_Module& operator=(const UEPHY_Module& other) = delete;
        
        /**
         * @brief Move constructor
         * @param other UEPHY_Module to move from
         */
        UEPHY_Module(UEPHY_Module&& other):
            transport_(std::move(other.transport_)),
            thread_(std::move(other.thread_)) {

        }
        
        /**
         * @brief Start the PHY module thread
         *
         * Initiates the PHY_module thread for message processing.
         */
        void start();
        
        /**
         * @brief Join the PHY module thread
         *
         * Blocks until all PHY instances run to completion.
         */
        void join();
        
        /**
         * @brief Get reference to the transport object
         * @return Reference to the module's phy/mac transport object
         */
        phy_mac_transport& transport() { return transport_; }
        
        protected:
        /**
         * @brief Callback method when a message is received by nvIPC
         *
         * Pure virtual function to be implemented by derived classes.
         *
         * @param ipc Reference to the received IPC message
         */
        virtual void on_msg(nv_ipc_msg_t & ipc) = 0;

        private:
        /**
         * @brief Thread loop for message processing
         */
        void thread_func();

        /**
         * @brief Receive and process a message
         * @return true if message was received, false otherwise
         */
        bool recv_msg();

        private:
        phy_mac_transport transport_;  ///< PHY-MAC transport object
        std::thread thread_;           ///< Module thread handle
    };
}
#endif