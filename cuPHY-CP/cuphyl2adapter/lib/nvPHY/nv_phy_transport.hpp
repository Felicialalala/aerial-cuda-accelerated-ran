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

#if !defined(NV_PHY_CONFIG_HPP_INCLUDED_)
#define NV_PHY_CONFIG_HPP_INCLUDED_

#include "yaml.hpp"
#include "nv_ipc_endpoint.hpp"
#include "nv_phy_base_common.hpp"

#include <iostream>

namespace nv {

    template <typename T> T* create_config(yaml::node doc) { return nullptr ;}

    template <> udp_config* create_config (yaml:: node doc) {
        return new udp_config( doc["ip"].as<std::string>(),
            doc["port"].as<int>(),
            doc["local_port"].as<int>());
    }

    template <> shm_config* create_config (yaml:: node doc) {
        return new shm_config(doc["cell_index"].as<int>());
    }
 

    class PHY_transport {
        public:
        PHY_transport(yaml::node transport_config) {
            
            std::string mac_ipc_type = transport_config["mac_endpoint"]["type"].as<std::string>();
            muxer = new epoll_muxer();
            std::string ue_ipc_type = transport_config["ue_endpoint"]["type"].as<std::string>();
            std::cout<<"MAC IPC:"<<mac_ipc_type<<" UE IPC:"<<ue_ipc_type<<std::endl;
            if (mac_ipc_type.compare("udp") == 0) {
                // Passing nullptr for reader and writer since get_reader() and get_reader() are
                // virtual methods
                mac_intf = new udp_endpoint(create_config<udp_config>(transport_config["mac_endpoint"]), nullptr, nullptr);
                muxer->add_endpoint(mac_intf, EPOLLIN | EPOLLET);
            } else if (mac_ipc_type.compare("shm") == 0) {
                mac_intf = new shm_endpoint(create_config<shm_config>(transport_config["mac_endpoint"]));
            }


            if (ue_ipc_type.compare("udp") == 0) {
                ue_intf = new udp_endpoint(create_config<udp_config>(transport_config["ue_endpoint"]), nullptr, nullptr);
                //muxer->add_endpoint(ue_intf, EPOLLIN | EPOLLET | EPOLLOUT);
            } else if (ue_ipc_type.compare("shm") == 0) {
                ue_intf = new shm_endpoint(create_config<shm_config>(transport_config["ue_endpoint"]));
            }
        }
        PHY_transport(const PHY_transport&) = delete;
        PHY_transport& operator=(const PHY_transport&) = delete;
        
        PHY_transport(PHY_transport && other) {
            mac_intf = other.mac_intf;
            ue_intf = other.ue_intf;
            muxer = other.muxer;
        }

        PHY_transport& operator=(PHY_transport && other) {
            mac_intf = other.mac_intf;
            ue_intf = other.ue_intf;
            muxer = other.muxer;
            // phy_instance_ = other.phy_instance_;
        }
        void init() {
            // if (phy_instance_ == nullptr) {
            //     std::runtime_error("Invalid Phy instance");
            //     return;
            // }
            // reader* read = phy_instance_->get_mac_handler()->get_reader();
            // mac_intf->set_reader(read);
            // mac_intf->set_writer(phy_instance_->get_mac_handler()->get_writer());
        }
        void start() {
            muxer->mux();
        }

        void add_to_mux(ipc_base * ipc) {
            muxer->add_endpoint(ipc, 0);
        }

        void remove_from_mux(ipc_base * ipc) {
            muxer->remove_endpoint(ipc);
        }
        ipc_endpoint* get_nb_intf() {
            return mac_intf;
        }

        ipc_endpoint* get_sb_intf() {
            return ue_intf;
        }
        private:
        ipc_endpoint* mac_intf;
        ipc_endpoint* ue_intf;
        epoll_muxer *muxer;
    };
}
#endif
