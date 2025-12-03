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

#include "nv_ue_phy_module.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 8) // "L2A.UEMD"

namespace nv {

    UEPHY_Module::UEPHY_Module(yaml::node config, uint32_t cell_num):
        transport_(config["transport"], NV_IPC_MODULE_PHY, cell_num) {

    }

    void UEPHY_Module::start() {
        std::thread t(&UEPHY_Module::thread_func, this);
        thread_.swap(t);
    }

    void UEPHY_Module::join() {
        thread_.join();
    }

    bool UEPHY_Module::recv_msg() {
        return true;
        // rx_msg_desc srmsg_desc(transport());
        // if(transport_.rx_recv(srmsg_desc.msg_desc) < 0) {
        //     NVLOGI_FMT(TAG, "No further messages");
        //     return false;
        // }
        // NVLOGI_FMT(TAG, "UEPHY_module::recv_msg(): MESSAGE RECEIVED!");
        // on_msg(srmsg_desc.msg_desc);
    }

    void UEPHY_Module::thread_func() {
        #if 1
            while(1)
        #else 
            do 
        #endif
        {
            try {
                transport_.rx_wait();
                while (recv_msg()) {

                }
                transport_.notify(1);
            }
        catch(std::exception& e)
        {
            NVLOGW_FMT(TAG, "PHY_module::thread_func() exception: {}", e.what());
        }
        catch(...)
        {
            NVLOGW_FMT(TAG, "PHY_module::thread_func() unknown exceptio");
        }

        #if 0
        }
        #else
        } while(0);
        #endif
    }

}
