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

#if !defined(NV_PHY_BASE_COMMON_HPP_INCLUDED_)
#define NV_PHY_BASE_COMMON_HPP_INCLUDED_

#include <cstdint>
#include <iostream>
namespace nv {

    struct timer_expiry_handler {
        virtual void operator()(const uint64_t& expiries) {}
    };

    struct ipc_endpoint;
    struct reader {
        virtual void operator()(void *buf,std::size_t buf_len, ipc_endpoint* intf = nullptr) = 0;
    };

    struct writer {
        virtual std::size_t operator()(void* buf, uint16_t msg_id, ipc_endpoint* intf = nullptr) = 0;
    };

   struct msg_handler {
        virtual reader* get_reader() { return nullptr; }
        virtual writer* get_writer() { return nullptr; }
    };

    enum msg_direction {
        mac_bound,
        remote_peer_bound
    };
    class PHY_instance_base {
        public:
        virtual msg_handler* get_mac_handler() {return nullptr;}
        virtual msg_handler* get_ue_sim_handler() { return nullptr; }
    };
 
    struct ipc_base {
        static constexpr int INVALID_FD = -1;
        virtual int get_fd() { return INVALID_FD;}
        virtual void set_blocking_fd(bool) {}
        virtual void read() {}
        virtual void write(uint16_t msg_id) {}
        virtual void write( void* buf, std::size_t length) {}

        virtual reader* get_reader() { return nullptr;}
        virtual writer* get_writer() { return nullptr;}

        virtual void set_writer(writer * writer) {}
        virtual void set_reader(reader* reader) {}
    };

    struct ipc_config {

        static constexpr std::size_t MAX_MSG_BUF_SIZE = 1024 * 1024;
        static constexpr std::size_t MAX_DATA_BUF_SIZE = MAX_MSG_BUF_SIZE;
        std::size_t max_msg_buf_size_;
        std::size_t max_data_buf_size_;
        
        ipc_config():
            max_msg_buf_size_(MAX_MSG_BUF_SIZE),
            max_data_buf_size_(MAX_DATA_BUF_SIZE)
        {

        }
    };
}

#endif
