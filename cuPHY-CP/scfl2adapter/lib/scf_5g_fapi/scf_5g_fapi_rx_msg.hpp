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

#if !defined(SCF_5G_FAPI_RX_MSG_HPP_INCLUDED_)
#define SCF_5G_FAPI_RX_MSG_HPP_INCLUDED_

#include "scf_5g_fapi.h"
#include "nv_phy_mac_transport.hpp"

namespace scf_5g_fapi
{

////////////////////////////////////////////////////////////////////////
// rx_msg_reader
// Helper class to read messages located in nvIPC message queue buffers.
class rx_msg_reader
{
public:
    class iterator
    {
    public:
        iterator(char* addr = nullptr, uint8_t index = 0, uint8_t end_index = 0) :
            addr_(addr),
            index_(index),
            end_index_(end_index)
        {
        }
        iterator& operator++()
        {
            index_ = std::min(static_cast<uint8_t>(index_ + 1), end_index_);
            addr_ = (index_ == end_index_) ? nullptr : (addr_ + header().length);
            return *this;
        }
        bool operator!=(const iterator& it) { return (index_ != it.index_); }
        scf_fapi_body_header_t& operator*() { return header(); }
    private:
        scf_fapi_body_header_t& header() { return *reinterpret_cast<scf_fapi_body_header_t*>(addr_); }
        char*   addr_;
        uint8_t index_;
        uint8_t end_index_;
    };
    //------------------------------------------------------------------
    // Constructor
    rx_msg_reader(nv_ipc_msg_t& msg) :
        ipc_msg_(msg)
    {
        hdr = ipc_msg_.msg_buf;
    }
    //------------------------------------------------------------------
    // header()
    // Return a reference to the header portion of the underlying buffer
    scf_fapi_header_t& header()
    {
        return *(static_cast<scf_fapi_header_t*>(hdr));
    }
    //------------------------------------------------------------------
    // begin()
    iterator begin()
    {
        return iterator(static_cast<char*>(hdr) + sizeof(scf_fapi_header_t),
                        0,
                        header().message_count);
    }
    //------------------------------------------------------------------
    // end()
    iterator end()
    {
        return iterator(nullptr, header().message_count, header().message_count);
    }
private:
    //------------------------------------------------------------------
    // Data
    nv_ipc_msg_t& ipc_msg_;
    void *hdr;
};

    
} // namespace scf_5g_fapi

#endif // !defined(SCF_5G_FAPI_RX_MSG_HPP_INCLUDED_)
