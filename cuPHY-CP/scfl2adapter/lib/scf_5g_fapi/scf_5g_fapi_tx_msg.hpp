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

#if !defined(SCF_5G_FAPI_TX_MSG_HPP_INCLUDED_)
#define SCF_5G_FAPI_TX_MSG_HPP_INCLUDED_

#include "scf_5g_fapi.h"
#include "nv_phy_mac_transport.hpp"

namespace scf_5g_fapi
{

////////////////////////////////////////////////////////////////////////
// tx_msg_builder
// Helper class to construct messages in-place in nvIPC message queue
// buffers.
class tx_msg_builder
{
public:
    //------------------------------------------------------------------
    // Constructor
    tx_msg_builder(nv_ipc_msg_t& msg) :
        ipc_msg(msg),
        next_addr(static_cast<char*>(msg.msg_buf))
    {
        header().message_count = 0;
        header().handle_id     = 0;
        next_addr += sizeof(scf_fapi_header_t);
    }
    //------------------------------------------------------------------
    // header()
    // Return a reference to the header portion of the underlying buffer
    scf_fapi_header_t& header()
    {
        return *(static_cast<scf_fapi_header_t*>(ipc_msg.msg_buf));
    }
    //------------------------------------------------------------------
    // alloc()
    // Allocate a message of the given type in the buffer. (Only suitable
    // for fixed size messages at the moment, in this form, because the
    // size of the static type is used to determine the message
    // allocation.)
    template <class TMsg> TMsg& alloc();
private:
    //------------------------------------------------------------------
    // Data
    nv_ipc_msg_t& ipc_msg;
    char*         next_addr;
};

////////////////////////////////////////////////////////////////////////
// tx_msg_builder::alloc<scf_fapi_config_request_msg_t>()
// Temporary implementation for initial development - this won't be a
// fixed size message.
template <>
scf_fapi_config_request_msg_t& tx_msg_builder::alloc<scf_fapi_config_request_msg_t>()
{
    // Cast the current address to the requested type
    scf_fapi_config_request_msg_t& r = reinterpret_cast<scf_fapi_config_request_msg_t&>(*next_addr);
    // Set the message header values (type and length)
    r.msg_hdr.type_id = SCF_FAPI_CONFIG_REQUEST;
    r.msg_hdr.length  = sizeof(r.msg_body);
    // Increment the number of messages
    ++header().message_count;
    // Advance the address for the next message allocation
    next_addr += sizeof(scf_fapi_config_request_msg_t);
    return r;
}
    
} // namespace scf_5g_fapi

#endif // !defined(SCF_5G_FAPI_TX_MSG_HPP_INCLUDED_)
