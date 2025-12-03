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

#include "scf_5g_fapi_module_dispatch.hpp"
#include "scf_5g_fapi.h"
#include "scf_5g_fapi_msg_helpers.hpp"

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 2) // "SCF.DISPATCH"

namespace scf_5g_fapi
{

////////////////////////////////////////////////////////////////////////
// module_dispatch::module_dispatch()
module_dispatch::module_dispatch()
{
    NVLOGI_FMT(TAG, "scf_5g_fapi::module_dispatch::module_dispatch(), {}", reinterpret_cast<void*>(this));
}

////////////////////////////////////////////////////////////////////////
// module_dispatch::dispatch()
bool module_dispatch::dispatch(nv_ipc_msg_t&                      msg,
                               std::vector<nv::PHY_instance_ref>& instances)
{
    NVLOGD_FMT(TAG, "scf_5g_fapi::module_dispatch::dispatch()");
    scf_fapi_header_t *fapi_hdr = static_cast<scf_fapi_header_t*>(msg.msg_buf);

    //------------------------------------------------------------------
    // Check for out-of-bounds handle IDs
    if(fapi_hdr->handle_id >= instances.size())
    {
        throw std::runtime_error(std::string("Invalid scf_5g_fapi header handle ID (") +
                                 std::to_string(fapi_hdr->handle_id) +
                                 std::string(")"));
    }
    if(fapi_hdr->handle_id != msg.cell_id)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "fapi_hdr->handle_id={} cell_id={}", fapi_hdr->handle_id, msg.cell_id);
    }
    //------------------------------------------------------------------
    // Call the instance message handler
    return instances[fapi_hdr->handle_id].get().on_msg(msg);
}


} // namespace scf_5g_fapi


