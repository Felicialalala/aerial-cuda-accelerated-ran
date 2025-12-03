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

#if !defined(SCF_5G_FAPI_MODULE_DISPATCH_HPP_INCLUDED_)
#define SCF_5G_FAPI_MODULE_DISPATCH_HPP_INCLUDED_

#include "nv_phy_module.hpp"

namespace scf_5g_fapi
{

////////////////////////////////////////////////////////////////////////
// scf_5g_fapi::module_dispatch
class module_dispatch : public nv::PHY_module_dispatch
{
public:
    module_dispatch();
    virtual bool dispatch(nv_ipc_msg_t&                      msg,
                          std::vector<nv::PHY_instance_ref>& instances) override;
};

} // namespace scf_5g_fapi

#endif // !defined(SCF_5G_FAPI_MODULE_DISPATCH_HPP_INCLUDED_)
