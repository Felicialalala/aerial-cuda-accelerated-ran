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

#include "scf_5g_fapi.hpp"
#include "scf_5g_fapi_mac.hpp"
#include "scf_5g_fapi_phy.hpp"
#include "scf_5g_fapi_module_dispatch.hpp"
#include "nv_mac_factory.hpp"
#include "nv_phy_factory.hpp"
#include "nv_phy_module.hpp"
#include "nv_module_dispatch_factory.hpp"

namespace
{

/**
 * @brief Create a SCF 5G FAPI MAC instance
 * @param node_config YAML configuration node
 * @param cell_num Cell number
 * @return Pointer to the created MAC instance
 */
nv::mac* create_scf_5g_fapi_mac(yaml::node node_config, uint32_t cell_num)
{
    return new scf_5g_fapi::mac(node_config, cell_num);
}

/// MAC creator structure for SCF 5G FAPI
nv::mac_creator scf_5g_fapi_mac_creator =
{
    "scf_5g_fapi",
    &create_scf_5g_fapi_mac
};

/**
 * @brief Create a SCF 5G FAPI PHY instance
 * @param phy_module Reference to the parent PHY module
 * @param node_config YAML configuration node
 * @return Unique pointer to the created PHY instance
 */
std::unique_ptr<nv::PHY_instance> create_scf_5g_fapi_phy(nv::PHY_module& phy_module,
                                         yaml::node      node_config)
{
    return std::make_unique<scf_5g_fapi::phy>(phy_module, node_config);
}

/// PHY creator structure for SCF 5G FAPI
nv::phy_creator scf_5g_fapi_phy_creator =
{
    "scf_5g_fapi",
    &create_scf_5g_fapi_phy
};

/**
 * @brief Create a SCF 5G FAPI module dispatcher
 * @return Pointer to the created module dispatcher instance
 */
nv::PHY_module_dispatch* create_scf_5g_fapi_module_dispatch()
{
    return new scf_5g_fapi::module_dispatch;
}

/// Module dispatcher creator structure for SCF 5G FAPI
nv::module_dispatch_creator scf_5g_fapi_module_dispatch_creator =
{
    "scf_5g_fapi",
    &create_scf_5g_fapi_module_dispatch
};

} // namespace


namespace scf_5g_fapi
{

/**
 * @brief Initialize SCF 5G FAPI module
 *
 * Registers MAC, PHY, and module dispatcher creators with their
 * respective factories to enable runtime instantiation.
 */
void init()
{
    nv::mac_factory::register_type(scf_5g_fapi_mac_creator);
    nv::phy_factory::register_type(scf_5g_fapi_phy_creator);
    nv::module_dispatch_factory::register_type(scf_5g_fapi_module_dispatch_creator);
}
    
} // namespace scf_5g_fapi

