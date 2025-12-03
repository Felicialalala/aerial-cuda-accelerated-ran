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

#if !defined(SCF_5G_FAPI_MAC_HPP_INCLUDED_)
#define SCF_5G_FAPI_MAC_HPP_INCLUDED_

#include "nv_mac.hpp"

namespace scf_5g_fapi
{

/**
 * @brief SCF 5G FAPI MAC implementation
 *
 * Concrete MAC implementation for the SCF 5G FAPI specification.
 */
class mac : public nv::mac
{
public:
    /**
     * @brief Constructor
     * @param node_config YAML configuration node
     * @param cell_num Cell number for this MAC instance
     */
    mac(yaml::node node_config, uint32_t cell_num);
private:
};

} // namespace scf_5g_fapi

#endif // !defined(SCF_5G_FAPI_MAC_HPP_INCLUDED_)
