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

#if !defined(NV_MAC_FACTORY_HPP_INCLUDED_)
#define NV_MAC_FACTORY_HPP_INCLUDED_

#include "nv_mac.hpp"
#include "yaml.hpp"
#include <vector>

namespace nv
{


struct mac_creator;

/**
 * @brief Factory class for creating MAC instances
 *
 * The mac_factory provides a registry-based factory pattern for creating
 * different types of MAC implementations at runtime.
 */
class mac_factory
{
public:
    /**
     * @brief Register a new MAC type with the factory
     * @param c Reference to the mac_creator containing type information
     */
    static void register_type(mac_creator& c);
    
    /**
     * @brief Create a MAC instance based on configuration
     * @param node_config YAML configuration node
     * @param cell_num Cell number for this MAC instance
     * @return Pointer to the created MAC instance
     */
    static mac* create(yaml::node node_config, uint32_t cell_num);
    
    /**
     * @brief Get list of all registered MAC types
     * @return Vector of type name strings
     */
    static std::vector<const char*> get_registered_types();
private:
};

/// Function pointer type for MAC creation functions
typedef mac* (*mac_create_func)(yaml::node node_config, uint32_t cell_num);

/**
 * @brief Structure holding MAC creator registration information
 *
 * This structure maintains a singly-linked list of registered MAC types
 * via the next pointer.
 */
struct mac_creator
{
    const char*     type_name;      ///< Name of the MAC type
    mac_create_func create_func;    ///< Function to create instance
    mac_creator*    next;           ///< Next creator in the linked list
    //mac_creator(const char* n, mac_create_func f);
};


} // namespace nv

#endif // !defined(NV_MAC_FACTORY_HPP_INCLUDED_)
