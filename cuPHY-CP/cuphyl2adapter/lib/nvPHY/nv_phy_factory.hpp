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

#if !defined(NV_PHY_FACTORY_HPP_INCLUDED_)
#define NV_PHY_FACTORY_HPP_INCLUDED_

#include "nv_phy_instance.hpp"
#include "nv_tick_generator.hpp"

#include "yaml.hpp"
#include <vector>
#include <memory>
namespace nv
{

struct phy_creator;
class PHY_module;

/**
 * @brief Factory class for creating PHY instances
 *
 * The phy_factory provides a registry-based factory pattern for creating
 * different types of PHY implementations at runtime.
 */
class phy_factory
{
public:
    /**
     * @brief Register a new PHY type with the factory
     * @param c Reference to the phy_creator containing type information
     */
    static void                     register_type(phy_creator& c);
    
    /**
     * @brief Create a PHY instance based on configuration
     * @param phy_class Name of the PHY class to instantiate
     * @param phy_module Reference to the parent PHY module
     * @param node_config YAML configuration node
     * @return Unique pointer to the created PHY instance
     */
    static std::unique_ptr<PHY_instance>            create(const char* phy_class,
                                           PHY_module& phy_module,
                                           yaml::node  node_config);
    
    /**
     * @brief Get list of all registered PHY types
     * @return Vector of type name strings
     */
    static std::vector<const char*> get_registered_types();
private:
};

/// Function pointer type for PHY creation functions
typedef std::unique_ptr<PHY_instance> (*phy_create_func)(PHY_module& phy_module,
                                         yaml::node  node_config);

/**
 * @brief Structure holding PHY creator registration information
 *
 * This structure maintains a singly-linked list of registered PHY types
 * via the next pointer.
 */
struct phy_creator
{
    const char*     type_name;      ///< Name of the PHY type
    phy_create_func create_func;    ///< Function to create instance
    phy_creator*    next;           ///< Next creator in the linked list
};


} // namespace nv

#endif // !defined(NV_PHY_FACTORY_HPP_INCLUDED_)
