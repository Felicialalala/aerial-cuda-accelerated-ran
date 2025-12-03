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

#include "nv_phy_factory.hpp"

namespace
{

/**
 * @brief Pointer to the first element of a singly-linked list of objects in
 * global memory
 *
 * The linked list is maintained via the phy_factory::register_type() function.
 */
nv::phy_creator* creator_list_head = nullptr;

} // namespace

namespace nv
{

/**
 * @brief Register a PHY type with the factory
 *
 * Adds the given phy_creator to the end of the singly-linked list of
 * registered PHY types.
 *
 * @param c Reference to the phy_creator to register
 */
void phy_factory::register_type(phy_creator& c)
{
    nv::phy_creator** pnext = &creator_list_head;
    while(*pnext)
    {
        pnext = &((*pnext)->next);
    }
    *pnext = &c;
}
    
/**
 * @brief Create a PHY instance based on class name
 *
 * Searches through the registered creators for a matching type name
 * and invokes the creation function.
 *
 * @param phy_class Name of the PHY class to create
 * @param phy_module Reference to the parent PHY module
 * @param node_config YAML configuration node
 * @return Unique pointer to the created PHY instance
 * @throws std::runtime_error if no matching creator is found
 */
std::unique_ptr<PHY_instance> phy_factory::create(const char* phy_class,
                                  PHY_module& phy_module,
                                  yaml::node  node_config)
{
    // Check each element of the registered msg type list for a string
    // match, and create an object if one is found
    nv::phy_creator* c = creator_list_head;
    while(c)
    {
        if(0 == strcmp(phy_class, c->type_name))
        {
            return c->create_func(phy_module, node_config);
        }
        c = c->next;
    }
    throw std::runtime_error(std::string("nv::phy_factory: no registered creator for phy class '") +
                             phy_class +
                             std::string("'"));
}

/**
 * @brief Get list of all registered PHY types
 *
 * Traverses the linked list of registered creators and collects all
 * type names into a vector.
 *
 * @return Vector containing pointers to type name strings
 */
std::vector<const char*> phy_factory::get_registered_types()
{
    std::vector<const char*> v;
    nv::phy_creator* c = creator_list_head;
    while(c)
    {
        v.push_back(c->type_name);
        c = c->next;
    }
    return v;
}
    
} // namespace nv
