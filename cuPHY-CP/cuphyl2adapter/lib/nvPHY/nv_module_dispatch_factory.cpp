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

#include "nv_module_dispatch_factory.hpp"

namespace
{

/**
 * @brief Pointer to the first element of a singly-linked list of objects in
 * global memory
 *
 * The linked list is maintained via the module_dispatch_factory::register_type() function.
 */
nv::module_dispatch_creator* creator_list_head = nullptr;

} // namespace

namespace nv
{

/**
 * @brief Register a module dispatcher type with the factory
 *
 * Adds the given module_dispatch_creator to the end of the singly-linked list
 * of registered module dispatcher types.
 *
 * @param c Reference to the module_dispatch_creator to register
 */
void module_dispatch_factory::register_type(module_dispatch_creator& c)
{
    nv::module_dispatch_creator** pnext = &creator_list_head;
    while(*pnext)
    {
        pnext = &((*pnext)->next);
    }
    *pnext = &c;
}
    
/**
 * @brief Create a PHY module dispatcher instance based on message type
 *
 * Searches through the registered creators for a matching type name
 * and invokes the creation function.
 *
 * @param msg_type Name of the message type to dispatch
 * @return Pointer to the created PHY module dispatcher
 * @throws std::runtime_error if no matching creator is found
 */
PHY_module_dispatch* module_dispatch_factory::create(const char* msg_type)
{
    // Check each element of the registered msg type list for a string
    // match, and create an object if one is found
    nv::module_dispatch_creator* c = creator_list_head;
    while(c)
    {
        if(0 == strcmp(msg_type, c->type_name))
        {
            return c->create_func();
        }
        c = c->next;
    }
    throw std::runtime_error(std::string("nv::module_dispatch_factory: no registered creator for msg type '") +
                             msg_type +
                             std::string("'"));
}

/**
 * @brief Get list of all registered module dispatcher types
 *
 * Traverses the linked list of registered creators and collects all
 * type names into a vector.
 *
 * @return Vector containing pointers to type name strings
 */
std::vector<const char*> module_dispatch_factory::get_registered_types()
{
    std::vector<const char*> v;
    nv::module_dispatch_creator* c = creator_list_head;
    while(c)
    {
        v.push_back(c->type_name);
        c = c->next;
    }
    return v;
}
    
} // namespace nv
