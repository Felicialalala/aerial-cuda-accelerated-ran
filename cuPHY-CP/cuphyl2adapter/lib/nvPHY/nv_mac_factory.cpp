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

#include "nv_mac_factory.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 2) // "L2A.MACFACT"

namespace
{

/**
 * @brief Pointer to the first element of a singly-linked list of objects in
 * global memory
 *
 * The linked list is maintained via the mac_factory::register_type() function.
 */
nv::mac_creator* creator_list_head = nullptr;

} // namespace

namespace nv
{

/**
 * @brief Constructor for mac_creator (commented out)
 *
 * Note: global constructors in static libraries are often removed by
 * the linker, so auto-registration is fragile. We will explicitly
 * initialize instead.
 */
//mac_creator::mac_creator(const char* n, mac_create_func f) :
//    type_name(n),
//    create_func(f),
//    next(nullptr)
//{
//    NVLOGI_FMT(TAG, "mac_creator::mac_creator() ({})", n);
//    mac_factory::register_type(*this);
//}

/**
 * @brief Register a MAC type with the factory
 *
 * Adds the given mac_creator to the end of the singly-linked list of
 * registered MAC types.
 *
 * @param c Reference to the mac_creator to register
 */
void mac_factory::register_type(mac_creator& c)
{
    //printf("mac_factory: registering type %s\n", c.type_name);
    nv::mac_creator** pnext = &creator_list_head;
    while(*pnext)
    {
        pnext = &((*pnext)->next);
    }
    *pnext = &c;
}
    
/**
 * @brief Create a MAC instance based on configuration
 *
 * Reads the "class" field from the YAML configuration to determine which
 * registered MAC type to instantiate. Searches through the registered
 * creators for a matching type name and invokes the creation function.
 *
 * @param node_config YAML configuration node containing class information
 * @param cell_num Cell number for this MAC instance
 * @return Pointer to the created MAC instance
 * @throws std::runtime_error if no matching creator is found
 */
mac* mac_factory::create(yaml::node node_config, uint32_t cell_num)
{
    // Get the class field from the YAML node
    std::string msg_type = node_config["class"].as<std::string>();
    
    // Check each element of the registered msg type list for a string
    // match, and create an object if one is found
    nv::mac_creator* c = creator_list_head;
    while(c)
    {
        if(0 == strcmp(msg_type.c_str(), c->type_name))
        {
            return c->create_func(node_config, cell_num);
        }
        c = c->next;
    }
    throw std::runtime_error(std::string("nv::mac_factory: no registered creator for msg_type '") +
                             msg_type +
                             std::string("'"));
}

/**
 * @brief Get list of all registered MAC types
 *
 * Traverses the linked list of registered creators and collects all
 * type names into a vector.
 *
 * @return Vector containing pointers to type name strings
 */
std::vector<const char*> mac_factory::get_registered_types()
{
    std::vector<const char*> v;
    nv::mac_creator* c = creator_list_head;
    while(c)
    {
        v.push_back(c->type_name);
        c = c->next;
    }
    return v;
}
    
} // namespace nv
