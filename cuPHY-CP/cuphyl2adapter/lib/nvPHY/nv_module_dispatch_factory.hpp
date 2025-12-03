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

#if !defined(NV_MODULE_DISPATCH_FACTORY_HPP_INCLUDED_)
#define NV_MODULE_DISPATCH_FACTORY_HPP_INCLUDED_

#include "yaml.hpp"
#include <vector>

namespace nv
{

class PHY_module_dispatch;
struct module_dispatch_creator;


/**
 * @brief Factory class for creating PHY module dispatcher instances
 *
 * The module_dispatch_factory provides a registry-based factory pattern for
 * creating different types of PHY module dispatchers at runtime.
 */
class module_dispatch_factory
{
public:
    /**
     * @brief Register a new module dispatcher type with the factory
     * @param c Reference to the module_dispatch_creator containing type information
     */
    static void                     register_type(module_dispatch_creator& c);
    
    /**
     * @brief Create a PHY module dispatcher instance
     * @param msg_type Name of the message type to dispatch
     * @return Pointer to the created PHY module dispatcher
     */
    static PHY_module_dispatch*     create(const char* msg_type);
    
    /**
     * @brief Get list of all registered module dispatcher types
     * @return Vector of type name strings
     */
    static std::vector<const char*> get_registered_types();
private:
};

/// Function pointer type for module dispatcher creation functions
typedef PHY_module_dispatch* (*module_dispatch_create_func)();

/**
 * @brief Structure holding module dispatcher creator registration information
 *
 * This structure maintains a singly-linked list of registered module
 * dispatcher types via the next pointer.
 */
struct module_dispatch_creator
{
    const char*                 type_name;      ///< Name of the dispatcher type
    module_dispatch_create_func create_func;    ///< Function to create instance
    module_dispatch_creator*    next;           ///< Next creator in the linked list
};


} // namespace nv

#endif // !defined(NV_MODULE_DISPATCH_FACTORY_HPP_INCLUDED_)
