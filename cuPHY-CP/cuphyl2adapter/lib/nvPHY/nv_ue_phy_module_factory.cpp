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

#include "nv_ue_phy_module_factory.hpp"

namespace {
    /**
     * @brief Pointer to the first element of a singly-linked list of UE PHY module creators
     */
    nv::ue_phy_module_creator* creator_list_head = nullptr;
}
namespace nv {

    /**
     * @brief Register a UE PHY module type with the factory
     *
     * Adds the given ue_phy_module_creator to the end of the singly-linked list
     * of registered UE PHY module types.
     *
     * @param c Reference to the ue_phy_module_creator to register
     */
    void ue_phy_module_factory::register_type(ue_phy_module_creator &c) {
        ue_phy_module_creator** list = &creator_list_head;
        while(*list) {
            list = &((*list)->next);
        }
        *list = &c;
    }

    /**
     * @brief Create a UE PHY module instance based on configuration
     *
     * Reads the "class" field from the YAML configuration to determine which
     * registered UE PHY module type to instantiate.
     *
     * @param config YAML configuration node containing class information
     * @return Pointer to the created UEPHY_Module instance
     * @throws std::runtime_error if no matching creator is found
     */
    UEPHY_Module* ue_phy_module_factory::create(yaml::node config) {

        std::string class_type = config["class"].as<std::string>();
        ue_phy_module_creator* create_list = creator_list_head;

        while (create_list)
        {
            if(0 == strcmp(class_type.c_str(), create_list->type_name)) {
                return create_list->create_func(config);
            }
             create_list = create_list->next;
        }
        throw std::runtime_error("nv::ue_phy_module_factory no registered creator for msg_type '"
                                + class_type 
                                + std::string("'"));
    }

    /**
     * @brief Get list of all registered UE PHY module types
     *
     * Traverses the linked list of registered creators and collects all
     * type names into a vector.
     *
     * @return Vector containing pointers to type name strings
     */
    std::vector<const char*> ue_phy_module_factory::get_registered_types() {
        std::vector<const char*> list;
        ue_phy_module_creator* reg_list = creator_list_head;
        while (reg_list)
        {
            list.push_back(reg_list->type_name);
            reg_list = reg_list->next;
        }
        return list;
    }
}