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

#if !defined(NV_UE_PHY_MODULE_FACTORY_HPP_INCLUDED_)
#define NV_UE_PHY_MODULE_FACTORY_HPP_INCLUDED_

#include "nv_ue_phy_module.hpp"

#include <vector>
namespace nv {

    struct ue_phy_module_creator;

    /**
     * @brief Factory class for creating UE PHY module instances
     *
     * Provides a registry-based factory pattern for creating different types
     * of UE PHY module implementations at runtime.
     */
    class ue_phy_module_factory {
        public:
        /**
         * @brief Register a new UE PHY module type with the factory
         * @param c Reference to the ue_phy_module_creator containing type information
         */
        static void register_type(ue_phy_module_creator &c);
        
        /**
         * @brief Create a UE PHY module instance based on configuration
         * @param node_config YAML configuration node
         * @return Pointer to the created UEPHY_Module instance
         */
        static UEPHY_Module* create(yaml::node node_config);
        
        /**
         * @brief Get list of all registered UE PHY module types
         * @return Vector of type name strings
         */
        static std::vector<const char*> get_registered_types();
    };

/// Function pointer type for UE PHY module creation functions
typedef UEPHY_Module* (*ue_phy_module_create_func)(yaml::node node_config);

/**
 * @brief Structure holding UE PHY module creator registration information
 *
 * Maintains a singly-linked list of registered UE PHY module types.
 */
struct ue_phy_module_creator {
    const char* type_name;                    ///< Name of the UE PHY module type
    ue_phy_module_create_func create_func;    ///< Function to create instance
    ue_phy_module_creator* next;              ///< Next creator in the linked list
};
}
#endif //NV_UE_PHY_MODULE_FACTORY_HPP_INCLUDED_