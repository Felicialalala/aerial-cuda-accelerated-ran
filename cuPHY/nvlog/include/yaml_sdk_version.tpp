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

#ifndef YAML_SDK_VERSION_TPP
#define YAML_SDK_VERSION_TPP

#include "fmt/format.h"
#include <type_traits>

#ifndef AERIAL_SDK_VERSION
#error AERIAL_SDK_VERSION is not defined, please pass -DAERIAL_SDK_VERSION=...
#endif // AERIAL_SDK_VERSION

namespace aerial {

// Templated version of check_yaml_version
#if AERIAL_HAS_CONCEPTS
template<YamlNodeType NodeType>
#else
template<typename NodeType>
#endif
void check_yaml_version(const NodeType& root, const std::string_view filename)
{
#if !AERIAL_HAS_CONCEPTS
    static_assert(is_yaml_node<NodeType>::value, 
        "NodeType must be either YAML::Node or yaml::node");
#endif

    try
    {
        // Use a lambda with explicit captures to determine has_key and potentially retrieve version
        const auto [has_key, version] = [&root](const char* param_name) -> std::pair<bool, std::string> {
            if constexpr (std::is_same_v<NodeType, YAML::Node>) {
                const bool exists = static_cast<bool>(root[param_name]);
                if (exists) {
                    return {exists, root[param_name].template as<std::string>()};
                }
                return {exists, {}};
            } else {
                const bool exists = root.has_key(param_name);
                if (exists) {
                    return {exists, root[param_name].operator std::string()};
                }
                return {exists, {}};
            }
        }(YAML_PARAM_AERIAL_SDK_VERSION);

        if (has_key) {
            // A valid version is either:
            // 1. Marked as an internal version, for internal builds
            // 2. Has a matching version as the -DAERIAL_SDK_VERSION
            // So basically, if the version is not matching what we pass in compile time, it's a failure.
            if (version != AERIAL_SDK_VERSION) {
                throw std::invalid_argument(fmt::format(
                    R"(Error, Version mismatch in YAML. CUPHY config version: "{}", vs compiled version: "{}")",
                    version,
                    AERIAL_SDK_VERSION));
            }
        } else {
            throw std::invalid_argument(fmt::format(
                R"(Failed to locate aerial_sdk_version attribute: "{}", with the expected, compiled value "{}"")",
                YAML_PARAM_AERIAL_SDK_VERSION,
                AERIAL_SDK_VERSION));
        }
    }
    catch (const std::exception& e)
    {
        throw std::invalid_argument(fmt::format(
            R"(cuPHY driver config version reading error. what: {}. YAML: "{}")",
            e.what(),
            filename));
    }
}

}// namespace aerial

#endif // YAML_SDK_VERSION_TPP 
