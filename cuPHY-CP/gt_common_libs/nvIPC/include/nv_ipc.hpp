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

#ifndef NV_IPC_HPP_INCLUDED_
#define NV_IPC_HPP_INCLUDED_

#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "yaml.hpp"

/**
 * Parse YAML configuration node
 *
 * @param[out] cfg Configuration structure to populate
 * @param[in] yaml_node YAML node containing configuration
 * @param[in] module_type Module type for this IPC instance
 * @return 0 on success, -1 on failure
 */
int nv_ipc_parse_yaml_node(nv_ipc_config_t* cfg, yaml::node* yaml_node, nv_ipc_module_t module_type);

#endif /* NV_IPC_HPP_INCLUDED_ */
