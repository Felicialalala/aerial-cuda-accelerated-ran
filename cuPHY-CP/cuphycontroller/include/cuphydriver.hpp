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

#ifndef PHYDRIVER_HPP
#define PHYDRIVER_HPP

#include "cuphydriver_api.hpp"
#include "yamlparser.hpp"
#include <vector>

int pc_init_phydriver(
                phydriver_handle * pdh,
                const context_config& ctx_cfg,
                std::vector<phydriverwrk_handle>& workers_descr
            );

int pc_finalize_phydriver(phydriver_handle pdh);

int pc_standalone_create_cells(phydriver_handle pdh, std::vector<cell_phy_info>& cell_configs);
int pc_start_l1(phydriver_handle pdh);
int pc_standalone_simulate_l2(phydriver_handle pdh, int usec, int num_slots, std::vector<struct slot_command_api::slot_command *> scl, int core, uint32_t workers_sched_priority);

#endif
