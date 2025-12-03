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

#pragma once

#include "scf_5g_slot_commands_common.hpp"
#include "scf_5g_slot_commands_mod_comp.hpp"

namespace scf_5g_fapi {

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, scf_fapi_ssb_pdu_t& cmd, int32_t cell_index, slot_indication & slotinfo, nv::phy_config& cell_params, uint8_t l_max, const uint16_t* lmax_symbols, nv::phy_config_option& config_options, pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
    void update_pm_weights_ssb_cuphy(cell_group_command* cell_grp_cmd, cuphyPerSsBlockDynPrms_t& block, cuphyPerCellSsbDynPrms_t& ssb_cell_params, scf_fapi_tx_precoding_beamforming_t& pdu, nv::phy_config_option& config_options,
        pm_group* prec_group, pm_weight_map_t& pm_map, cell_sub_command& cell_cmd, int32_t cell_index, nv::slot_detail_t* slot_detail, nv::phy_config& cell_params, bool mmimo_enabled);

}