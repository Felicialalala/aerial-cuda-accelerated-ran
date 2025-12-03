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

#include "slot_command/slot_command.hpp"
#include "scf_5g_fapi.h"
#include "nv_phy_fapi_msg_common.hpp"
#include "nvlog_fmt.hpp"
#include "scf_5g_slot_commands_common.hpp"
#include "nv_phy_limit_errors.hpp"


namespace scf_5g_fapi {

#ifdef ENABLE_L2_SLT_RSP
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode,
        int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
        pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled, nv::pdcch_limit_error_t* pdcch_error);
#else
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode,
        int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
        pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
#endif
}