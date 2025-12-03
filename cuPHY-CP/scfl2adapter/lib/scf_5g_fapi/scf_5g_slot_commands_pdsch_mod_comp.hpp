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
#include "nv_phy_fapi_msg_common.hpp"
#include "nvlog_fmt.hpp"
#include "scf_5g_slot_commands_common.hpp"
#include "scf_5g_slot_commands_mod_comp.hpp"


namespace scf_5g_fapi {

    using PdschFunc = void (*)(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType);
    using PdschCsirsFunc = void (*)(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids);
    void handleNewPdschSegment(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType);
    void handleNewPdschCsirsSegment(uint8_t tempPdschSym, uint8_t tempPdschNum, const pm_weight_map_t & pm_map, pdsch_fh_prepare_params& pdsch_fh_param, nv::slot_detail_t* slot_detail, ru_type ru, channel_type chType, uint8_t num_csirs_eaxcids);

};
