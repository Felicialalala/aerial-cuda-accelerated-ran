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

#ifndef __SCF_5G_FAPI_UL_VALIDATE__
#define __SCF_5G_FAPI_UL_VALIDATE__

#include "nvlog.hpp"
#include "scf_5g_fapi.h"
#include "cuphy.h"
#include "slot_command/slot_command.hpp"
#include "nv_phy_limit_errors.hpp"
using namespace slot_command_api;

int validate_ul_tti_req(scf_fapi_ul_tti_req_t& msg, uint64_t validate_mask);
int validate_pusch_pdu(scf_fapi_pusch_pdu_t& pdu_info);
int validate_ul_dl_bfw_cvi_req (scf_fapi_ul_bfw_cvi_request_t& msg,  uint8_t is_ul_bfw, uint64_t validate_mask);

#ifdef ENABLE_L2_SLT_RSP
int validate_pusch_pdu_l1_limits(const scf_fapi_pusch_pdu_t& pdu, nv::pusch_limit_error_t& error);
int validate_srs_pdu_l1_limits(const scf_fapi_srs_pdu_t& pdu, nv::srs_limit_error_t& error);
int validate_pucch_pdu_l1_limits(const scf_fapi_pucch_pdu_t& pdu, nv::pucch_limit_error_t& error);
int validate_prach_pdu_l1_limits(const scf_fapi_prach_pdu_t& pdu, nv::prach_limit_error_t& error);
using error_pair = std::pair<uint16_t, uint8_t>;
error_pair check_ul_tti_l1_limit_errors(const nv::slot_limit_cell_error_t& cell_error, const nv::slot_limit_group_error_t& group_error);
#endif //  End ENABLE_L2_SLT_RSP

#endif //  End _SCF_5G_FAPI_UL_VALIDATE__
