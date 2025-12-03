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

#ifndef __SCF_5G_FAPI_DL_VALIDATE__
#define __SCF_5G_FAPI_DL_VALIDATE__

#include "nvlog.hpp"
#include "scf_5g_fapi.h"
#include "cuphy.h"
#include "slot_command/slot_command.hpp"
#include "nv_phy_limit_errors.hpp"
using namespace slot_command_api;

int validate_dl_tti_req(scf_fapi_dl_tti_req_t& msg, uint64_t validate_mask, bool& pdsch_pdu_check);
int validate_pdcch_pdu(scf_fapi_pdcch_pdu_t& pdcch_pdu);
#ifdef ENABLE_L2_SLT_RSP
int validate_pdcch_pdu_l1_limits(const scf_fapi_pdcch_pdu_t& pdu, nv::pdcch_limit_error_t& error);
void update_pdcch_error_contexts(const scf_fapi_dl_dci_t& dci, nv::pdcch_limit_error_t& pdcch_error, const uint8_t& index);
int validate_ssb_pdu_l1_limits(const scf_fapi_ssb_pdu_t& pdu, nv::ssb_pbch_limit_error_t& error);
int validate_csirs_pdu_l1_limits(const scf_fapi_csi_rsi_pdu_t& pdu, nv::csirs_limit_error_t& error);
int validate_pdsch_pdu_l1_limits(const scf_fapi_pdsch_pdu_t& pdu, nv::pdsch_limit_error_t& error, nv::pdsch_pdu_error_ctxts_info_t& pdsch_pdu_error_contexts_info);
using error_pair = std::pair<uint16_t, uint8_t>;
error_pair check_dl_tti_l1_limit_errors(const nv::slot_limit_cell_error_t& cell_error, const nv::slot_limit_group_error_t& group_error);
error_pair check_ul_dci_l1_limit(const nv::slot_limit_cell_error_t& error); 
#endif //  End ENABLE_L2_SLT_RSP

#endif //  End _SCF_5G_FAPI_DL_VALIDATE__
