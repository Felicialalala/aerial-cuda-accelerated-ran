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

#include "ru_emulator.hpp"
#include "fh.hpp"

#ifdef STANDALONE

static inline uint8_t adjustPrbCount(uint16_t numPrbc)
{
    if(numPrbc == ORAN_MAX_PRB_X_SLOT)
    {
        return 0;
    }
    else if (numPrbc > ORAN_MAX_PRB_X_SECTION)
    {
        return ORAN_MAX_PRB_X_SECTION;
    }
    else
    {
        return static_cast<uint8_t>(numPrbc);
    }
}

bool RU_Emulator::has_ul_for_slot(int launch_pattern_slot, int cell_index, ul_tv_object& tv_object)
{
    return tv_object.launch_pattern[launch_pattern_slot].find(cell_index) != tv_object.launch_pattern[launch_pattern_slot].end();
}

void *standalone_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with standalone_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(arg);
    return rue->standalone_core(arg);
}

void* RU_Emulator::standalone_core(void *arg) {
    char threadname[30];
    sprintf(threadname, "%s", __FUNCTION__);
    SET_THREAD_NAME(threadname);
    uint64_t slot_count = 0;
    struct fssId fss{0,0,0};
    int launch_pattern_slot;
    uint64_t prev_slot_time;
    uint64_t next_slot_time;
    uint64_t enqueue_count = 0;
    uint64_t t0,t1,t2,t3,t4;
    uint64_t symbol_prepare_time = 0;
    uint64_t symbol_spin_time = 0;
    uint64_t symbol_enqueue_time = 0;

    re_cons("Start Standalone core");
    wait_ns(2 * NS_X_S);

    prev_slot_time = get_ns();
    next_slot_time = prev_slot_time + opt_tti_us * NS_X_US;

    while(!check_force_quit())
    {
        launch_pattern_slot = fss_to_launch_pattern_slot(fss, launch_pattern_slot_size);
        if (pusch_object.launch_pattern[launch_pattern_slot].size() > 0 || pucch_object.launch_pattern[launch_pattern_slot].size() > 0)
        {
            aerial_fh::CPlaneMsgSendInfo         message_infos[224];
            aerial_fh::CPlaneSectionInfo         section_infos[3584];
            size_t               message_index = 0;
            size_t               section_index = 0;
            size_t               start_section_index = 0;
            uint16_t             ud_comp_hdr = 0;
            uint16_t             section_id = 0;
            for(int symbol_id = 0; symbol_id < ORAN_ALL_SYMBOLS; ++symbol_id)
            {
                symbol_prepare_time = 0;
                symbol_enqueue_time = 0;
                symbol_spin_time = 0;
                for(int cell_index = 0; cell_index < opt_num_cells; ++cell_index)
                {
                    t0 = get_ns();
                    message_index = 0;
                    for(int flow_id = 0; flow_id < cell_configs[cell_index].num_ul_flows; ++flow_id)
                    {
                        if(has_ul_for_slot(launch_pattern_slot, cell_index, pusch_object) || has_ul_for_slot(launch_pattern_slot, cell_index, pucch_object))
                        {
                            aerial_fh::CPlaneMsgSendInfo message_template;
                            aerial_fh::CPlaneSectionInfo section_template;
                            auto& radio_app_hdr             = message_template.section_common_hdr.sect_1_common_hdr.radioAppHdr;
                            radio_app_hdr.payloadVersion    = ORAN_DEF_PAYLOAD_VERSION;
                            radio_app_hdr.filterIndex       = ORAN_DEF_FILTER_INDEX;
                            radio_app_hdr.frameId           = fss.frameId;
                            radio_app_hdr.subframeId        = fss.subframeId;
                            radio_app_hdr.slotId            = fss.slotId;
                            radio_app_hdr.sectionType       = ORAN_CMSG_SECTION_TYPE_1;
                            radio_app_hdr.dataDirection     = DIRECTION_UPLINK;
                            radio_app_hdr.startSymbolId     = symbol_id;
                            section_index = 0;

                            if(has_ul_for_slot(launch_pattern_slot, cell_index, pusch_object))
                            {
                                radio_app_hdr.numberOfSections  = 0;
                                message_template.sections = &section_infos[section_index];

                                auto& tv_info = pusch_object.tv_info[pusch_object.launch_pattern[launch_pattern_slot][cell_index]];
                                for(int i = 0; i < tv_info.pdu_infos.size(); ++i)
                                {
                                    pdu_info& pdu = tv_info.pdu_infos[i];
                                    if(symbol_id < pdu.startSym || symbol_id >= pdu.startSym + pdu.numSym)
                                    {
                                        continue;
                                    }
                                    auto &section_info  = section_template.sect_1;
                                    section_info.rb     = false;
                                    section_info.symInc = false;
                                    section_info.ef     = 0;
                                    section_info.startPrbc      = pdu.startPrb;
                                    section_info.numPrbc        = adjustPrbCount(pdu.numPrb);
                                    section_info.reMask         = 0;
                                    section_info.numSymbol      = 1;
                                    section_info.sectionId      = i;
                                    radio_app_hdr.filterIndex   = 0;
                                    ++radio_app_hdr.numberOfSections;
                                    memcpy(&section_infos[section_index++], &section_template, sizeof(section_template));
                                }
                                message_template.flow = ul_peer_flow_map[cell_index][flow_id];
                                if(radio_app_hdr.numberOfSections > 0)
                                    memcpy(&message_infos[message_index++], &message_template, sizeof(message_template));
                            }

                            if(has_ul_for_slot(launch_pattern_slot, cell_index, pucch_object))
                            {
                                radio_app_hdr.numberOfSections  = 0;
                                message_template.sections = &section_infos[section_index];
                                auto& tv_info = pucch_object.tv_info[pucch_object.launch_pattern[launch_pattern_slot][cell_index]];
                                for(int i = 0; i < tv_info.pdu_infos.size(); ++i)
                                {
                                    pdu_info& pdu = tv_info.pdu_infos[i];
                                    if(symbol_id < pdu.startSym || symbol_id >= pdu.startSym + pdu.numSym)
                                    {
                                        continue;
                                    }
                                    auto &section_info  = section_template.sect_1;
                                    section_info.rb     = false;
                                    section_info.symInc = false;
                                    section_info.ef     = 0;
                                    section_info.startPrbc      = pdu.startPrb;
                                    section_info.numPrbc        = adjustPrbCount(pdu.numPrb);
                                    section_info.reMask         = 0;
                                    section_info.numSymbol      = 1;
                                    section_info.sectionId      = i;
                                    radio_app_hdr.filterIndex   = 0;
                                    ++radio_app_hdr.numberOfSections;
                                    memcpy(&section_infos[section_index++], &section_template, sizeof(section_template));
                                }
                                message_template.flow = ul_peer_flow_map[cell_index][flow_id];
                                if(radio_app_hdr.numberOfSections > 0)
                                    memcpy(&message_infos[message_index++], &message_template, sizeof(message_template));
                            }
                        }
                    }
                    t1 = get_ns();
                    if(message_index > 0)
                    {
                        aerial_fh::TxRequestHandle tx_request;
                        aerial_fh::prepare_cplane(peer_list[0], const_cast<const aerial_fh::CPlaneMsgSendInfo*>(&message_infos[0]), message_index, &tx_request);
                        t2 = get_ns();
                        while(get_ns() < prev_slot_time + symbol_id * 35 * NS_X_US)
                        {
                            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                                __asm__ __volatile__ ("");
                        }
                        t3 = get_ns();
                        aerial_fh::ring_enqueue_bulk_tx_request_cplane_mbufs(standalone_c_plane_rings[cell_index], tx_request, peer_list[0], message_index);
                        enqueue_count += message_index;
                    }
                    t4 = get_ns();
                    symbol_prepare_time += t2-t0;
                    symbol_enqueue_time += t4-t1;
                    symbol_spin_time += t3-t2;
                }
                re_info("Standalone: symbol {} prepare time {} spin time {} enqueue time {}", symbol_id, symbol_prepare_time, symbol_spin_time, symbol_enqueue_time);

            }
        }

        while(get_ns() < next_slot_time)
        {
            for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt)
                __asm__ __volatile__ ("");
        }
        prev_slot_time = next_slot_time;
        next_slot_time += opt_tti_us * NS_X_US;

        increment_fss(fss, max_slot_id);

        slot_count++;
        if(slot_count == 5)
        {
            return nullptr;
        }
    }
    usleep(2000000);
    return nullptr;
}

#endif