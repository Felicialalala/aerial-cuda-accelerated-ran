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

#include "fh_generator.hpp"
#include "oran_slot_iterator.hpp"
#include "worker.hpp"
#include "utils.hpp"

using namespace fh_gen;

void FhGenerator::calculate_expected_packet_counts()
{
    memset((void*)pkt_exp.count, 0, sizeof(pkt_exp.count));
    CPlaneTXs ulcplane;
    CPlaneTXs dlcplane;
    UPlaneTXs uluplane;
    UPlaneTXs dluplane;
    auto slot_count = yaml_parser_.slot_count();
    auto& peer_infos = yaml_parser_.get_peer_info();
    
    for(int i = 0; i < peer_infos.size(); ++i)
    {
        ulcplane[i].resize(slot_count);
        dlcplane[i].resize(slot_count);
        uluplane[i].resize(slot_count);
        dluplane[i].resize(slot_count);
    }
    auto flow_infos = yaml_parser_.get_flow_info();

    for(const auto& flow_info: flow_infos)
    {
        for(auto& cplane_tx_info : flow_info.cplane_tx_info)
        {
            auto cell_id = flow_info.cell_id;
            auto& peer_info = peer_infos[cell_id];
            auto eAxC    = flow_info.info.eAxC;
            auto vlan_id = flow_info.info.vlan_tag.vid;
            auto slot_id = cplane_tx_info.slot_id;
            auto         symbol_id        = cplane_tx_info.symbol_id;
            oran_pkt_dir direction        = std::string("UL") == cplane_tx_info.direction ? DIRECTION_UPLINK : DIRECTION_DOWNLINK;
            auto         section_count    = cplane_tx_info.section_count;
            CPlaneTX cplane{
                .flow          = cplane_flows_[cell_id][eAxC],
                .eAxC          = eAxC,
                .vlan_id       = vlan_id,
                .slot_offset   = 0,
                .section_count = section_count,
                .symbol_id     = symbol_id,
                .direction     = direction,
            };
            for(auto& section_info: cplane_tx_info.section_list)
            {
                CPlaneTXSection section {
                    .start_sym = section_info.start_sym,
                    .num_sym = section_info.num_sym,
                    .start_prbc = section_info.start_prbc,
                    .num_prbc = section_info.num_prbc
                };
                cplane.section_list.push_back(section);
            }
            if(direction == DIRECTION_UPLINK)
            {
                ulcplane[cell_id][slot_id].push_back(cplane);
            }
            else
            {
                dlcplane[cell_id][slot_id].push_back(cplane);
            }
        }

        for(const auto& uplane_rx_info : flow_info.uplane_rx_info)
        {
            auto cell_id        = flow_info.cell_id;
            auto& peer_info     = peer_infos[cell_id];
            auto eAxC           = flow_info.info.eAxC;
            auto vlan_id        = flow_info.info.vlan_tag.vid;
            auto slot_id        = uplane_rx_info.slot_id;
            auto symbol_id      = uplane_rx_info.symbol_id;
            auto iq_data_buffer = uplane_rx_info.iq_data_buffer;
            
            UPlaneTX uplane{
                .flow           = uplane_flows_[cell_id][eAxC],
                .eAxC           = eAxC,
                .vlan_id        = vlan_id,
                .slot_offset    = 0,
                .iq_data_buffer = iq_data_buffers_[iq_data_buffer].first,
                .symbol_id      = symbol_id,
                .section_id     = uplane_rx_info.section_id,
            };

            for(auto& section_info: uplane_rx_info.section_list)
            {
                UPlaneTXSection section {
                    .start_prbu = section_info.start_prbu,
                    .num_prbu = section_info.num_prbu
                };
                uplane.section_list.push_back(section);
            }
            uluplane[cell_id][slot_id].push_back(uplane);
        }

        for(const auto& uplane_tx_info : flow_info.uplane_tx_info)
        {
            auto cell_id        = flow_info.cell_id;
            auto& peer_info     = peer_infos[cell_id];
            auto eAxC           = flow_info.info.eAxC;
            auto vlan_id        = flow_info.info.vlan_tag.vid;
            auto slot_id        = uplane_tx_info.slot_id;
            auto symbol_id      = uplane_tx_info.symbol_id;
            auto iq_data_buffer = uplane_tx_info.iq_data_buffer;
            
            UPlaneTX uplane{
                .flow           = uplane_flows_[cell_id][eAxC],
                .eAxC           = eAxC,
                .vlan_id        = vlan_id,
                .slot_offset    = 0,
                .iq_data_buffer = iq_data_buffers_[iq_data_buffer].first,
                .symbol_id      = symbol_id,
                .section_id     = uplane_tx_info.section_id,
            };

            for(auto& section_info: uplane_tx_info.section_list)
            {
                UPlaneTXSection section {
                    .start_prbu = section_info.start_prbu,
                    .num_prbu = section_info.num_prbu
                };
                uplane.section_list.push_back(section);
            }
            dluplane[cell_id][slot_id].push_back(uplane);
        }
    }

    OranSlotIterator                   oran_slot_iterator{{0,0,0}};
    for(int peer_index = 0; peer_index < peer_infos.size(); ++peer_index)
    {
        auto& peer_info       = peer_infos[peer_index];
        auto peer_id          = peer_info.info.id;
        auto peer             = peers_[peer_id];
        for(int slot_index_tdd = 0; slot_index_tdd < slot_count; ++slot_index_tdd)
        {
            auto slot_iterator = oran_slot_iterator.get_next();

            for(auto const& tx : dlcplane[peer_index][slot_index_tdd])
            {
                std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount> cplane_msg_infos;
                std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount> cplane_sections;
                int cplane_msg_infos_num = 0;
                int cplane_sections_num = 0;
                prepare_c_plane(slot_iterator, cplane_msg_infos, cplane_sections, tx, cplane_msg_infos_num, cplane_sections_num, 0);
                auto num_pkts = aerial_fh::prepare_cplane_count_packets(peer, &cplane_msg_infos[0], cplane_msg_infos_num);
                pkt_exp.count[DLPacketCounterType::DLC][peer_index][slot_index_tdd][tx.symbol_id] += num_pkts;
            }
            for(auto const& tx : ulcplane[peer_index][slot_index_tdd])
            {
                std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount> cplane_msg_infos;
                std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount> cplane_sections;
                int cplane_msg_infos_num = 0;
                int cplane_sections_num = 0;
                prepare_c_plane(slot_iterator, cplane_msg_infos, cplane_sections, tx, cplane_msg_infos_num, cplane_sections_num, 0);
                auto num_pkts = aerial_fh::prepare_cplane_count_packets(peer, &cplane_msg_infos[0], cplane_msg_infos_num);
                pkt_exp.count[DLPacketCounterType::ULC][peer_index][slot_index_tdd][tx.symbol_id] += num_pkts;
            }

            for(auto const& tx : uluplane[peer_index][slot_index_tdd])
            {
                aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
                prepare_u_plane(slot_iterator, uplane_msg, tx, 0, 0, nullptr, 0, 0);
                auto num_pkts = aerial_fh::prepare_uplane_count_packets(peer, &uplane_msg);
                pkt_exp.count[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULU][peer_index][slot_index_tdd][tx.symbol_id] += num_pkts;
            }

            for(auto const& tx : dluplane[peer_index][slot_index_tdd])
            {
                for(int i = 0; i < tx.section_list.size(); ++i)
                {
                    aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
                    prepare_u_plane_single_section(slot_iterator, uplane_msg, tx, 0, 0, nullptr, 0, 0, i);
                    auto num_pkts = aerial_fh::prepare_uplane_count_packets(peer, &uplane_msg);
                    pkt_exp.count[DLPacketCounterType::DLU][peer_index][slot_index_tdd][tx.symbol_id] += num_pkts;
                }
            }
        }
    }

    int slot_index_tdd = 0;
    for(int peer_index = 0; peer_index < peer_infos.size(); ++peer_index)
    {
        for(int i = 0; i < yaml_parser_.test_slots(); ++i)
        {
            for(int sym_id = 0; sym_id < ORAN_ALL_SYMBOLS; ++sym_id)
            {
                pkt_exp.total[DLPacketCounterType::DLC][peer_index] += pkt_exp.count[DLPacketCounterType::DLC][peer_index][slot_index_tdd][sym_id];
                pkt_exp.total[DLPacketCounterType::DLU][peer_index] += pkt_exp.count[DLPacketCounterType::DLU][peer_index][slot_index_tdd][sym_id];
                pkt_exp.total[DLPacketCounterType::ULC][peer_index] += pkt_exp.count[DLPacketCounterType::ULC][peer_index][slot_index_tdd][sym_id];
                pkt_exp.total[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULU][peer_index] += pkt_exp.count[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULU][peer_index][slot_index_tdd][sym_id];
            }
            slot_index_tdd = (slot_index_tdd + 1) % slot_count;
        }
        NVLOGC_FMT(TAG, "Cell {} total DLC {} DLU {} ULC {} ULU {} ", peer_index,
            pkt_exp.total[DLPacketCounterType::DLC][peer_index],
            pkt_exp.total[DLPacketCounterType::DLU][peer_index],
            pkt_exp.total[DLPacketCounterType::ULC][peer_index],
            pkt_exp.total[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULU][peer_index]
        );
    }
}

bool FhGenerator::check_packet_count_pass_criteria()
{
    bool pass = true;
    if(fh_gen_type_ == FhGenType::DU)
    {
        // Check packet counts
        for(int peer_index = 0; peer_index < yaml_parser_.get_peer_info().size(); ++peer_index)
        {
            auto expected_packet_count = pkt_exp.total[DLPacketCounterType::DLCounterTypeMax + ULPacketCounterType::ULU][peer_index];
            auto received_packet_count = ulu_stats.get_cell_total_count(peer_index);
            if(expected_packet_count != received_packet_count)
            {
                NVLOGC_FMT(TAG, "Cell {} UL U expected {} received {}", peer_index, expected_packet_count, received_packet_count);
                pass = false;
            }
            else
            {
                NVLOGI_FMT(TAG, "Cell {} UL U expected {} received {}", peer_index, expected_packet_count, received_packet_count);
            }
        }
    }
    else
    {
        for(int peer_index = 0; peer_index < yaml_parser_.get_peer_info().size(); ++peer_index)
        {
            auto type = DLPacketCounterType::DLC;
            for(int type = 0; type < DLPacketCounterType::DLCounterTypeMax; ++type)
            {
                auto expected_packet_count = pkt_exp.total[type][peer_index];
                uint64_t received_packet_count = get_cell_total_count(peer_index, type);
                if(expected_packet_count != received_packet_count)
                {
                    NVLOGC_FMT(TAG, "Cell {} {} expected {} received {}", peer_index, packet_type_to_char_string(type), expected_packet_count, received_packet_count);
                    pass = false;
                }
                else
                {
                    NVLOGI_FMT(TAG, "Cell {} {} expected {} received {}", peer_index, packet_type_to_char_string(type), expected_packet_count, received_packet_count);
                }
            }
        }
    }

    if(pass)
    {
        NVLOGC_FMT(TAG, "All expected packets received");
    }
    else
    {
        NVLOGC_FMT(TAG, "Some packets were dropped, please check above logs");
    }

    return pass;
}
