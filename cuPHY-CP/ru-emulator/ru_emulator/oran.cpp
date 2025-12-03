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
#include "oran_structs.hpp"
#include <iostream>
using namespace std;

#define OFFSET_AS(b, t, o) ((t)((uint8_t *)b + (o)))

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Dump functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int oran_dump_ethvlan_hdr(struct oran_eth_hdr * ethvlan_hdr)
{
    if(!ethvlan_hdr)
        return RE_EINVAL;

    printf("\n### Ethernet VLAN HDR\n");
    printf("# Src Eth Addr = %02X:%02X:%02X:%02X:%02X:%02X\n",
        ethvlan_hdr->eth_hdr.src_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.src_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.src_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.src_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.src_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.src_addr.addr_bytes[5]
    );

    printf("# Dst Eth Addr = %02X:%02X:%02X:%02X:%02X:%02X\n",
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[5]
    );

    printf("# Eth type= 0x%x\n", ethvlan_hdr->eth_hdr.ether_type);
    printf("# VLAN TCI = 0x%x Eth Proto = 0x%x\n", ethvlan_hdr->vlan_hdr.vlan_tci, ethvlan_hdr->vlan_hdr.eth_proto);

    return RE_OK;
}

int oran_dump_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr)
{
    if(!ecpri_hdr)
        return RE_EINVAL;

    printf("\n### eCPRI HDR\n");
    printf("# ecpriVersion = %x\n", (uint32_t)ecpri_hdr->ecpriVersion);
    printf("# ecpriReserved = %x\n", (uint32_t)ecpri_hdr->ecpriReserved);
    printf("# ecpriConcatenation = %s\n", ((uint32_t)ecpri_hdr->ecpriConcatenation == 0 ? "No" : "Yes"));
    printf("# ecpriMessage = %s\n", ecpri_msgtype_to_string((uint32_t)ecpri_hdr->ecpriMessage));
    printf("# ecpriPayload = %d\n", (uint32_t)ecpri_hdr->ecpriPayload);
    if(ecpri_hdr->ecpriMessage == ECPRI_MSG_TYPE_IQ) printf("# ecpriPcid = %x\n", (uint32_t)ecpri_hdr->ecpriPcid);
    else printf("# ecpriRtcid = %x\n", (uint32_t)ecpri_hdr->ecpriRtcid);
    printf("# ecpriSeqid = %d\n", (uint32_t)ecpri_hdr->ecpriSeqid);
    printf("# ecpriEbit = %x\n", (uint32_t)ecpri_hdr->ecpriEbit);
    printf("# ecpriSubSeqid = %x\n", (uint32_t)ecpri_hdr->ecpriSubSeqid);

    return RE_OK;

}

int oran_dump_cmsg_radio_app_hdr(struct oran_cmsg_radio_app_hdr * cmsg_hdr)
{
    if(!cmsg_hdr)
        return RE_EINVAL;

    printf("\n### C-msg HDR\n");
    printf("# dataDirection = %s\n", oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)cmsg_hdr->dataDirection)));
    printf("# payloadVersion = %x\n", (uint32_t)cmsg_hdr->payloadVersion);
    printf("# filterIndex = %x\n", (uint32_t)cmsg_hdr->filterIndex);
    printf("# frameId = %x\n", (uint32_t)cmsg_hdr->frameId);
    printf("# subframeId = %x\n", (uint32_t)cmsg_hdr->subframeId);
    printf("# slotId = %x\n", (uint32_t)cmsg_hdr->slotId);
    printf("# startSymbolId = %x\n", (uint32_t)cmsg_hdr->startSymbolId);
    printf("# numberOfSections = %x\n", (uint32_t)cmsg_hdr->numberOfSections);
    printf("# sectionType = %x\n", (uint32_t)cmsg_hdr->sectionType);
    return RE_OK;
}

int oran_dump_cmsg_uldl_hdr(struct oran_cmsg_sect1 * cmsg_uldl_hdr)
{
    if(!cmsg_uldl_hdr)
        return RE_EINVAL;

    printf("\n### C-msg ULDL Section HDR\n");
    printf("# Section ID = %d\n", (uint32_t)cmsg_uldl_hdr->sectionId);
    printf("# Resource Block Indicator = %s\n", (cmsg_uldl_hdr->rb == 0 ? "No" : "Yes"));
    printf("# Symbol number increment = %s\n", (cmsg_uldl_hdr->symInc == 0 ? "No" : "Yes"));
    printf("# Start Prbc = %d\n", (uint32_t)cmsg_uldl_hdr->startPrbc);
    printf("# Tot PRBs number = %d\n", (uint32_t)cmsg_uldl_hdr->numPrbc);
    printf("# Resource Element Mask = %x\n", (uint32_t)cmsg_uldl_hdr->reMask);
    printf("# Symbol number = %d\n", (uint32_t)cmsg_uldl_hdr->numSymbol);
    printf("# Extension Flag = %x\n", (uint32_t)cmsg_uldl_hdr->ef);
    printf("# Beam ID = %x\n", (uint32_t)cmsg_uldl_hdr->beamId);

    return RE_OK;
}

int oran_dump_cmsg_hdrs(struct oran_cmsg_uldl_hdrs * cmsg)
{
    if(!cmsg)
        return RE_EINVAL;

    printf("\n============ C-msg Dump ============\n");
    oran_dump_ethvlan_hdr(&(cmsg->ethvlan));
    oran_dump_ecpri_hdr(&(cmsg->ecpri));
    oran_dump_cmsg_radio_app_hdr(&(cmsg->sect1_hdr.radioAppHdr));
    oran_dump_cmsg_uldl_hdr(&(cmsg->sect1_fields));
    printf("\n====================================\n");

    return RE_OK;
}

int oran_dump_custom_sect1_cmsg_cell_info(struct oran_cmsg_uldl_hdrs * cmsg)
{
    if(!cmsg)
        return RE_EINVAL;
    struct oran_eth_hdr * ethvlan_hdr = &(cmsg->ethvlan);
    struct oran_cmsg_radio_app_hdr * cmsg_hdr = &(cmsg->sect1_hdr.radioAppHdr);
    struct oran_ecpri_hdr * ecpri_hdr = &(cmsg->ecpri);

    re_dbg("# F{}S{}S{} Sym {:2d}, Dst Eth = {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}, FlowID {}, dataDirection {}, sectionType = {}, sectionId {}, startSym {}, numSym {}, startPrb {}, numPrb {}",
        (uint32_t)cmsg_hdr->frameId, (uint32_t)cmsg_hdr->subframeId, (uint32_t)cmsg_hdr->slotId, (uint32_t)cmsg_hdr->startSymbolId,
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[5],
        oran_msg_get_flowid((uint8_t*)cmsg), oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)cmsg_hdr->dataDirection)),
        (uint32_t)cmsg_hdr->sectionType,(uint32_t)cmsg->sect1_fields.sectionId,
        (uint16_t)cmsg_hdr->startSymbolId, (uint16_t)cmsg->sect1_fields.numSymbol,
        (uint16_t)cmsg->sect1_fields.startPrbc, (uint16_t)cmsg->sect1_fields.numPrbc
    );
    return RE_OK;
}

int oran_dump_custom_sect3_cmsg_cell_info(struct oran_cmsg_prach_hdrs * cmsg)
{
    if(!cmsg)
        return RE_EINVAL;
    struct oran_eth_hdr * ethvlan_hdr = &(cmsg->ethvlan);
    struct oran_cmsg_radio_app_hdr * cmsg_hdr = &(cmsg->sect3_hdr.radioAppHdr);
    struct oran_ecpri_hdr * ecpri_hdr = &(cmsg->ecpri);

    re_dbg("# F{}S{}S{} Sym {:2d}, Dst Eth = {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}, FlowID {}, dataDirection {}, sectionType = {}, sectionId {}, startSym {}, numSym {}, startPrb {}, numPrb {}",
        (uint32_t)cmsg_hdr->frameId, (uint32_t)cmsg_hdr->subframeId, (uint32_t)cmsg_hdr->slotId, (uint32_t)cmsg_hdr->startSymbolId,
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[0], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[1],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[2], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[3],
        ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[4], ethvlan_hdr->eth_hdr.dst_addr.addr_bytes[5],
        oran_msg_get_flowid((uint8_t*)cmsg), oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)cmsg_hdr->dataDirection)),
        (uint32_t)cmsg_hdr->sectionType,(uint32_t)cmsg->sect3_fields.sectionId,
        (uint16_t)cmsg_hdr->startSymbolId, (uint16_t)cmsg->sect3_fields.numSymbol,
        (uint16_t)cmsg->sect3_fields.startPrbc, (uint16_t)cmsg->sect3_fields.numPrbc
    );
    return RE_OK;
}

int oran_dump_umsg_hdr(struct oran_umsg_iq_hdr * umsg_hdr)
{
    if(!umsg_hdr)
        return RE_EINVAL;

    printf("\n### U-msg HDR\n");
    printf("# dataDirection = %s\n", oran_direction_to_string((enum oran_pkt_dir) ((uint32_t)umsg_hdr->dataDirection)));
    printf("# payloadVersion = %x\n", (uint32_t)umsg_hdr->payloadVersion);
    printf("# filterIndex = %x\n", (uint32_t)umsg_hdr->filterIndex);
    printf("# frameId = %x\n", (uint32_t)umsg_hdr->frameId);
    printf("# subframeId = %x\n", (uint32_t)umsg_hdr->subframeId);
    printf("# slotId = %x\n", (uint32_t)umsg_hdr->slotId);
    printf("# symbolId = %x\n", (uint32_t)umsg_hdr->symbolId);

    return RE_OK;
}

int oran_dump_umsg_iq_hdr(struct oran_u_section_uncompressed * sec_hdr)
{
    if(!sec_hdr)
        return RE_EINVAL;

    printf("\n### U-msg IQ Section HDR\n");
    printf("# Section ID = %d\n", (uint32_t)sec_hdr->sectionId);
    printf("# Resource Block Indicator = %s\n", (sec_hdr->rb == 0 ? "No" : "Yes"));
    printf("# Symbol number increment = %s\n", (sec_hdr->symInc == 0 ? "No" : "Yes"));
    printf("# Start Prbu = %d\n", (uint32_t)sec_hdr->startPrbu);
    printf("# Tot PRBs number = %d\n", (uint32_t)sec_hdr->numPrbu);

    return RE_OK;
}

int oran_dump_umsg_hdrs(struct oran_umsg_hdrs * umsg)
{
    if(!umsg)
        return RE_EINVAL;

    printf("\n============ U-msg Dump ============\n");
    oran_dump_ethvlan_hdr(&(umsg->ethvlan));
    oran_dump_ecpri_hdr(&(umsg->ecpri));
    oran_dump_umsg_hdr(&(umsg->iq_hdr));
    oran_dump_umsg_iq_hdr(&(umsg->sec_hdr));
    printf("\n====================================\n");

    return RE_OK;
}
