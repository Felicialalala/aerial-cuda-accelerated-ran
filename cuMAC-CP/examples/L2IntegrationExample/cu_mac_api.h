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

#ifndef CU_MAC_API_H
#define CU_MAC_API_H

#include "cumac_msg.h"

// todo: move the following constants to test case YAML file
#define MAX_NUM_LCG 4
#define MAX_NUM_LC 4
#define MAX_NUM_UE 100 // a maximum of 512 UEs is supported
#define MAX_NUM_OUTPUT_SORTED_LC_PER_QOS MAX_NUM_UE*MAX_NUM_LC // maximum number of output sorted LCs/LCGs per DL/UL QoS type, used for GPU memory allocation
#define MAX_NUM_SCHEDULED_UE 16 // maximum number of scheduled UEs per time slot for both DL and UL
#define IIR_ALPHA 0.001

// #define SLOT_INTERVAL_NS (500)
#define SLOT_INTERVAL_NS 1E9 // Time slot interval in nanoseconds. Set to 1 second for debug only, 
#define SLOT_DURATION 0.0005 // Time slot interval in seconds, 0.5ms

#define NUM_TIME_SLOTS 10 // total number of time slots to simulate

struct PFM_DL_LC_INFO {
    uint32_t        tbs_scheduled;
    uint32_t        ravg;
    uint32_t        pfm; 
    uint8_t         flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0
    uint8_t         qos_type; // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical
    uint8_t         padding[2];  // Padding to align to 32-bit
};

struct PFM_UL_LCG_INFO {
    uint32_t        tbs_scheduled;
    uint32_t        ravg;
    uint32_t        pfm; 
    uint8_t         flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0
    uint8_t         qos_type; // 0 - ul_gbr_critical, 1 - ul_gbr_non_critical, 2 - ul_ngbr_critical, 3 - ul_ngbr_non_critical, 4 - ul_mbr_non_critical 
    uint8_t         padding[2];  // Padding to align to 32-bit
};

struct PFM_UE_INFO {
    PFM_DL_LC_INFO      dl_lc_info[MAX_NUM_LC];
    PFM_UL_LCG_INFO     ul_lcg_info[MAX_NUM_LCG];
    uint32_t            ambr;
    uint32_t            rcurrent_dl;
    uint32_t            rcurrent_ul;
    uint16_t            rnti;
    uint8_t             num_layers_dl;
    uint8_t             num_layers_ul;
    uint8_t             flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - is_scheduled_dl, flags & 0x04 - is_scheduled_ul
    uint8_t             carrier_id;
    uint8_t             num_dl_lcs;
    uint8_t             num_ul_lcgs;
};

/* ======================================================= */

struct PFM_DL_OUTPUT_INFO {
    uint16_t  rnti;
    uint8_t   lc_id;
    uint8_t   padding;  // Padding to align to 32-bit
};

struct PFM_UL_OUTPUT_INFO {
    uint16_t  rnti;
    uint8_t   lcg_id;
    uint8_t   padding;  // Padding to align to 32-bit
};

// expected output from the GPU
PFM_DL_OUTPUT_INFO dl_gbr_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_DL_OUTPUT_INFO dl_gbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_DL_OUTPUT_INFO dl_ngbr_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_DL_OUTPUT_INFO dl_ngbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_DL_OUTPUT_INFO dl_mbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];

PFM_UL_OUTPUT_INFO ul_gbr_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_UL_OUTPUT_INFO ul_gbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_UL_OUTPUT_INFO ul_ngbr_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_UL_OUTPUT_INFO ul_ngbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];
PFM_UL_OUTPUT_INFO ul_mbr_non_critical[MAX_NUM_OUTPUT_SORTED_LC_PER_QOS];

struct cumac_pfm_tti_req_t {
    cumac_msg_header_t header;
    uint32_t offset_ue_info_arr; // Offset of the PFM_UE_INFO data array in nvipc_buf->data_buf
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_ue; // total number of UEs in the payload
    uint16_t num_output_sorted_lc[10]; 
    // number of output sorted LCs/LCGs per DL/UL QoS type, 0-9 are the indices of the QoS types: 
    // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical, 
    // 5 - ul_gbr_critical, 6 - ul_gbr_non_critical, 7 - ul_ngbr_critical, 8 - ul_ngbr_non_critical, 9 - ul_mbr_non_critical.
};

struct cumac_pfm_tti_resp_t {
    cumac_msg_header_t header;
    uint16_t sfn;
    uint16_t slot;
    uint32_t offset_output_sorted_lc[10];
    // offset of output sorted LCs/LCGs list per DL/UL QoS type, 0-9 are the indices of the QoS types: 
    // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical, 
    // 5 - ul_gbr_critical, 6 - ul_gbr_non_critical, 7 - ul_ngbr_critical, 8 - ul_ngbr_non_critical, 9 - ul_mbr_non_critical.
    uint16_t num_output_sorted_lc[10]; 
    // number of output sorted LCs/LCGs per DL/UL QoS type, 0-9 are the indices of the QoS types: 
    // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical, 
    // 5 - ul_gbr_critical, 6 - ul_gbr_non_critical, 7 - ul_ngbr_critical, 8 - ul_ngbr_non_critical, 9 - ul_mbr_non_critical.   
};

#endif  // CU_MAC_API_H
