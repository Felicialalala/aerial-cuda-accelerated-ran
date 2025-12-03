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

#ifndef ORAN_STRUCTS_H__
#define ORAN_STRUCTS_H__

#include <inttypes.h>
#include <assert.h>

#ifdef CUDA_ENABLED
    //Ensure CUDA for the __device__ keyword
    #include "cuda.hpp"
#endif

// COMMENTED FOR FUTURE USE
// enum csec_type {
//     CSEC_PERIODS = 0,
//     CSEC_ULDL,
//     CSEC_PRACH,
//     CSEC_UE_SCHED,
//     CSEC_UE_CHAN
// };


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Dump functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int oran_dump_ethvlan_hdr(struct oran_eth_hdr * ethvlan_hdr);
int oran_dump_ecpri_hdr(struct oran_ecpri_hdr * ecpri_hdr);

int oran_dump_cmsg_hdr(struct oran_cmsg_radio_app_hdr * cmsg_hdr);
int oran_dump_cmsg_uldl_hdr(struct oran_cmsg_sect1 * cmsg_uldl_hdr);
int oran_dump_cmsg_hdrs(struct oran_cmsg_uldl_hdrs * cmsg);
int oran_dump_custom_sect1_cmsg_cell_info(struct oran_cmsg_uldl_hdrs * cmsg);
int oran_dump_custom_sect3_cmsg_cell_info(struct oran_cmsg_prach_hdrs * cmsg);

int oran_dump_umsg_hdr(struct oran_umsg_iq_hdr * umsg_hdr);
int oran_dump_umsg_iq_hdr(struct oran_u_section_uncompressed * sec_hdr);
int oran_dump_umsg_hdrs(struct oran_umsg_hdrs * umsg);

//----- Keep



#endif /*ifndef ORAN_STRUCTS_H__*/
