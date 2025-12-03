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

#include "cumac.h"

constexpr uint16_t MAX_NUM_CELL = 6; // maximum number of cells for joint scheduling. Default: 1
constexpr uint16_t MAX_NUM_UE_ANT_PORT = 4; // maximum number of antenna ports per UE. Default: 4
constexpr uint16_t MAX_NUM_BS_ANT_PORT = 64; // number of antenna ports per RU. Default: 64
constexpr uint16_t MAX_NUM_PRG = 136; // maximum number of PRGs per cell. Default:272 PRBs/2 prbPerPrg = 136 PRGs
constexpr uint16_t MAX_NUM_SUBBAND = 4; // maximum number of subbands per UE considered for UE grouping. Default: 4
constexpr uint16_t MAX_NUM_PRG_SAMP_PER_SUBBAND = 4; // maximum number of per-PRG SRS channel estimate samples per subband. Default: 2
constexpr uint16_t MAX_NUM_UE_SRS_INFO_PER_SLOT = 48; // maximum number of SRS info per slot. Default: 48, assume SRS comb: 4, max CS: 12, max SRS port: 4, max SRS UE/symbol: 12, max SRS symbols/SLOT: 4(SS), 2(UL)
constexpr uint16_t MAX_NUM_SRS_UE_PER_CELL = 384; // SRS UE capacity per cell. Default: 384, i.e., totally 384 UEs can be configured for SRS channel estimation in a cell
constexpr uint16_t MAX_NUM_SCHD_UE_PER_CELL = 16; // maximum number of scheduled UEs per cell. Default: 16, assume max 16 UEs can be scheduled per cell
constexpr uint16_t MAX_NUM_UE_FOR_GRP_PER_CELL = 64; // maximum number of UEs for grouping per cell. Default: 32
constexpr float    UNAVAILABLE_CHAN_ORTH_VAL = std::numeric_limits<float>::max(); // value for unavailable channel orthogonality

constexpr uint16_t MAX_NUM_UE_PER_GRP = 16; // maximum number of UEs per UEG. Default: 16
constexpr uint16_t MAX_NUM_LAYER_PER_GRP = 16; // maximum number of layers per UEG. Default: 16
constexpr uint16_t MAX_NUM_UEG_PER_CELL = MAX_NUM_SUBBAND; // maximum number of UEGs scheduled per cell per TTI. Default: 16

/*****************************************************/
// input to cuMAC UE grouping on GPU
struct cumac_muUeGrp_req_ue_info_t { // per SRS enabled UE
   uint32_t    avgRate; // average rate in DL, in bits/s
   uint32_t    currRate; // current instantaneous rate in DL, in bits/s
   uint32_t    bufferSize; // current buffer size in bits
   uint16_t    rnti; // C-RNTI ranging from 1 to 65535
   uint16_t    id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
   uint16_t    numAllocPrgLastTx; // number of PRGs allocated to the UE in the last transmission
   uint16_t    srsInfoIdx; // index of the SRS info in the SRS info array
   uint8_t     layerSelLastTx; // number of layers selected for the UE in the last transmission
   uint8_t     nUeAnt; // number of SRS TX antenna ports. Value: 2, 4
   uint8_t     flags = 0x00; 
   // 1st bit (flags & 0x01) - is a valid UE info, 0: invalid, >0: valid
   // 2nd bit (flags & 0x02) - new TX indication, 0: re-TX, >0: new TX
   // 3rd bit (flags & 0x04) - SRS chanEst available, 0: no, >0: yes
   // 4th bit (flags & 0x08) - has updated SRS info in the current slot, 0: no, >0: yes
};

struct cumac_muUeGrp_req_srs_info_t { // per SRS enabled UE with updated SRS channel estimation
   float       srsWbSnr;
   uint16_t    rnti; // C-RNTI ranging from 1 to 65535
   uint16_t    id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
   uint8_t     nUeAnt; // number of SRS TX antenna ports. Value: 2, 4
   uint8_t     flags = 0x00; 
   // 1st bit (flags & 0x01) - is a valid SRS info, 0: invalid, 1: valid
   cuComplex   srsChanEst[MAX_NUM_BS_ANT_PORT*MAX_NUM_UE_ANT_PORT*MAX_NUM_SUBBAND*MAX_NUM_PRG_SAMP_PER_SUBBAND];
   // for each subband, each PRG, each UE/RU antenna port, each RU antenna port
};

struct cumac_muUeGrp_req_info_t { // per cell
   float       betaCoeff = 1.0; // exponent applied to the instantaneous rate for proportional-fair scheduling. Default value is 1.0.
   float       muCoeff = 1.5; // coefficient for prioritizing UEs feasible for MU-MIMO transmissions. Default value is 1.5.
   float       chanCorrThr = 0.7; // threshold on the channel vector correlation value for UE grouping. Value: a real number between 0 and 1.0. Default: 0.7
   float       srsSnrThr = -3.0; // Threshold on measured SRS SNR in dB for determining the feasibility of MU-MIMO transmission. Default value is -3.0 (dB).
   float       muGrpSrsSnrMaxGap = 100.0; // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG. Value: a real number greater than 0.0. Default: 100.0
   float       muGrpSrsSnrSplitThr = -100.0; // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately. Value: a real number greater than 0.0. Default: -100.0
   uint16_t    numUeInfo; // number of effective ueInfo in the payload
   uint16_t    numSrsInfo; // number of effective srsInfo in the payload
   uint16_t    numSubband; // number of subbands considered for UE grouping.
   uint16_t    numPrgSampPerSubband; // number of per-PRG SRS channel estimate samples per subband.
   uint16_t    numUeForGrpPerCell = 64; // number of UEs considered for UE grouping per cell. Default: 64
   uint16_t    nPrbGrp; // number of PRGs that can be allocated.
   uint8_t     nBsAnt = 64; // Each RU’s number of TX & RX antenna ports. Default: 64
   uint8_t     nMaxUeSchdPerCellTTI = 64; // maximum number of UEs scheduled per cell per TTI. Default: 16
   uint8_t     nMaxUePerGrp = 16; // maximum number of UEs per UEG. Default: 16
   uint8_t     nMaxLayerPerGrp = 16; // maximium number of layers per UEG. Default: 16
   uint8_t     nMaxLayerPerUeSu = 4; // maximium number of layers per UE for SU-MIMO. Default: 4
   uint8_t     nMaxLayerPerUeMu = 4; // maximium number of layers per UE for MU-MIMO. Default: 4
   uint8_t     nMaxUegPerCell = 4; // maximum number of UEGs per cell. Default: 4
   uint8_t     allocType = 1; // PRB allocation type. Currently only support 1: consecutive type-1 allocation.  
   cumac_muUeGrp_req_srs_info_t srsInfo[0];
   cumac_muUeGrp_req_ue_info_t ueInfo[0]; 
};

struct cumac_muUeGrp_req_msg_t {
   uint16_t    sfn;
   uint16_t    slot;
   uint32_t    offsetData; // Offset of data payload in the nvipc_buf->data_buf
   uint8_t     extraPayload[0];   // extra payload for future use
};
/*****************************************************/

/*****************************************************/
// output from cuMAC UE grouping on GPU
struct cumac_muUeGrp_resp_ue_info_t {
   uint16_t    rnti; // C-RNTI ranging from 1 to 65535
   uint16_t    id; // 0-based UE ID used for cuMAC scheduling, randing from 0 to MAX_NUM_SRS_UE_PER_CELL-1
   uint8_t     layerSel; 
   // bit map of layer selection for the current transmission
   // bit 0: layer corresponding to antenna port 0
   // bit 1: layer corresponding to antenna port 1
   // bit 2: layer corresponding to antenna port 2
   // bit 3: layer corresponding to antenna port 3
   uint8_t     ueOrderInGrp; // UE order in the UEG for the current transmission (to assist beamforming)
   uint8_t     nSCID; // The DMRS sequence initialization parameter n_SCID assigned to the UE. Range: 0, 1.
   uint8_t     flags; // flags & 0x01 - is_valid, flags & 0x02 - MU-MIMO indication (0: SU-MIMO, 1: MU-MIMO)
};

struct cumac_muUeGrp_resp_ueg_info_t {
   int16_t     allocPrgStart; // PRG index of the first PRG allocated to the UEG in the current transmission
   int16_t     allocPrgEnd; // one plus the PRG index of the last PRG allocated to the UEG in the current transmission
   uint8_t     numUeInGrp; // number of UEs in the UEG
   uint8_t     flags; // flags & 0x01 - is_valid
   cumac_muUeGrp_resp_ue_info_t ueInfo[MAX_NUM_UE_PER_GRP];
};

struct cumac_muUeGrp_resp_info_t {
   uint32_t    numSchdUeg; // total number of scheduled UEGs
   cumac_muUeGrp_resp_ueg_info_t schdUegInfo[MAX_NUM_UEG_PER_CELL];
};

struct cumac_muUeGrp_resp_msg_t {
   uint16_t    sfn;
   uint16_t    slot;
   uint32_t    offsetData; // Offset of data payload in the nvipc_buf->data_buf
   uint8_t     extraPayload[0];   // extra payload for future use
};

typedef enum
{
    CUMAC_PARAM_REQUEST = 0x00,
    CUMAC_PARAM_RESPONSE = 0x01,
    CUMAC_CONFIG_REQUEST = 0x02,
    CUMAC_CONFIG_RESPONSE = 0x03,
    CUMAC_START_REQUEST = 0x04,
    CUMAC_STOP_REQUEST = 0x05,
    CUMAC_STOP_RESPONSE = 0x06,
    CUMAC_ERROR_INDICATION = 0x07,
    CUMAC_START_RESPONSE = 0x08,

    CUMAC_DL_TTI_REQUEST = 0x80,
    CUMAC_UL_TTI_REQUEST = 0x81,

    CUMAC_SCH_TTI_REQUEST = 0x82,
    CUMAC_SCH_TTI_RESPONSE = 0x83,

    CUMAC_TTI_END = 0x8F,
    CUMAC_TTI_ERROR_INDICATION = 0x90,
} cumac_muUeGrp_msg_t;