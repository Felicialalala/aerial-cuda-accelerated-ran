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

#if !defined(SRSTX_HPP_INCLUDED_)
#define SRSTX_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * @brief Struct that tracks all necessary parameters for CSI-RS computation.
 */
typedef struct _SrsTxParams
{
    uint8_t  nAntPorts;                // number of SRS antenna ports. 1,2, or 4
    uint8_t  nSyms;                    // number of SRS symbols. 1,2, or 4
    uint8_t  nRepetitions;             // number of repititions. 1,2, or 4
    uint8_t  combSize;                 // SRS comb size. 2 or 4
    uint8_t  startSym;                 // starting SRS symbol. 0-13
    uint16_t sequenceId;               // SRS sequence id. 0-1023
    uint8_t  configIdx;                // SRS bandwidth cfg idx. 0-63
    uint8_t  bandwidthIdx;             // SRS bandwidth index. 0-3
    uint8_t  combOffset;               // SRS comb offset. 0-3
    uint8_t  cyclicShift;              // cyclic shift. 0-11
    uint8_t  frequencyPosition;        // frequency domain position. 0-67
    uint16_t frequencyShift;           // frequency domain shift. 0-268
    uint8_t  frequencyHopping;         // freuqnecy hopping options. 0-3
    uint8_t  resourceType;             // Type of SRS allocation. 0: aperiodic. 1: semi-persistent. 2: periodic
    uint16_t Tsrs;                     // SRS periodicity in slots. 0,2,3,5,8,10,16,20,32,40,64,80,160,320,640,1280,2560
    uint16_t Toffset;                  // slot offset value. 0-2569
    uint8_t  groupOrSequenceHopping;   // Hopping configuration. 0: no hopping. 1: groupHopping. 2: sequenceHopping
    uint16_t idxSlotInFrame;
    uint16_t idxFrame;
    uint16_t nSlotsPerFrame;
    uint16_t nSymbsPerSlot;
} SrsTxParams;

void kernelSelectGenSrsTx(cuphyGenSrsTxLaunchCfg_t* pLaunchCfg, uint32_t numParams);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // SRSTX_HPP_INCLUDED_
