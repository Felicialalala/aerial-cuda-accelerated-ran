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

#ifndef _DEFINES_HPP_
#define _DEFINES_HPP_

typedef enum
{
    PUSCH       = 0,
    PDSCH       = 1,
    PDCCH_UL    = 2,
    PDCCH_DL    = 3,
    PBCH        = 4,
    PUCCH       = 5,
    PRACH       = 6,
    CSI_RS      = 7,
    SRS         = 8,
    DL_BFW      = 9,
    UL_BFW      = 10,
    CHANNEL_MAX = 11
} channel_type_t;

static inline const char* get_channel_name(int channel) {
    switch (channel) {
    case channel_type_t::PUSCH:
        return "PUSCH";
    case channel_type_t::PDSCH:
        return "PDSCH";
    case channel_type_t::PDCCH_UL:
        return "PDCCH_UL";
    case channel_type_t::PDCCH_DL:
        return "PDCCH_DL";
    case channel_type_t::PBCH:
        return "PBCH";
    case channel_type_t::PUCCH:
        return "PUCCH";
    case channel_type_t::PRACH:
        return "PRACH";
    case channel_type_t::CSI_RS:
        return "CSI_RS";
    case channel_type_t::SRS:
        return "SRS";
    case channel_type_t::DL_BFW:
        return "BFW_DL";
    case channel_type_t::UL_BFW:
        return "BFW_UL";
    default:
        return "INVALID";
    }
}


#endif /* _DEFINES_HPP_ */