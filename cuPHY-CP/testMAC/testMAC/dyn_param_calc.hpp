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

#include <iostream>
#include <fstream>
#include <cmath>
// #include "common_defines.hpp"
#include "common_utils.hpp"

using namespace std;

typedef struct {
    uint32_t modulation_order;
    float target_code_rate;
} mcs_table_t;

// 3GPP 38.214
static constexpr mcs_table_t pdsch_mcs_tables[4][32] = {
    // PDSCH MCS Table 1
    {
        {2, 120}, // 0
        {2, 157}, // 1
        {2, 193}, // 2
        {2, 251}, // 3
        {2, 308}, // 4
        {2, 379}, // 5
        {2, 449}, // 6
        {2, 526}, // 7
        {2, 602}, // 8
        {2, 679}, // 9
        {4, 340}, // 10
        {4, 378}, // 11
        {4, 434}, // 12
        {4, 490}, // 13
        {4, 553}, // 14
        {4, 616}, // 15
        {4, 658}, // 16
        {6, 438}, // 17
        {6, 466}, // 18
        {6, 517}, // 19
        {6, 567}, // 20
        {6, 616}, // 21
        {6, 666}, // 22
        {6, 719}, // 23
        {6, 772}, // 24
        {6, 822}, // 25
        {6, 873}, // 26
        {6, 910}, // 27
        {6, 948}, // 28
        {2, 0}, // 29
        {4, 0}, // 30
        {6, 0}, // 31
    },

    // PDSCH MCS Table 2
    {
        {2, 120}, // 0
        {2, 193}, // 1
        {2, 308}, // 2
        {2, 449}, // 3
        {2, 602}, // 4
        {4, 378}, // 5
        {4, 434}, // 6
        {4, 490}, // 7
        {4, 553}, // 8
        {4, 616}, // 9
        {4, 658}, // 10
        {6, 466}, // 11
        {6, 517}, // 12
        {6, 567}, // 13
        {6, 616}, // 14
        {6, 666}, // 15
        {6, 719}, // 16
        {6, 772}, // 17
        {6, 822}, // 18
        {6, 873}, // 19
        {8, 682.5}, // 20
        {8, 711}, // 21
        {8, 754}, // 22
        {8, 797}, // 23
        {8, 841}, // 24
        {8, 885}, // 25
        {8, 916.5}, // 26
        {8, 948}, // 27
        {2, 0}, // 28
        {4, 0}, // 29
        {6, 0}, // 30
        {8, 0}, // 31
    },

    // PDSCH MCS Table 3
    {
        {2, 30}, // 0
        {2, 40}, // 1
        {2, 50}, // 2
        {2, 64}, // 3
        {2, 78}, // 4
        {2, 99}, // 5
        {2, 120}, // 6
        {2, 157}, // 7
        {2, 193}, // 8
        {2, 251}, // 9
        {2, 308}, // 10
        {2, 379}, // 11
        {2, 449}, // 12
        {2, 526}, // 13
        {2, 602}, // 14
        {4, 340}, // 15
        {4, 378}, // 16
        {4, 434}, // 17
        {4, 490}, // 18
        {4, 553}, // 19
        {4, 616}, // 20
        {6, 438}, // 21
        {6, 466}, // 22
        {6, 517}, // 23
        {6, 567}, // 24
        {6, 616}, // 25
        {6, 666}, // 26
        {6, 719}, // 27
        {6, 772}, // 28
        {2, 0}, // 29
        {4, 0}, // 30
        {6, 0}, // 31
    },

    // PDSCH MCS Table 4
    {
        {2, 120}, // 0
        {2, 193}, // 1
        {2, 449}, // 2
        {4, 378}, // 3
        {4, 490}, // 4
        {4, 616}, // 5
        {6, 466}, // 6
        {6, 517}, // 7
        {6, 567}, // 8
        {6, 616}, // 9
        {6, 666}, // 10
        {6, 719}, // 11
        {6, 772}, // 12
        {6, 822}, // 13
        {6, 873}, // 14
        {8, 682.5}, // 15
        {8, 711}, // 16
        {8, 754}, // 17
        {8, 797}, // 18
        {8, 841}, // 19
        {8, 885}, // 20
        {8, 916.5}, // 21
        {8, 948}, // 22
        {10, 805.5}, // 23
        {10, 853}, // 24
        {10, 900.5}, // 25
        {10, 948}, // 26
        {2, 0}, // 27
        {4, 0}, // 28
        {6, 0}, // 29
        {8, 0}, // 30
        {10, 0}, // 31
    }
};

static constexpr uint32_t tbs_select[] = {
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
            136,
            144,
            152,
            160,
            168,
            176,
            184,
            192,
            208,
            224,
            240,
            256,
            272,
            288,
            304,
            320,
            336,
            352,
            368,
            384,
            408,
            432,
            456,
            480,
            504,
            528,
            552,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            808,
            848,
            888,
            928,
            984,
            1032,
            1064,
            1128,
            1160,
            1192,
            1224,
            1256,
            1288,
            1320,
            1352,
            1416,
            1480,
            1544,
            1608,
            1672,
            1736,
            1800,
            1864,
            1928,
            2024,
            2088,
            2152,
            2216,
            2280,
            2408,
            2472,
            2536,
            2600,
            2664,
            2728,
            2792,
            2856,
            2976,
            3104,
            3240,
            3368,
            3496,
            3624,
            3752,
            3824,
        };

static inline mcs_table_t get_pdsch_mcs_table(uint32_t table_index, uint32_t mcs_index)
{
    return pdsch_mcs_tables[table_index - 1][mcs_index];
}

static inline mcs_table_t get_pusch_mcs_table(uint32_t table_index, uint32_t mcs_index)
{
    // Use the same MCS table for PDSCH and PUSCH
    return pdsch_mcs_tables[table_index - 1][mcs_index];
}

static inline uint32_t get_bit_one_num(uint32_t var)
{
    uint32_t bit_one_num = 0;
    while (var)
    {
        if (var & 0x1)
        {
            bit_one_num ++;
        }
        var >>= 1;
    }
    return bit_one_num;
}

static inline uint32_t calculate_tbs(mcs_table_t& mcs_table, uint32_t dmrs_syn_loc_bmsk, uint32_t n_prb, uint32_t n_symbols, uint32_t n_layers)
{
    uint32_t tbs = 0;

    uint32_t qm = mcs_table.modulation_order;
    float r = mcs_table.target_code_rate;
    uint32_t n_sc = 12;

    uint32_t dmrs_one_bits = get_bit_one_num(dmrs_syn_loc_bmsk);
    uint32_t n_re = n_prb * (n_symbols - dmrs_one_bits) * n_sc; // n_oh = 0

    float n_info = n_re * r / 1024 * qm * n_layers;

    if (n_info <= 3824)
    {
        int32_t n = max(3.0f, floor(log2(n_info)) - 6);
        float n_info_prime = max((double)24, pow(2, n) * floor(n_info / pow(2, n)));
        for (uint32_t tbs_item : tbs_select)
        {
            if (tbs_item >= n_info_prime)
            {
                tbs = tbs_item;
                break;
            }
        }
    }
    else
    {
        uint32_t n = floor(log2(n_info - 24)) - 5;
        float n_info_prime = max((double)3840, pow(2, n) * round((n_info - 24) / pow(2, n)));
        if (r / 1024.0f < 0.25)
        {
            uint32_t C = ceil((n_info_prime + 24) / 3816);
            tbs = int(8 * C * ceil((n_info_prime + 24) / 8 / C));
        }
        else
        {
            if (n_info_prime > 8424)
            {
                uint32_t C = ceil((n_info_prime + 24) / 8424);
                tbs = int(8 * C * ceil((n_info_prime + 24) / 8 / C));
            }
            else
            {
                tbs = int(8 * ceil((n_info_prime + 24) / 8));
            }
        }
        tbs -= 24;
    }
    return tbs;
}

static inline void calculate_dyn_slot_params(dyn_slot_param_t& dyn_param)
{
    for (uint32_t pdu_id = 0; pdu_id < dyn_param.pdus.size(); pdu_id ++)
    {
        dyn_pdu_param_t& pdu_param = dyn_param.pdus[pdu_id];
        mcs_table_t table = get_pdsch_mcs_table(pdu_param.mcs_table, pdu_param.mcs);
        uint32_t n_prb = pdu_param.prb.prbEnd - pdu_param.prb.prbStart + 1;
        pdu_param.tb_size = calculate_tbs(table, pdu_param.dmrs_sym_loc_bmsk, n_prb, pdu_param.nrOfSymbols, pdu_param.layer);
        pdu_param.tb_size /= 8; //MB to Mb
        pdu_param.modulation_order = table.modulation_order;
        pdu_param.target_code_rate = table.target_code_rate * 10;
    }
}