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

struct dlRateMatchingDescr
{
    const uint32_t* d_rate_matching_input;
    uint32_t* d_rate_matching_output;
    uint32_t* d_restructure_rate_matching_output;

    const uint32_t* d_Er_array;
    const uint32_t* d_k0_array;
    const uint32_t* d_TB_start_offset_array;

    bool enable_scrambling;
    bool enable_layer_mapping;

    uint32_t num_TBs;
    uint32_t cmax;
    uint32_t emax; // max number of rate matched bits across all TB, rounded up to be divisible by 32
    uint32_t max_bits_per_layer;
    uint32_t num_layers;
    const PdschPerTbParams* cfg_workspace;

    //Extension for modulation fusing
    PdschDmrsParams* d_params;
    uint16_t* temp_xtf_re_map;

    uint16_t  max_PRB_BWP; // used to index into xtf_re_map. Max is 273

    PdschUeGrpParams* d_ue_grp_params;
};
typedef struct dlRateMatchingDescr dlRateMatchingDescr_t;
