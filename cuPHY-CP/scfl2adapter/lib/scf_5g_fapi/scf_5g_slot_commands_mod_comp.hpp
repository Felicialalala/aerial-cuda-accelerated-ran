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

#include "slot_command/slot_command.hpp"

using namespace slot_command_api;

using comp_method = aerial_fh::UserDataCompressionMethod;

constexpr float SQRT_2 = std::sqrt(2.0);
constexpr float SQRT_0_5 = std::sqrt(0.5);

namespace scf_5g_fapi {

static constexpr uint8_t DEFAULT_MODCOMP_EXT_TYPE = 4;

static constexpr std::array<float, 5> QAM_SCALER_LUT{1.0, (2.0/sqrt(2.0)/SQRT_2), (4.0/sqrt(10.0)/SQRT_2), (8.0/sqrt(42.0)/SQRT_2), (16.0/sqrt(170.0)/SQRT_2)};

inline void update_mod_comp_info_common(prb_info_t& prb_info, float bwScaler);
inline void update_mod_comp_info_section(prb_info_t& prb_info, uint16_t reMask ,float beta, uint16_t qamOrder);

static constexpr uint8_t DEFAULT_CSF = 1;
static constexpr uint8_t BPSK_CSF = 0;

#define MODTAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

using bw_scaler_map_t = std::unordered_map<uint16_t, float>;
static bw_scaler_map_t BW_MAP = {
    //15Mhz FR1
    {25,  1.0/std::sqrt(25 * 12)},
    {52,  1.0/std::sqrt(52 * 12)},
    {79,  1.0/std::sqrt(79 * 12)},
    {106,  1.0/std::sqrt(106 * 12)},
    {133,  1.0/std::sqrt(133 * 12)},
    {160,  1.0/std::sqrt(160 * 12)},
    {216,  1.0/std::sqrt(216 * 12)},
    {270, 1.0/std::sqrt(270 * 12)},
    //30Mhz FR1
    {11, 1.0/std::sqrt(11 * 12)},
    {24, 1.0/std::sqrt(24 * 12)},
    {38, 1.0/std::sqrt(38 * 12)},
    {51, 1.0/std::sqrt(51 * 12)},
    {75, 1.0/std::sqrt(75 * 12)},
    {162, 1.0/std::sqrt(162 * 12)},
    {189, 1.0/std::sqrt(189 * 12)},
    {217, 1.0/std::sqrt(217 * 12)},
    {245, 1.0/std::sqrt(245 * 12)},
    {273, 1.0/std::sqrt(273 * 12)},
    // 60Mhz FR1
    {18, 1.0/std::sqrt(18 * 12)},
    {24, 1.0/std::sqrt(24 * 12)},
    {31, 1.0/std::sqrt(31 * 12)},
    {79, 1.0/std::sqrt(79 * 12)},
    {93, 1.0/std::sqrt(93 * 12)},
    {107, 1.0/std::sqrt(107 * 12)},
    {121, 1.0/std::sqrt(121 * 12)},
    {135, 1.0/std::sqrt(135 * 12)},
    //FR2
    {32, 1.0/std::sqrt(32 * 12)},
    {66, 1.0/std::sqrt(66 * 12)},
    {132, 1.0/std::sqrt(132 * 12)},
    {264, 1.0/std::sqrt(264 * 12)}
};


inline float getBwScaler(uint16_t num_dl_prb) {
    auto iter = BW_MAP.find(num_dl_prb);
    if (iter!= BW_MAP.end()) {
        return iter->second;
    } else {
        return 0.0f;
    }
}

inline void update_mod_comp_info_common(prb_info_t& prb_info, float bwScaler) {

    prb_info.comp_info.common.ef = 1;
    prb_info.comp_info.common.extType = ORAN_CMSG_SECTION_EXT_TYPE_5;
    prb_info.comp_info.common.nSections = 0;
    prb_info.comp_info.common.udIqWidth = 0;
    prb_info.comp_info.bwScaler = bwScaler;
    prb_info.common.reMask  = 0;
}

inline uint8_t getUdIqWidth(uint8_t qamOrder) {
    switch(qamOrder) {
        case CUPHY_QAM_2: return CUPHY_QAM_4;
        case CUPHY_QAM_4: return CUPHY_QAM_4 >> 1;
        case CUPHY_QAM_16:
        case CUPHY_QAM_64:
        case CUPHY_QAM_256:
        default:
            return qamOrder >> 1;
    }
}

inline uint8_t getTblIndex(uint8_t qamOrder) {
    switch(qamOrder) {
        case CUPHY_QAM_2: return CUPHY_QAM_2 - 1;
        case CUPHY_QAM_4: return CUPHY_QAM_4 - 1;
        case CUPHY_QAM_16:
        case CUPHY_QAM_64:
        case CUPHY_QAM_256:
        default:
            return qamOrder >> 1;
    }
}

inline void update_mod_comp_info_common_pdsch(prb_info_t& prb_info, float bwScaler) {

    prb_info.comp_info.common.ef = 1;
    prb_info.comp_info.common.extType = ORAN_CMSG_SECTION_EXT_TYPE_5;
    prb_info.comp_info.common.nSections = 0;
    prb_info.comp_info.common.udIqWidth = 0;
    prb_info.comp_info.bwScaler = bwScaler;
}

/**
 * @brief Encodes a floating-point scale value into a 16-bit modCompScaler format.
 *
 * The modCompScaler format uses 4 bits for exponent and 11 bits for mantissa.
 * This function finds the best (exponent, mantissa) pair to approximate the input scale.
 *
 * @param scale The scale value to encode. Must be in the range (0, 1).
 * @return Encoded 16-bit modCompScaler value.
 * @throws std::invalid_argument if scale is not in (0, 1).
 */
inline uint16_t float_to_modcompscaler(float scale) {

    if (!(scale > 0.0f && scale < 1.0f)) {
        throw std::invalid_argument("Scale must be in the range (0, 1) for modCompScaler encoding.");
    }
    
    int exp = 0;
    float mantissa_float = std::frexp(scale, &exp);
    
    uint16_t best_modcompscaler = 0;
    float best_error = std::numeric_limits<float>::max();
    
    float scaled_mantissa = mantissa_float * 2048.0f;
    int mantissa = static_cast<int>(std::round(scaled_mantissa));
    int custom_exp = -exp;  
    
    mantissa = std::clamp(mantissa, 0, 0x7FF);
    
    custom_exp = std::clamp(custom_exp, 0, 15);
    if (custom_exp == 15) {
        mantissa = 0x7FF;
    } else if (custom_exp == 0) {
        mantissa = std::clamp(static_cast<int>(std::round(scale * 2048.0f)), 0, 0x7FF);
    }
    
    // Calculate error for direct approach
    float reconstructed = static_cast<float>(mantissa) / (2048 << custom_exp);
    float error = std::fabs(reconstructed - scale);
    best_error = error;
    best_modcompscaler = static_cast<uint16_t>((custom_exp << 11) | mantissa);
    
    if (error == 0.0f) {
        return best_modcompscaler;
    }
    
    // Only search if error is significant
    if (error > 1e-4f) {  // 0.01% error threshold
        const int start_exp = std::max(0, custom_exp - 1);
        const int end_exp = std::min(15, custom_exp + 1);
        
        for (int e = start_exp; e <= end_exp; ++e) {
            if (e == custom_exp) continue; 
            
            const float scale_factor = static_cast<float>(2048 << e);
            const int test_mantissa = static_cast<int>(std::round(scale * scale_factor));
            
            // Check if mantissa is in valid range
            if (test_mantissa < 0 || test_mantissa > 0x7FF) {
                continue;
            }
            
            // Calculate error for this combination
            const float test_reconstructed = static_cast<float>(test_mantissa) / scale_factor;
            const float test_error = std::fabs(test_reconstructed - scale);
            
            if (test_error < best_error) {
                best_error = test_error;
                best_modcompscaler = static_cast<uint16_t>((e << 11) | test_mantissa);
                
                // Early termination for perfect match
                if (test_error == 0.0f) {
                    break;
                }
            }
        }
    }
    
    return best_modcompscaler;
}

inline void update_mod_comp_info_section(prb_info_t& prb_info, uint16_t reMask ,float beta, uint16_t qamOrder, uint8_t shift) {

    auto& section = prb_info.comp_info.sections[prb_info.comp_info.common.nSections.get()];
    prb_info.common.reMask |= reMask;
    section.mcScaleReMask = reMask;
    uint8_t prev_udIqWidth = +prb_info.comp_info.common.udIqWidth.get();
    uint8_t curr_udIqWidth = getUdIqWidth(qamOrder);
    if (!prb_info.comp_info.common.nSections.get()) {
        section.csf = shift;
    } else {
        auto prev_csf = static_cast<uint8_t>(prb_info.comp_info.sections[prb_info.comp_info.common.nSections.get() - 1 ].csf.get());
        if (!prev_csf && prev_udIqWidth == CUPHY_QAM_4) { //BPSK
            section.csf = 0;
        } else { // Mixed MCS
            // set prev csf = 0 and current scf =1
            /*
            *     In the Figure A-7 example, for overlain 16QAM (green) and 64QAM (red), the high-MCS (red) points are shifted by ½
            *     the high-MCS resolution (here, -1/8) to allow all points to share the same “grid”, as shown in the middle figure wherein
            *     the red and green points overlay. All I and Q values can be represented by 3 bits each on the fronthaul interface.
            *     The O-DU uses the constellation shift flag (csf) to tell the O-RU which data (red points) to “unshift” by adding 1/8 to
            *     them, thereby restoring the original constellation values. After that, modCompScaler (or mcScaleOffset) is applied to
            *     set the data to the correct power levels (separate modCompScaler or mcScaleOffset values may be used for the differing
            *     MCS data).
            *     When decompressing, the O-RU must “unshift” the constellation (or perhaps not, depending on “csf”) and also apply a
            *     scale factor for the constellation types represented in the section. There are expected to be either one or two modulation
            *     types in the section, no more. The modulation type is inferred from the reMask bits, where each “one” bit indicates the
            *     shift command (“csf”) and scale factor (“modCompScaler” when using extension type 4, and “mcScaleOffset” when
            *     using extension type 5) for the REs in the subject PRB. The scale factor allows not only for correcting for different
            *     constellation scaling (e.g. for multiplexed channel data in a PRB inclusing QPSK and 16QAM, QPSK involves a
            *     2/sqrt(2) factor while 16QAM involves a 4/sqrt(10) factor), but also allows different channel power scaling which is
            *     permitted as a 3GPP option.
            */
            section.csf = 1;
            if (prev_udIqWidth < curr_udIqWidth) {
                prb_info.comp_info.sections[prb_info.comp_info.common.nSections.get() - 1].csf = 0;
            } else if (prev_udIqWidth > curr_udIqWidth) {
                section.csf = 0;
            }
        }
    }
    const auto tBlIndex = getTblIndex(qamOrder);

    prb_info.comp_info.modCompScalingValue[prb_info.comp_info.common.nSections.get()] = beta * QAM_SCALER_LUT[tBlIndex];
    
    const float calculated_scale = prb_info.comp_info.modCompScalingValue[prb_info.comp_info.common.nSections.get()] * prb_info.comp_info.bwScaler;
    section.mcScaleOffset = float_to_modcompscaler(calculated_scale);
    
    NVLOGD_FMT(MODTAG, "beta {} tableIndex {} QAM_SCALER_LUT[{}] {} BWscaler {} modCompScalingValue[{}] {} calculated_scale = {} mcScaleOffset = {}", 
                beta, tBlIndex, tBlIndex, QAM_SCALER_LUT[tBlIndex], prb_info.comp_info.bwScaler, 
                prb_info.comp_info.common.nSections.get(), prb_info.comp_info.modCompScalingValue[prb_info.comp_info.common.nSections.get()], calculated_scale, section.mcScaleOffset.get());

    prb_info.comp_info.common.udIqWidth = std::max(curr_udIqWidth,  prev_udIqWidth);
    prb_info.comp_info.common.nSections = prb_info.comp_info.common.nSections.get() + 1;
}

}