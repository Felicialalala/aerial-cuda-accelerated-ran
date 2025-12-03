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

#ifndef _ORAN_UTILS_CONVERSION_NVLOG_H_
#define _ORAN_UTILS_CONVERSION_NVLOG_H_

#include <cstdint>

inline uint64_t sfn_to_tai(int sfn, int slot, uint64_t approx_tai_time_ns, int64_t gps_alpha, int64_t gps_beta, int mu)
{
    static const uint64_t TAI_TO_GPS_OFFSET_NS = (315964800ULL + 19ULL) * 1000000000ULL;
    int64_t gps_offset = ((gps_beta * 1000000000LL) / 100LL) + ((gps_alpha * 10000ULL) / 12288ULL);
    static const uint64_t FRAME_PERIOD_NS = 10000000;
    static const int SFN_MAX_PLUS1 = 1024;
    static const int slot_period_ns[] = {1000000, 500000, 250000, 125000, 62500};

    // First, figure out the base SFN
    uint64_t approx_gps_time_ns = approx_tai_time_ns - TAI_TO_GPS_OFFSET_NS;
    int64_t full_wrap_period_ns = FRAME_PERIOD_NS * SFN_MAX_PLUS1;
    int64_t half_wrap_period_adjust_ns = full_wrap_period_ns / 2 - sfn * FRAME_PERIOD_NS - slot * slot_period_ns[mu];

    uint64_t base_gps_time_ns = (approx_gps_time_ns - gps_offset + half_wrap_period_adjust_ns) / full_wrap_period_ns;
    base_gps_time_ns *= full_wrap_period_ns;
    base_gps_time_ns += gps_offset%full_wrap_period_ns;    
    uint64_t base_tai_time_ns = base_gps_time_ns + TAI_TO_GPS_OFFSET_NS;

    return base_tai_time_ns + sfn * FRAME_PERIOD_NS + slot * slot_period_ns[mu];
}

#endif // _ORAN_UTILS_CONVERSION_NVLOG_H_
