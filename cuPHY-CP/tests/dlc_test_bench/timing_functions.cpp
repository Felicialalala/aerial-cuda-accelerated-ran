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

// Timing utility functions for test bench
// These are simplified versions for the test environment

#include <cstdint>
#include <chrono>

// From RU emulator timing_utils.hpp - provide struct definition
struct t0_toa_result {
    int64_t slot_t0;
    int64_t startSym_t0;
    int64_t toa;
};

// Constants from RU emulator
#define ORAN_MAX_FRAME_ID 1024
#define ORAN_MAX_SUBFRAME_ID 10
#define NS_X_US 1000
#define ORAN_ALL_SYMBOLS 14

int64_t get_frame_cycle_time_ns(uint16_t max_slot_id, uint16_t opt_tti_us) {
    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID;
    frame_cycle_time_ns *= (max_slot_id + 1);
    frame_cycle_time_ns *= opt_tti_us;
    frame_cycle_time_ns *= NS_X_US;
    return frame_cycle_time_ns;
}

int64_t get_first_f0s0s0_time() {
    auto now = std::chrono::system_clock::now();
    int64_t first_f0s0s0_time = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    first_f0s0s0_time -= first_f0s0s0_time % (10000000ULL * 1024ULL);
    first_f0s0s0_time += (364ULL * 10000000ULL);
    return first_f0s0s0_time;
}

static inline int64_t calculate_fsss_offset(uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t start_sym,
                                    uint16_t max_slot_id, uint16_t opt_tti_us) {
    int64_t fsss_offset = ((int64_t)frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) + 
                          (int64_t)subframe_id * (max_slot_id + 1) + 
                          (int64_t)slot_id);
    fsss_offset *= opt_tti_us * NS_X_US;
    fsss_offset += (int)(opt_tti_us * NS_X_US * (float)start_sym / ORAN_ALL_SYMBOLS);
    return fsss_offset;
}

static inline int64_t calculate_fss_offset(uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t max_slot_id, uint16_t opt_tti_us) {
    int64_t fss_offset = ((int64_t)frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) + 
                          (int64_t)subframe_id * (max_slot_id + 1) + 
                          (int64_t)slot_id);
    fss_offset *= opt_tti_us * NS_X_US;
    return fss_offset;
}

int64_t calculate_t0(int64_t current_time, uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t max_slot_id, uint16_t opt_tti_us) {
    //Get duration of a frame cycle
    int64_t sfn_cycle_duration = 10000000ULL * 1024ULL;
    int64_t frame_cycle_duration = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);

    //Calculate the time for an arbitrary SFN rollover
    int64_t sfn_cycle_time = current_time;
    sfn_cycle_time -= sfn_cycle_time % sfn_cycle_duration;
    sfn_cycle_time += (364ULL * 10000000ULL);

    //Round down to nearest frame rollover
    int64_t sfn_cycle_delta = current_time - sfn_cycle_time;
    while(sfn_cycle_delta < 0) {
        sfn_cycle_time -= sfn_cycle_duration;
        sfn_cycle_delta = current_time - sfn_cycle_time;
    }

    //Determine appropriate frame cycle time based on sfn_cycle_delta
    int64_t frame_cycle_time = sfn_cycle_time + frame_cycle_duration * (sfn_cycle_delta / frame_cycle_duration);
    
    //Determine the two FSS times that are closest to the current time
    int64_t fss_offset = calculate_fss_offset(frame_id, subframe_id, slot_id, max_slot_id, opt_tti_us);
    int64_t time1 = frame_cycle_time + fss_offset;
    int64_t time2 = time1 + frame_cycle_duration;

    //Determine which time is closer to the current time
    if(current_time - time1 < time2 - current_time) {
        return time1;
    } else {
        return time2;
    }
}

t0_toa_result calculate_t0_toa(int64_t packet_time, int64_t first_f0s0s0_time, int64_t frame_cycle_time_ns,
                                    uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t start_sym,
                                    uint16_t max_slot_id, uint16_t opt_tti_us) {
    t0_toa_result result;

    // Calculate time offset from F0S0S0 based on frame, subframe, slot and symbol
    int64_t fsss_offset = calculate_fsss_offset(frame_id, subframe_id, slot_id, start_sym, max_slot_id, opt_tti_us);
    int64_t fss_offset = calculate_fss_offset(frame_id, subframe_id, slot_id, max_slot_id, opt_tti_us);
    int64_t sym_offset = fsss_offset - fss_offset;

    // first_f0s0s0_time is F0S0S0 sym 0 time in the past
    int64_t delta = packet_time - first_f0s0s0_time;
    int64_t most_recent_f0s0s0_time = first_f0s0s0_time + (delta - (delta % frame_cycle_time_ns));
    int64_t t0_in_same_cycle = most_recent_f0s0s0_time + fsss_offset;

    // if close to boundary of the cycle, the actual t0 may be in different cycle
    if(packet_time - t0_in_same_cycle > frame_cycle_time_ns/2)
    {
        result.startSym_t0 = t0_in_same_cycle + frame_cycle_time_ns;
    }
    else if(packet_time - t0_in_same_cycle < -frame_cycle_time_ns/2)
    {
        result.startSym_t0 = t0_in_same_cycle - frame_cycle_time_ns;
    }
    else
    {
        result.startSym_t0 = t0_in_same_cycle;
    }

    // toa is the time to arrival w.r.t. to the t0 of the symbol
    result.toa = packet_time - result.startSym_t0;
    result.slot_t0 = result.startSym_t0 - sym_offset;
    return result;
}



