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

#include "timing_utils.hpp"

#include "nvlog.h"

/**
 * Calculate the duration of a complete frame cycle in nanoseconds
 *
 * Frame cycle = 1024 frames × 10 subframes × (max_slot_id+1) slots × TTI duration
 */
int64_t get_frame_cycle_time_ns(uint16_t max_slot_id, uint16_t opt_tti_us) {
    // Calculate total frame cycle time: frames × subframes × slots × TTI × nanoseconds/microsecond
    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID;        // 256 frames (8-bit frame ID)
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID;            // × 10 subframes/frame
    frame_cycle_time_ns *= (max_slot_id + 1);               // × slots/subframe (depends on numerology)
    frame_cycle_time_ns *= opt_tti_us;                      // × microseconds/slot
    frame_cycle_time_ns *= NS_X_US;                         // × 1000 nanoseconds/microsecond
    return frame_cycle_time_ns;
}

/**
 * Calculate the first Frame 0, Subframe 0, Slot 0 (F0S0S0) time reference
 *
 * This aligns to 10ms boundaries and adjusts for GPS epoch conversion
 */
int64_t get_first_f0s0s0_time() {
    int64_t first_f0s0s0_time = get_ns();  // Get current time in nanoseconds
    // Round down to nearest 10.24s boundary (1024 SFN cycles × 10ms/frame)
    first_f0s0s0_time -= first_f0s0s0_time % (10000000ULL * 1024ULL);
    // Adjust to SFN = 0, accounting for GPS vs TIA (Unix epoch) conversion
    // GPS to Unix TIA: 315964800s + 19 leap seconds = 31596481900 frames % 1024 = 364 frames
    first_f0s0s0_time += (364ULL * 10000000ULL);
    return first_f0s0s0_time;
}

/**
 * Calculate time offset from F0S0S0 to a specific frame/subframe/slot/symbol (FSSS)
 *
 * This includes the symbol offset within the slot
 */
inline int64_t calculate_fsss_offset(uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t start_sym,
                                    uint16_t max_slot_id, uint16_t opt_tti_us) {
    // Calculate total slot count: (frame × 10 subframes + subframe) × slots_per_subframe + slot
    int64_t fsss_offset = ((int64_t)frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) +
                          (int64_t)subframe_id * (max_slot_id + 1) +
                          (int64_t)slot_id);
    // Convert slot count to nanoseconds
    fsss_offset *= opt_tti_us * NS_X_US;
    // Add symbol offset within the slot (fractional slot time)
    fsss_offset += (int)(opt_tti_us * NS_X_US * (float)start_sym / ORAN_ALL_SYMBOLS);
    return fsss_offset;
}

/**
 * Calculate time offset from F0S0S0 to a specific frame/subframe/slot (FSS)
 *
 * This is the slot boundary time (symbol 0 of the slot)
 */
inline int64_t calculate_fss_offset(uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t max_slot_id, uint16_t opt_tti_us) {
    // Calculate total slot count: (frame × 10 subframes + subframe) × slots_per_subframe + slot
    int64_t fss_offset = ((int64_t)frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) +
                          (int64_t)subframe_id * (max_slot_id + 1) +
                          (int64_t)slot_id);
    // Convert slot count to nanoseconds
    fss_offset *= opt_tti_us * NS_X_US;
    return fss_offset;
}

/**
 * Calculate slot t0 (reference time) for a given FSS (Frame/Subframe/Slot)
 *
 * This function finds the closest occurrence of the specified FSS to the current time.
 * Since the FSS repeats every frame cycle (256 frames), there can be multiple occurrences.
 * We find the two candidates closest to current_time and return the nearest one.
 */
int64_t calculate_t0(int64_t current_time, uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t max_slot_id, uint16_t opt_tti_us) {

    // Get duration of an SFN cycle (10ms × 1024 frames = 10.24 seconds)
    int64_t sfn_cycle_duration = 10000000ULL * 1024ULL;  // 10.24s in nanoseconds
    // Get duration of a complete frame cycle (256 frames, based on 8-bit frame ID)
    int64_t frame_cycle_duration = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);

    // Calculate the time for an arbitrary SFN rollover
    // Note: GPS time to Unix TIA conversion is 315964800 seconds + 19 leap seconds
    // This represents 315964819 * 100 = 31596481900 frames
    // This represents 31596481900 % 1024 = 364 SFN rollovers
    int64_t sfn_cycle_time = current_time;
    sfn_cycle_time -= sfn_cycle_time % sfn_cycle_duration;  // Round down to nearest SFN rollover (relative to Unix Epoch)
    sfn_cycle_time += (364ULL * 10000000ULL);  // Adjust to GPS epoch under assumption of 19 leap seconds

    // Round down to nearest frame rollover
    // Handle negative case by going back one SFN cycle
    int64_t sfn_cycle_delta = current_time - sfn_cycle_time;
    while(sfn_cycle_delta < 0) {
        sfn_cycle_time -= sfn_cycle_duration;
        sfn_cycle_delta = current_time - sfn_cycle_time;
    }

    // Determine appropriate frame cycle time based on sfn_cycle_delta
    // Round down to the most recent frame cycle boundary
    int64_t frame_cycle_time = sfn_cycle_time + frame_cycle_duration * (sfn_cycle_delta / frame_cycle_duration);

    // Determine the two FSS times that are closest to the current time
    // One before and one after the current frame cycle
    int64_t fss_offset = calculate_fss_offset(frame_id, subframe_id, slot_id, max_slot_id, opt_tti_us);
    int64_t time1 = frame_cycle_time + fss_offset;              // FSS time in current cycle
    int64_t time2 = time1 + frame_cycle_duration;               // FSS time in next cycle

    // Determine which time is closer to the current time
    if(current_time - time1 < time2 - current_time) {
        return time1;  // Current cycle FSS is closer
    } else {
        return time2;  // Next cycle FSS is closer
    }
}

/**
 * Calculate both t0 (reference time) and TOA (time of arrival) for a received packet
 *
 * This function determines the expected symbol start time (startSym_t0) and the
 * time difference between packet arrival and expected time (TOA). The TOA is used
 * to classify packets as early/on-time/late.
 */
t0_toa_result calculate_t0_toa(int64_t packet_time, int64_t first_f0s0s0_time, int64_t frame_cycle_time_ns,
                                    uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t start_sym,
                                    uint16_t max_slot_id, uint16_t opt_tti_us) {
    t0_toa_result result;

    // Calculate time offset from F0S0S0 based on frame, subframe, slot and symbol
    int64_t fsss_offset = calculate_fsss_offset(frame_id, subframe_id, slot_id, start_sym, max_slot_id, opt_tti_us);
    int64_t fss_offset = calculate_fss_offset(frame_id, subframe_id, slot_id, max_slot_id, opt_tti_us);
    int64_t sym_offset = fsss_offset - fss_offset;  // Symbol offset within the slot

    // Find the most recent F0S0S0 time before the packet arrival
    // first_f0s0s0_time is F0S0S0 sym 0 time in the past
    int64_t delta = packet_time - first_f0s0s0_time;
    // Round down to nearest frame cycle boundary
    int64_t most_recent_f0s0s0_time = first_f0s0s0_time + (delta - (delta % frame_cycle_time_ns));
    // Calculate expected symbol start time in the same frame cycle
    int64_t t0_in_same_cycle = most_recent_f0s0s0_time + fsss_offset;

    // If close to boundary of the cycle, the actual t0 may be in a different cycle
    // Use frame_cycle_time_ns/2 as threshold to determine which cycle
    if(packet_time - t0_in_same_cycle > frame_cycle_time_ns/2)
    {
        // Packet is closer to the next cycle's occurrence
        result.startSym_t0 = t0_in_same_cycle + frame_cycle_time_ns;
    }
    else if(packet_time - t0_in_same_cycle < -frame_cycle_time_ns/2)
    {
        // Packet is closer to the previous cycle's occurrence
        result.startSym_t0 = t0_in_same_cycle - frame_cycle_time_ns;
    }
    else
    {
        // Packet is closest to the current cycle's occurrence
        result.startSym_t0 = t0_in_same_cycle;
    }

    // Calculate TOA: time to arrival w.r.t. the t0 of the symbol
    // Used to classify packets as early/on-time/late
    result.toa = packet_time - result.startSym_t0;

    // Calculate slot t0 by subtracting the symbol offset
    result.slot_t0 = result.startSym_t0 - sym_offset;

    return result;
}