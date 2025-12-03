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

#include <cstdint>
#include <utils.hpp>
#include "aerial-fh-driver/oran.hpp"

// Time conversion constants
#define US_X_MS 1000         //!< Microseconds per millisecond
#define US_X_S  1000000      //!< Microseconds per second
#define NS_X_US 1000         //!< Nanoseconds per microsecond
#define NS_X_S 1000000000    //!< Nanoseconds per second

/**
 * Result structure for t0 and time-of-arrival calculations
 */
struct t0_toa_result {
    int64_t slot_t0;        //!< Slot start time (t0) in nanoseconds
    int64_t startSym_t0;    //!< Start symbol t0 in nanoseconds
    int64_t toa;            //!< Time of arrival relative to slot boundary in nanoseconds
};

/**
 * Calculate the frame cycle time in nanoseconds
 *
 * @param[in] max_slot_id Maximum slot ID value (0 for 1ms TTI, 1 for 500us TTI)
 * @param[in] opt_tti_us TTI duration in microseconds
 * @return Frame cycle time in nanoseconds
 */
int64_t get_frame_cycle_time_ns(uint16_t max_slot_id, uint16_t opt_tti_us);

/**
 * Get the timestamp of the first frame 0, subframe 0, slot 0
 *
 * @return Timestamp of first F0S0S0 in nanoseconds since epoch
 */
int64_t get_first_f0s0s0_time();

/**
 * Calculate slot start time (t0) from frame/subframe/slot information
 *
 * @param[in] current_time Current timestamp in nanoseconds
 * @param[in] frame_id Frame identifier (0-255)
 * @param[in] subframe_id Subframe identifier (0-9)
 * @param[in] slot_id Slot identifier (0-max_slot_id)
 * @param[in] max_slot_id Maximum slot ID value
 * @param[in] opt_tti_us TTI duration in microseconds
 * @return Slot start time (t0) in nanoseconds
 */
int64_t calculate_t0(int64_t current_time, uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t max_slot_id, uint16_t opt_tti_us);

/**
 * Calculate slot start time (t0) and time-of-arrival (TOA) from packet information
 *
 * Computes both the slot boundary time and the packet's arrival time relative to that boundary
 *
 * @param[in] packet_time Packet receive timestamp in nanoseconds
 * @param[in] first_f0s0s0_time Timestamp of first frame 0, subframe 0, slot 0
 * @param[in] frame_cycle_time_ns Frame cycle duration in nanoseconds
 * @param[in] frame_id Frame identifier (0-255)
 * @param[in] subframe_id Subframe identifier (0-9)
 * @param[in] slot_id Slot identifier (0-max_slot_id)
 * @param[in] start_sym Starting symbol index
 * @param[in] max_slot_id Maximum slot ID value
 * @param[in] opt_tti_us TTI duration in microseconds
 * @return Structure containing slot t0, symbol t0, and TOA values
 */
t0_toa_result calculate_t0_toa(int64_t packet_time, int64_t first_f0s0s0_time, int64_t frame_cycle_time_ns,
                              uint16_t frame_id, uint16_t subframe_id, uint16_t slot_id, uint16_t start_sym,
                              uint16_t max_slot_id, uint16_t opt_tti_us);