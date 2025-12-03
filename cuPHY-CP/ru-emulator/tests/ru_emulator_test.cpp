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

#include <gtest/gtest.h>
#include "ru_emulator.hpp"
#include "timing_utils.hpp"

// Test fixture for RU Emulator tests
class RUEmulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code for all tests
    }

    void TearDown() override {
        // Common cleanup code for all tests
    }
};

// Test t0/toa calculation
TEST_F(RUEmulatorTest, CalculateT0ToA) {
    // Test case 1: Basic calculation
    int64_t packet_time = 1000000;  // 1ms
    int64_t beginning_of_time = 0;
    int64_t frame_cycle_time_ns = 10000000;  // 10ms
    uint16_t frame_id = 0;
    uint16_t subframe_id = 0;
    uint16_t slot_id = 0;
    uint16_t start_sym = 0;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;  // 1ms

    auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                 frame_id, subframe_id, slot_id, start_sym,
                                 max_slot_id, opt_tti_us);

    // Verify the results
    EXPECT_GE(result.slot_t0, beginning_of_time);
    EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    EXPECT_GE(result.toa, -frame_cycle_time_ns/2);
}

// Test t0/toa calculation with different frame/slot combinations
TEST_F(RUEmulatorTest, CalculateT0ToAWithDifferentFrames) {
    int64_t packet_time = 1000000;
    int64_t beginning_of_time = 0;
    int64_t frame_cycle_time_ns = 10000000;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;

    // Test different frame/subframe/slot combinations
    struct TestCase {
        uint16_t frame_id;
        uint16_t subframe_id;
        uint16_t slot_id;
        uint16_t start_sym;
    };

    std::vector<TestCase> test_cases = {
        {0, 0, 0, 0},    // First frame, first subframe, first slot
        {0, 0, 9, 0},    // First frame, first subframe, last slot
        {0, 9, 0, 0},    // First frame, last subframe, first slot
        {1, 0, 0, 0},    // Second frame, first subframe, first slot
        {0, 0, 0, 7},    // First frame, first subframe, first slot, middle symbol
    };

    for (const auto& tc : test_cases) {
        // Calculate time offset based on frame, subframe, slot and symbol
        int64_t time_offset = ((int64_t)tc.frame_id * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) + 
                              (int64_t)tc.subframe_id * (max_slot_id + 1) + 
                              (int64_t)tc.slot_id);
        time_offset *= opt_tti_us * NS_X_US;
        time_offset += (int)(opt_tti_us * NS_X_US * (float)tc.start_sym / ORAN_ALL_SYMBOLS);

        // Adjust beginning_of_time to be in the correct range relative to time_offset
        int64_t adjusted_beginning_of_time = beginning_of_time;
        while(packet_time < adjusted_beginning_of_time + time_offset) {
            adjusted_beginning_of_time -= frame_cycle_time_ns;
        }
        while(packet_time > adjusted_beginning_of_time + time_offset + frame_cycle_time_ns) {
            adjusted_beginning_of_time += frame_cycle_time_ns;
        }

        auto result = calculate_t0_toa(packet_time, adjusted_beginning_of_time, frame_cycle_time_ns,
                                     tc.frame_id, tc.subframe_id, tc.slot_id, tc.start_sym,
                                     max_slot_id, opt_tti_us);

        // Verify the results
        EXPECT_GE(result.slot_t0, adjusted_beginning_of_time);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
        EXPECT_GE(result.toa, -frame_cycle_time_ns/2);

        // Verify time offset calculation
        EXPECT_EQ(result.slot_t0 - adjusted_beginning_of_time, time_offset);
    }
}

// Test t0/toa calculation with edge cases
TEST_F(RUEmulatorTest, CalculateT0ToAEdgeCases) {
    int64_t frame_cycle_time_ns = 10000000;
    uint16_t max_slot_id = 9;
    uint16_t opt_tti_us = 1000;

    // Test case 1: Packet time very close to frame boundary
    {
        int64_t packet_time = frame_cycle_time_ns - 1;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    }

    // Test case 2: Packet time just after frame boundary
    {
        int64_t packet_time = frame_cycle_time_ns + 1;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_GE(result.toa, -frame_cycle_time_ns/2);
    }

    // Test case 3: Very large frame number
    {
        int64_t packet_time = 1000000;
        int64_t beginning_of_time = 0;
        auto result = calculate_t0_toa(packet_time, beginning_of_time, frame_cycle_time_ns,
                                     0xFFFF, 0, 0, 0, max_slot_id, opt_tti_us);
        EXPECT_LE(result.toa, frame_cycle_time_ns/2);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 