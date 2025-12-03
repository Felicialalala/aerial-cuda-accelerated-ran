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

#ifndef _TEST_MAC_STATS_HPP
#define _TEST_MAC_STATS_HPP

typedef struct{
    uint32_t preamble_id;
    float timing_advance_range_low;
    float timing_advance_range_high;
    uint32_t prach_occassion;
    uint32_t preamble_detected;
    uint32_t preamble_error;
    uint32_t preamble_timing_offset_error;
}prach_conformance_test_t;

typedef struct{
    uint32_t pf0_occassion;
    uint32_t pf0_ack;
    uint32_t pf0_nack;
    uint32_t pf0_dtx;
}pf0_conformance_test_t;

typedef struct{
    uint32_t pf1_occassion;
    uint8_t pf1_ack_nack_pattern;
    uint32_t pf1_ack_bits;
    uint32_t pf1_nack_bits;
    uint32_t pf1_dtx_bits;
    uint32_t pf1_nack_to_ack_bits;
    uint32_t pf1_ack_to_nack_bits;
    uint32_t pf1_ack_to_dtx_bits;
    uint32_t pf1_nack_to_dtx_bits;
}pf1_conformance_test_t;

typedef struct{
    uint32_t pf2_harq_occassion;
    uint32_t pf2_csi_occassion;
    uint32_t pf2_harq_ack_bits;
    uint32_t pf2_harq_nack_bits;
    uint32_t pf2_bler;
    uint32_t pf2_csi_success;
}pf2_conformance_test_t;

typedef struct{
    uint32_t pf3_occassion;
    uint32_t pf3_csi_success;
    uint32_t pf3_bler;
}pf3_conformance_test_t;


class ch8_conformance_test_stats {
public:
    ch8_conformance_test_stats()
    {
    }

    ~ch8_conformance_test_stats()
    {
    }
    
    prach_conformance_test_t & get_prach_stats()
    {
       return prach_stats;
    }
    
    pf0_conformance_test_t & get_pf0_stats()
    {
       return pf0_stats;
    }  

    pf1_conformance_test_t & get_pf1_stats()
    {
       return pf1_stats;
    }

    pf2_conformance_test_t & get_pf2_stats()
    {
       return pf2_stats;
    }
    pf3_conformance_test_t & get_pf3_stats()
    {
       return pf3_stats;
    }
private:
    //TS 38.141-1 chapter 8 conformance test stats
    prach_conformance_test_t prach_stats = {};
    pf0_conformance_test_t pf0_stats = {};
    pf1_conformance_test_t pf1_stats = {};
    pf2_conformance_test_t pf2_stats = {};
    pf3_conformance_test_t pf3_stats = {};
};
#endif

