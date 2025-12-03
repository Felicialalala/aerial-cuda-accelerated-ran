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

#if !defined(NV_PHY_FAPI_MSG_COMMON_HPP_)
#define NV_PHY_FAPI_MSG_COMMON_HPP_ 

#include <stdint.h>
#include <cstring>
#include <cmath>
#include <array>
#include <chrono>

#include "cuphy.h"

namespace nv {
    struct carrier_config {
        /// DL Carrier config
        uint8_t dmrs_typeA_pos;
        uint16_t dl_bandwidth;
        uint32_t dl_freq_abs_A;
        uint16_t dlk0[5];
        uint16_t dl_grid_size[5];
        uint16_t num_tx_ants;
        uint16_t num_tx_port;
        uint16_t dl_fft_size;
        //// UL Carrier config
        uint16_t ul_bandwidth;
        uint32_t ul_freq_abs_A;
        uint16_t ulk0[5];
        uint16_t ul_grid_size[5];
        uint16_t num_rx_ants;
        uint16_t num_rx_port;
        uint16_t ul_fft_size;
        uint8_t freq_shift_7p5_khz;//FrequencyShift7p5KHz
        uint8_t aggregation_level; // need to put this in PDCCH config
    };

    struct cell_config {
        int32_t carrier_idx;
        uint16_t phy_cell_id;
        uint8_t frame_duplex_type;
    };

    struct ssb_config {
        uint32_t ssb_pbch_power;
        uint8_t ssb_scs;
        uint8_t ssb_c_offset;
        /// SCS for initial access 
        uint8_t sub_c_common; // numerology
    };

    static constexpr int NV_MAX_PRACH_FD_OCCASION_NUM = 8; // MAX numPRACHFdOccasions
    static constexpr int NV_MAX_UNUSED_ROOT_SEQUENCE_NUM = 256; // MAX numUnusedRootSequences
    struct prach_root_seq {
        uint16_t seq_index;
        uint8_t number_root_sequence;
        uint16_t k1; // nPrachFreqStart
        uint8_t fdm;
        uint8_t zero_conf;
        uint8_t number_unused_sequence; // for noise estimation
        uint8_t unused_sequence[NV_MAX_UNUSED_ROOT_SEQUENCE_NUM];
        int32_t freqOffset;
    };
 
    // #define NV_MAX_PRACH_FD_OCCASION_NUM 8 // MAX numPRACHFdOccasions
    struct prach_config {
        //prachSequenceLength scf 5G FAPI
        uint8_t prach_seq_length;
        uint8_t prach_conf_index;
        //prachSubCSpacing
        uint8_t prach_scs;
        //restrictedSetConfig - scf 5G nPrachRestrictSet
        uint8_t restricted_set_config;
        //numPrachFdOccasions - scf 5G
        uint8_t num_prach_fd_occasions;
        prach_root_seq root_sequence[NV_MAX_PRACH_FD_OCCASION_NUM];
        uint8_t ssb_per_rach;
        uint8_t multiple_carriers_prach; // prachMultipleCarriersInABand
	    uint8_t start_ro_index;  //start of rach occasion
    };

    struct ssb_table {
        uint8_t ssb_period;
        uint16_t ssb_offset_point_A;
        uint8_t beta_pss;
        uint8_t ssb_sub_carrier_offset;
        uint32_t ssb_mask[2];
        uint8_t mib[3];// 24 bits used 
        uint8_t ss_pbch_multiple_carriers; //ssPbchMultipleCarriersInABand
        uint8_t multiple_cells_pbch;//multipleCellsSsPbchInACarrier
    };

    enum SlotConfig: uint8_t {
        DL_SLOT = 0,
        UL_SLOT = 1,
        GUARD_SLOT = 2
    };
    static constexpr int NV_MAX_TDD_PERIOD = 2;
    static constexpr int NV_MAX_TDD_PERIODICITY = 80;
    

    enum slot_type {
        SLOT_NONE = 0,
        SLOT_UPLINK = 1,
        SLOT_DOWNLINK = 2,
        // Need info from cu
        SLOT_SPECIAL = 3
    };

    struct slot_detail_ {
        slot_type type;
        uint8_t max_dl_symbols;
        uint8_t max_ul_symbols;
        int8_t start_sym_dl;
        int8_t start_sym_ul;
    };

    using slot_detail_t = slot_detail_;

    struct tdd_table {
        uint8_t tdd_period_num;
        std::array<nv::slot_detail_t, nv::NV_MAX_TDD_PERIODICITY> s_detail = {};
    };

    typedef struct {
        uint8_t rssiMeasurement;
        uint8_t rsrpMeasurement;
    } measurement_config_t;

    typedef struct {
        uint8_t pnMeasurement;
        uint8_t pf_234_interference;
        uint8_t prach_interference;
        uint32_t srsChest_buff_size;
        uint8_t pusch_aggr_factor; //!< PUSCH aggregation factor for TTI bundling (valid range: 1-8)
    } vendor_config_t;

    struct phy_config { 
        carrier_config carrier_config_;
        cell_config cell_config_;
        ssb_config ssb_config_;
        prach_config prach_config_;
        ssb_table ssb_table_;
        tdd_table tdd_table_;
        measurement_config_t meas_config_;
        vendor_config_t vendor_config_;
    
        phy_config():
            carrier_config_(),
            cell_config_(),
            ssb_config_(),
            prach_config_(),
            ssb_table_(),
            tdd_table_(),
            meas_config_(),
            vendor_config_() {}

        phy_config(const phy_config&) = delete;
        phy_config& operator=(const phy_config&) = delete;
        phy_config(phy_config&& other) = default;
        phy_config& operator=(phy_config && other) = default;
    };
   struct cell_update_config {
        carrier_config carrier_config_;
        cell_config cell_config_;
        prach_config prach_config_;
    
        cell_update_config():
            carrier_config_(),
            cell_config_(),
            prach_config_() {}

    };

   struct oran_slot_ind {
        uint8_t oslotid_;
        uint8_t osfid_;
        uint8_t oframe_id_;
        uint8_t padding;
    };
    struct slot_indication {
        uint16_t sfn_;
        uint16_t slot_;
        uint64_t tick_;
        slot_indication(): sfn_(0), slot_(0),tick_(0) { }
	    slot_indication( uint16_t sfn, uint16_t slot, uint64_t tick):
            sfn_(sfn), slot_(slot), tick_(tick) {
        }
    };

    enum prach_n_rep
    {
        n_rep_1 = 1,
        n_rep_2 = 2,
        n_rep_4 = 4,
        n_rep_6 = 6,
        n_rep_12 = 12  
    };

    static constexpr uint32_t PRACH_SHORT_FORMAT_FFT = 256;
    static constexpr uint32_t PRACH_LONG_FORMAT_FFT = 1024;
    static constexpr uint32_t PRACH_PREAMBLES = 64;
    enum prach_k_bar
    {
        kbar_1 = 1,
        kbar_2 = 2,
        kbar_7 = 7,
        kbar_10 = 10,
        kbar_12 = 12,
        kbar_133 = 133
    };

    static constexpr float TC = 1/(480.0 * 1000 * 4096);

    typedef struct prach_addln_config_t_
    {
        uint16_t l_ra;
        uint8_t n_ra_slot;
        uint8_t n_ra_dur;
        uint8_t n_ra_rb;
        uint8_t n_ra_t;
    } prach_addln_config_t;

    static constexpr uint16_t TA_OFFSET = 31;
    static constexpr float STEP_CONST = 16 * 64 * nv::TC;

    inline uint16_t get_timing_adv(float& rawTA, uint8_t scs) {
        float step = STEP_CONST/(1 << scs);
        return static_cast<uint16_t>(std::ceil(rawTA/step)) + nv::TA_OFFSET;
    }

    enum nr_bands: uint16_t {
        N1 = 0,
        N2 = 1,
        N3 = 2,
        N5 = 3,
        N7 = 4,
        N8 = 5,
        N12 = 6,
        N20 = 7,
        N25 = 8,
        N28 = 9,
        N34 = 10,
        N38 = 11,
        N39 = 12,
        N40 = 13,
        N41 = 14,
        N50 = 15,
        N51 = 16,
        N66 = 17,
        N70 = 18,
        N71 = 19,
        N74 = 20,
        N75 = 21,
        N76 = 22,
        N77 = 23,
        N78 = 24,
        N79 = 25,
        N80 = 26,
        N81 = 27,
        N82 = 28,
        N83 = 29,
        N84 = 30,
        N85 = 31,
        N86 = 32,
        N257 = 33,
        N258 = 34,
        N260 = 35,
        N261 = 36
    };

    enum ssb_case: uint8_t {
        CASE_A = 0,
        CASE_B = 1,
        CASE_C = 2,
        CASE_D = 3,
        CASE_E = 4,
        CASE_UNKNOWN = 5
    };

    enum duplex: uint8_t
    {
        FDD = 0x00,
        TDD = 0x01,
        SDL = 0x02,
        SUL = 0x03
    };

    struct __attribute__((__packed__)) ss_band_entry
    {
        nr_bands band;
        uint8_t ssb_scs;
        ssb_case block_pattern;
    };

    struct __attribute__((__packed__)) nr_band_entry
    {
        nr_bands band;
        uint16_t ul_band_low;
        uint16_t ul_band_hi;
        uint16_t dl_band_low;
        uint16_t dl_band_hi;
        duplex duplex_type;
    };

    constexpr std::array<nr_band_entry, 36> bands = {{
        {nr_bands::N1, 1920, 1980, 2110, 2170, duplex::FDD},
        {nr_bands::N2, 1850, 1910, 1930, 1990, duplex::FDD},
        {nr_bands::N3, 1710, 1785, 1805, 1880, duplex::FDD},
        {nr_bands::N5, 824, 849, 869, 894, duplex::FDD},
        {nr_bands::N7, 2500, 2570, 2620, 2690, duplex::FDD},
        {nr_bands::N8, 880, 915, 925, 960, duplex::FDD},

        {nr_bands::N12, 699, 716, 729, 746, duplex::FDD},
        {nr_bands::N20, 832, 862, 791, 821, duplex::FDD},
        {nr_bands::N25, 1850, 1915, 1920, 1995, duplex::FDD},
        {nr_bands::N28, 703, 748, 758, 803, duplex::FDD},
        {nr_bands::N34, 2010, 2025, 2010, 2025, duplex::TDD},
        {nr_bands::N38, 2570, 2620, 925, 960, duplex::TDD},

        {nr_bands::N39, 1880, 1920, 1880, 1920, duplex::TDD},
        {nr_bands::N40, 2300, 2400, 2300, 2400, duplex::TDD},
        {nr_bands::N41, 2496, 2690, 2496, 2690, duplex::TDD},
        {nr_bands::N50, 1432, 1517, 1432, 1517, duplex::TDD},
        {nr_bands::N51, 1427, 1432, 1427, 1432, duplex::TDD},
        {nr_bands::N66, 1710, 1780, 2110, 2200, duplex::FDD},

        {nr_bands::N70, 1695, 1710, 1995, 2200, duplex::FDD},
        {nr_bands::N71, 663, 698, 617, 652, duplex::FDD},
        {nr_bands::N74, 1427, 1470, 1475, 1518, duplex::FDD},
        {nr_bands::N75, UINT16_MAX, UINT16_MAX, 1432, 1517, duplex::SDL},
        {nr_bands::N76, UINT16_MAX, UINT16_MAX, 1427, 1432, duplex::SDL},
        {nr_bands::N77, 3300, 4200, 3300, 4200, duplex::TDD},

        {nr_bands::N78, 3300, 3800, 3300, 3800, duplex::TDD},
        {nr_bands::N79, 4400, 5000, 4400, 5000, duplex::TDD},
        {nr_bands::N80, 1710, 1785, UINT16_MAX, UINT16_MAX, duplex::SUL},
        {nr_bands::N81, 880, 915, UINT16_MAX, UINT16_MAX, duplex::SUL},
        {nr_bands::N82, 832, 862, UINT16_MAX, UINT16_MAX, duplex::SUL},

        {nr_bands::N83, 703, 748, UINT16_MAX, UINT16_MAX, duplex::SUL},
        {nr_bands::N84, 1920, 1980, UINT16_MAX, UINT16_MAX, duplex::SUL},
        {nr_bands::N85, 1710, 1780, UINT16_MAX, UINT16_MAX, duplex::SUL},

        {nr_bands::N257, 26500, 29500, 26500, 29500, duplex::TDD},
        {nr_bands::N258, 24250, 27500, 24250, 27500, duplex::TDD},
        {nr_bands::N260, 37000, 40000, 37000, 40000, duplex::TDD},
        {nr_bands::N261, 27500, 28350, 27500, 28350, duplex::TDD}
    }};

    constexpr std::array<ss_band_entry, 37> ss_raster = {{
        {nr_bands::N1, 0, ssb_case::CASE_A},
        {nr_bands::N2, 0, ssb_case::CASE_A},
        {nr_bands::N3, 0, ssb_case::CASE_A},
        {nr_bands::N5, 0, ssb_case::CASE_A},
        {nr_bands::N5, 1, ssb_case::CASE_B},
        
        {nr_bands::N7, 0, ssb_case::CASE_A},
        {nr_bands::N8, 0, ssb_case::CASE_A},
        {nr_bands::N12, 0, ssb_case::CASE_A},
        {nr_bands::N20, 0, ssb_case::CASE_A},
        {nr_bands::N25, 0, ssb_case::CASE_A},

        {nr_bands::N28, 0, ssb_case::CASE_A},
        {nr_bands::N34, 0, ssb_case::CASE_A},
        {nr_bands::N38, 0, ssb_case::CASE_A},
        {nr_bands::N39, 0, ssb_case::CASE_A},
        {nr_bands::N40, 0, ssb_case::CASE_A},

        {nr_bands::N41, 0, ssb_case::CASE_A},
        {nr_bands::N41, 1, ssb_case::CASE_C},
        {nr_bands::N50, 0, ssb_case::CASE_A},
        {nr_bands::N51, 0, ssb_case::CASE_A},
        {nr_bands::N66, 0, ssb_case::CASE_A},

        {nr_bands::N66, 1, ssb_case::CASE_B},
        {nr_bands::N70, 0, ssb_case::CASE_A},
        {nr_bands::N71, 0, ssb_case::CASE_A},
        {nr_bands::N74, 0, ssb_case::CASE_A},
        {nr_bands::N75, 0, ssb_case::CASE_A},

        {nr_bands::N76, 0, ssb_case::CASE_A},
        {nr_bands::N77, 1, ssb_case::CASE_C},
        {nr_bands::N78, 1, ssb_case::CASE_C},
        {nr_bands::N79, 1, ssb_case::CASE_C},

        {nr_bands::N257, 3, ssb_case::CASE_D},
        {nr_bands::N257, 4, ssb_case::CASE_E},
        {nr_bands::N258, 3, ssb_case::CASE_D},
        {nr_bands::N258, 4, ssb_case::CASE_E},
        {nr_bands::N260, 3, ssb_case::CASE_D},

        {nr_bands::N260, 4, ssb_case::CASE_E},
        {nr_bands::N261, 3, ssb_case::CASE_D},
        {nr_bands::N261, 4, ssb_case::CASE_E}
    }};

   using pucch_dtx_t_list = std::array<float, 5>;

   using guard_per_mu_fr1 = std::array<float, 13>;
   using guard_per_mu_fr2 = std::array<float, 4>;
   using guard_per_mu_fr2_240_khz = std::array<float, 3>;

    constexpr std::array<guard_per_mu_fr1, 3> fr1_bw_config_table = {{
    {
        242.5, 312.5, 382.5, 452.5, 522.5, 592.5,
        552.5, 692.5, 0.0, 0.0, 0.0, 0.0, 0.0
    },

    {
        505, 665, 645, 805, 785, 945, 905,
        1045, 825, 965, 925, 885, 845
    },

    { 
        0.0, 1010, 990, 1330, 1310, 1290, 1610,
        1570, 1530, 1490, 1450, 1410, 1370 
    }
   }};

    constexpr std::array<guard_per_mu_fr2, 2> fr2_bw_config_table = {{
    {
        1210, 2450, 4930, 0
    },

    {
        1900, 2420, 4900, 9860
    }
    }};

    constexpr guard_per_mu_fr2_240_khz fr2_240_khz_bw_config_table = {{
        3800, 7720, 15560
    }};
}
#endif
