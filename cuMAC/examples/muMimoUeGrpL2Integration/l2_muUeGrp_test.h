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

#include "../../src/cumac.h"
#include "cumac_muUeGrp_api.h"
#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include <yaml-cpp/yaml.h>

// * For simplicity, this L2 stack model uses a single CPU thread for the L2 processing of multiple cells. A real L2 stack implementation may use different CPU threads to handle different cells.

constexpr const char* YAML_PARAM_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/config.yaml"; // path to the system configuration parameters YAML file
constexpr const char* YAML_L2_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/l2_nvipc.yaml"; // path to the L2 NVIPC configuration parameters YAML file

constexpr int MAX_PATH_LEN = 1024; // maximum path length

// Log TAG configured in nvlog
constexpr int TAG = (NVLOG_TAG_BASE_NVIPC + 0);

constexpr uint16_t MAX_NUM_RNTI_PER_CELL = 65535; // max number of RNTIs per cell
constexpr uint16_t MIN_RNTI = 1; // min C-RNTI
constexpr uint16_t MAX_RNTI = 65535; // max C-RNTI

// structure for mapping UE RNTIs to cuMAC 0-based UE IDs
struct ue_id_rnti_t {
    uint16_t id; // cuMAC 0-based UE ID
    uint16_t rnti; // C-RNTI
    uint16_t srsInfoIdx; // index of the SRS info in the SRS info array
    uint8_t flags; // flags for the UE
    // 1st bit (flags & 0x01) - SRS chanEst available, 0: no, 1: yes
    // 2nd bit (flags & 0x02) - has updated SRS info in the current slot, 0: no, 1: yes

    ue_id_rnti_t(uint16_t id_, uint16_t rnti_, uint16_t srsInfoIdx_, uint8_t flags_)
        : id(id_), rnti(rnti_), srsInfoIdx(srsInfoIdx_), flags(flags_) {}
};

// system configuration parameters
struct sys_param_t {
    int num_time_slots; // number of time slots
    int l2_sender_thread_core; // L2 sender thread core
    int l2_recv_thread_core; // L2 receiver thread core
    int cumac_worker_thread_core; // cuMAC worker thread core
    int cumac_recv_thread_core; // cuMAC receiver thread core
    std::string TDD_pattern; // TDD pattern
    int num_cell; // number of cells
    int num_ue_ant_port; // number of antenna ports per UE
    int num_bs_ant_port; // number of antenna ports per RU
    int num_srs_ue_per_cell; // conneccted SRS UEs in each cell
    int num_subband; // number of subbands
    int num_prg_samp_per_subband; // number of per-PRG SRS channel estimate samples per subband
    int num_prg_per_cell; // number of PRGs per cell
    int num_srs_ue_per_slot; // number of SRS UEs scheduled per S-slot
    int max_num_ue_schd_per_cell_tti; // maximum number of UEs scheduled per cell per TTI
    int max_num_ue_for_grp_per_cell; // maximum number of UEs considered for MU-MIMO UE grouping per cell per TTI

    float scs; // subcarrier spacing
    uint64_t slot_interval_ns; // slot interval in nanoseconds

    float srs_chan_est_coeff_var; // * not a system configuration parameter. Used for generating SRS channel estimates.
    
    // CUDA kernel config
    int num_blocks_per_row_chanOrtMat; // cuMAC CUDA kernel configuration parameter

    sys_param_t(const char* yaml_path)
    {
        // load parameters from YAML file
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (config["TDD_PATTERN"]) {
            TDD_pattern = config["TDD_PATTERN"].as<std::string>();
        }
        if (config["NUM_TIME_SLOTS"]) {
            num_time_slots = config["NUM_TIME_SLOTS"].as<int>();
        }
        if (config["L2_SENDER_THREAD_CORE"]) {
            l2_sender_thread_core = config["L2_SENDER_THREAD_CORE"].as<int>();
        }
        if (config["L2_RECV_THREAD_CORE"]) {
            l2_recv_thread_core = config["L2_RECV_THREAD_CORE"].as<int>();
        }
        if (config["cuMAC_WORKER_THREAD_CORE"]) {
            cumac_worker_thread_core = config["cuMAC_WORKER_THREAD_CORE"].as<int>();
        }
        if (config["cuMAC_RECV_THREAD_CORE"]) {
            cumac_recv_thread_core = config["cuMAC_RECV_THREAD_CORE"].as<int>();
        }
        if (config["NUM_CELL"]) {
            num_cell = config["NUM_CELL"].as<int>();
        }
        if (config["NUM_UE_ANT_PORT"]) {
            num_ue_ant_port = config["NUM_UE_ANT_PORT"].as<int>();
        }
        if (config["NUM_BS_ANT_PORT"]) {
            num_bs_ant_port = config["NUM_BS_ANT_PORT"].as<int>();
        }
        if (config["NUM_SRS_UE_PER_CELL"]) {
            num_srs_ue_per_cell = config["NUM_SRS_UE_PER_CELL"].as<int>();
        }
        if (config["NUM_SUBBAND"]) {
            num_subband = config["NUM_SUBBAND"].as<int>();
        }
        if (config["NUM_PRG_SAMP_PER_SUBBAND"]) {
            num_prg_samp_per_subband = config["NUM_PRG_SAMP_PER_SUBBAND"].as<int>();
        }
        if (config["NUM_PRG_PER_CELL"]) {
            num_prg_per_cell = config["NUM_PRG_PER_CELL"].as<int>();
        }
        if (config["NUM_SRS_UE_PER_SLOT"]) {
            num_srs_ue_per_slot = config["NUM_SRS_UE_PER_SLOT"].as<int>();
        }
        if (config["MAX_NUM_UE_SCHEDULED_PER_CELL_TTI"]) {
            max_num_ue_schd_per_cell_tti = config["MAX_NUM_UE_SCHEDULED_PER_CELL_TTI"].as<int>();
        }
        if (config["MAX_NUM_UE_FOR_GRP_PER_CELL"]) {
            max_num_ue_for_grp_per_cell = config["MAX_NUM_UE_FOR_GRP_PER_CELL"].as<int>();
        }
        if (config["SCS"]) {
            scs = config["SCS"].as<float>();
        }
        if (config["SRS_CHAN_EST_COEFF_VAR"]) {
            srs_chan_est_coeff_var = config["SRS_CHAN_EST_COEFF_VAR"].as<float>();
        }
        if (config["NUM_BLOCKS_PER_ROW_CHAN_OR_MAT"]) {
            num_blocks_per_row_chanOrtMat = config["NUM_BLOCKS_PER_ROW_CHAN_OR_MAT"].as<int>();
        }
        if (config["SLOT_INTERVAL_NS"]) {
            slot_interval_ns = config["SLOT_INTERVAL_NS"].as<uint64_t>();
        }

        // validate parameters
        if (num_cell > MAX_NUM_CELL) {
            throw std::runtime_error("num_cell > MAX_NUM_CELL");
        }
        if (num_srs_ue_per_slot > MAX_NUM_UE_SRS_INFO_PER_SLOT) {
            throw std::runtime_error("num_srs_ue_per_slot > MAX_NUM_UE_SRS_INFO_PER_SLOT");
        }
        if (num_subband > MAX_NUM_SUBBAND) {
            throw std::runtime_error("num_subband > MAX_NUM_SUBBAND");
        }
        if (num_prg_samp_per_subband > MAX_NUM_PRG_SAMP_PER_SUBBAND) {
            throw std::runtime_error("num_prg_samp_per_subband > MAX_NUM_PRG_SAMP_PER_SUBBAND");
        }
        if (num_prg_per_cell > MAX_NUM_PRG) {
            throw std::runtime_error("num_prg_per_cell > MAX_NUM_PRG");
        }
        if (num_srs_ue_per_cell > MAX_NUM_SRS_UE_PER_CELL) {
            throw std::runtime_error("num_srs_ue_per_cell > MAX_NUM_SRS_UE_PER_CELL");
        }
        if (num_ue_ant_port > MAX_NUM_UE_ANT_PORT) {
            throw std::runtime_error("num_ue_ant_port > MAX_NUM_UE_ANT_PORT");
        }
        if (num_bs_ant_port > MAX_NUM_BS_ANT_PORT) {
            throw std::runtime_error("num_bs_ant_port > MAX_NUM_BS_ANT_PORT");
        }
    }
};

// class for managing the connected SRS UEs in a cell
// * this class is used for illustration purpose. A real L2 stack implementation may differ in logic and data structure.
class l2_connected_ue_list_t {
    public:
        l2_connected_ue_list_t(const int cell_id, const int num_srs_ue){
            m_cell_id = cell_id; // cell ID
            m_num_srs_ue = num_srs_ue; // number of SRS UEs in this cell

            // initialize the available RNTIs 
            for (uint16_t rnti = MIN_RNTI; rnti < (MAX_NUM_SRS_UE_PER_CELL+MIN_RNTI); rnti++) {
                available_rnti.push_back(rnti);
            }

            // initialize the available cuMAC 0-based IDs
            for (uint16_t i = 0; i < MAX_NUM_SRS_UE_PER_CELL; i++) {
                available_id.push_back(i);
            }

            // initialize the connected UE list and the SRS schedule queue
            for (int i = 0; i < m_num_srs_ue; i++) {
                uint16_t id = available_id.front();
                available_id.pop_front();
                uint16_t rnti = available_rnti.front();
                available_rnti.pop_front();
                srs_ue_list.push_back(ue_id_rnti_t(id, rnti, 0xFFFF, 0x00));
                srs_schedule_queue.push_back(ue_id_rnti_t(id, rnti, 0xFFFF, 0x00));
            }
        }
        ~l2_connected_ue_list_t() {}
        
        bool ue_arrival(const int num_ue_arrivel) { // add new arrived UEs and assign cuMAC 0-based IDs and RNTIs
            for (int i = 0; i < num_ue_arrivel; i++) {
                if (available_id.empty() || available_rnti.empty()) {
                    printf("No available ID or RNTI for UE arrival in cell %d\n", m_cell_id);
                    return false;
                }

                if (srs_ue_list.size() >= MAX_NUM_SRS_UE_PER_CELL) {
                    printf("Maximum number of SRS UEs reached in cell %d\n", m_cell_id);
                    return false;
                }

                uint16_t id = available_id.front();
                available_id.pop_front();
                uint16_t rnti = available_rnti.front();
                available_rnti.pop_front();
                srs_ue_list.push_back(ue_id_rnti_t(id, rnti, 0xFFFF, 0x00));
                srs_schedule_queue.push_front(ue_id_rnti_t(id, rnti, 0xFFFF, 0x00));
                m_num_srs_ue++;
            }

            return true;
        }

        bool ue_departure(uint16_t rnti) { // remove UE from the connected UE list with a given RNTI
            // validate RNTI value
            if (rnti < MIN_RNTI || rnti > MAX_RNTI) {
                printf("RNTI %d is out of range\n", rnti);
                return false;
            }

            auto it = std::find_if(srs_ue_list.begin(), srs_ue_list.end(), [rnti](const ue_id_rnti_t& p) {
                return p.rnti == rnti;
            });

            if (it != srs_ue_list.end()) {
                available_id.push_front(it->id);
                available_rnti.push_front(it->rnti);
                srs_ue_list.erase(it);
                m_num_srs_ue--;

                auto it = std::find_if(srs_schedule_queue.begin(), srs_schedule_queue.end(), [rnti](const ue_id_rnti_t& p) {
                    return p.rnti == rnti;
                });

                if (it != srs_schedule_queue.end()) {
                    srs_schedule_queue.erase(it);
                }
                return true;
            } else {
                printf("UE with RNTI %d is not found in the connected UE list of cell %d\n", rnti, m_cell_id);
                return false;
            }
        }

        uint16_t get_num_connected_srs_ue() { // get the number of connected SRS UEs
            if (srs_ue_list.size() != m_num_srs_ue || srs_schedule_queue.size() != m_num_srs_ue) {
                printf("Number of SRS UEs in the connected UE list of cell %d is not consistent with the number of SRS UEs in the SRS schedule queue. Current number of SRS UEs connected: %d, number of SRS UEs in the SRS schedule queue: %d\n", m_cell_id, srs_ue_list.size(), srs_schedule_queue.size());
                return 0xFFFF;
            }
            return srs_ue_list.size();
        }

        uint16_t get_id(uint16_t rnti) { // get the cuMAC 0-based ID of a UE with a given RNTI
            // validate RNTI value
            if (rnti < MIN_RNTI || rnti > MAX_RNTI) {
                printf("RNTI %d is out of range\n", rnti);
                return 0xFFFF;
            }

            auto it = std::find_if(srs_ue_list.begin(), srs_ue_list.end(), [rnti](const ue_id_rnti_t& p) {
                return p.rnti == rnti;
            });

            if (it != srs_ue_list.end()) {    
                return it->id;
            } else {
                printf("UE with RNTI %d is not found in the connected UE list of cell %d\n", rnti, m_cell_id);
                return 0xFFFF;
            }
        }

        uint16_t get_ue_idx_in_list(uint16_t rnti) { // get the index of a UE in the connected UE list with a given RNTI
            auto it = std::find_if(srs_ue_list.begin(), srs_ue_list.end(), [rnti](const ue_id_rnti_t& p) {
                return p.rnti == rnti;
            });

            if (it != srs_ue_list.end()) {
                return std::distance(srs_ue_list.begin(), it);
            } else {
                printf("UE with RNTI %d is not found in the connected UE list of cell %d\n", rnti, m_cell_id);
                return 0xFFFF;
            }
        }

        void set_srs_info(uint16_t rnti, uint16_t srs_info_idx) { // set the SRS info index of a UE with a given RNTI
            auto it = std::find_if(srs_ue_list.begin(), srs_ue_list.end(), [rnti](const ue_id_rnti_t& p) {
                return p.rnti == rnti;
            });

            if (it != srs_ue_list.end()) {
                it->srsInfoIdx = srs_info_idx;
                it->flags = 0x03; // SRS chanEst available and has updated SRS info in the current slot
            } else {
                printf("UE with RNTI %d is not found in the connected UE list of cell %d\n", rnti, m_cell_id);
            }
        }

        void set_num_srsInfo(int num_srsInfo) { // set the number of SRS info for the current slot
            m_num_srsInfo = num_srsInfo;
        }

        int get_num_srsInfo() { // get the number of SRS info for the current slot
            return m_num_srsInfo;
        }

        void set_flags(uint16_t rnti, uint8_t flags) { // set the flags of a UE with a given RNTI
            auto it = std::find_if(srs_ue_list.begin(), srs_ue_list.end(), [rnti](const ue_id_rnti_t& p) {
                return p.rnti == rnti;
            });

            if (it != srs_ue_list.end()) {
                it->flags = flags;
            } else {
                printf("UE with RNTI %d is not found in the connected UE list of cell %d\n", rnti, m_cell_id);
            }
        }

        ue_id_rnti_t get_ue_id_rnti(uint16_t ue_idx) { // get the cuMAC 0-based ID and RNTI of a UE
            if (ue_idx >= srs_ue_list.size()) {
                printf("UE index %d is out of range in the connected UE list of cell %d. Current number of SRS UEs connected: %d\n", ue_idx, m_cell_id, srs_ue_list.size());
                return ue_id_rnti_t(0xFFFF, 0xFFFF, 0xFFFF, 0xFF);
            }
            return srs_ue_list[ue_idx];
        }

        ue_id_rnti_t get_next_srs_schedule() { // get the next SRS UE in the SRS schedule queue
            if (srs_schedule_queue.empty()) {
                printf("SRS schedule queue is empty in cell %d\n", m_cell_id);
                return ue_id_rnti_t(0xFFFF, 0xFFFF, 0xFFFF, 0xFF);
            }
            ue_id_rnti_t ue_id_rnti = srs_schedule_queue.front();
            srs_schedule_queue.pop_front();
            srs_schedule_queue.push_back(ue_id_rnti);
            return ue_id_rnti;
        }

    private:
        std::vector<ue_id_rnti_t> srs_ue_list; // list of SRS UEs in this cell (with cell ID m_cell_id)
        std::deque<uint16_t> available_rnti; // queue of available RNTIs
        std::deque<uint16_t> available_id; // queue of available cuMAC 0-based UE IDs

        std::deque<ue_id_rnti_t> srs_schedule_queue; // queue for SRS scheduling

        int m_cell_id; // cell ID
        int m_num_srs_ue; // number of SRS UEs in this cell
        int m_num_srsInfo; // number of SRS info for the current slot in this cell
};

// function for getting the next slot timespec
void get_next_slot_timespec(struct timespec* ts, uint64_t interval_nsec)
{
    ts->tv_nsec += interval_nsec;
    while (ts->tv_nsec >= 1E9)
    {
        ts->tv_nsec -= 1E9;
        ts->tv_sec++;
    }
}

// function for preparing the slot data
void prepare_slot_data(const sys_param_t& sys_param, std::vector<std::unique_ptr<l2_connected_ue_list_t>>& connected_ue_list_vec, std::vector<cumac_muUeGrp_req_info_t*> req_data, const int slot_idx);

// function for getting the name of a cuMAC message
const char* get_cumac_msg_name(int msg_id)
{
    switch(msg_id)
    {
    case CUMAC_PARAM_REQUEST:
        return "PARAM.req";
    case CUMAC_PARAM_RESPONSE:
        return "PARAM.resp";

    case CUMAC_CONFIG_REQUEST:
        return "CONFIG.req";
    case CUMAC_CONFIG_RESPONSE:
        return "CONFIG.resp";

    case CUMAC_START_REQUEST:
        return "START.req";
    case CUMAC_START_RESPONSE:
        return "START.resp";

    case CUMAC_STOP_REQUEST:
        return "STOP.req";
    case CUMAC_STOP_RESPONSE:
        return "STOP.resp";

    case CUMAC_ERROR_INDICATION:
        return "ERR.ind";

    case CUMAC_TTI_ERROR_INDICATION:
        return "TTI_ERR.ind";
    case CUMAC_DL_TTI_REQUEST:
        return "DL_TTI.req";
    case CUMAC_UL_TTI_REQUEST:
        return "UL_TTI.req";

    case CUMAC_SCH_TTI_REQUEST:
        return "SCH_TTI.req";
    case CUMAC_SCH_TTI_RESPONSE:
        return "SCH_TTI.resp";

    case CUMAC_TTI_END:
        return "TTI_END.req";

    default:
        return "UNKNOWN_CUMAC_MSG";
    }
}