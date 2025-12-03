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

#include "scf_5g_fapi_metrics.hpp"
#include "memtrace.h"

namespace scf_5g_fapi 
{
    void metrics::update_carrier_id(int32_t carrier_id)
    {
#ifdef AERIAL_METRICS
        MemtraceDisableScope md; // disable memtrace while this variable is in scope        
        static std::unordered_map<scf_fapi_message_id_e, std::string> rx_msg_type_names
        {
            {SCF_FAPI_PARAM_REQUEST, "PARAM.request"},
            {SCF_FAPI_CONFIG_REQUEST, "CONFIG.request"},
            {SCF_FAPI_START_REQUEST, "START.request"},
            {SCF_FAPI_STOP_REQUEST, "STOP.request"},
            {SCF_FAPI_DL_TTI_REQUEST, "DL_TTI.request"},
            {SCF_FAPI_UL_TTI_REQUEST, "UL_TTI.request"},
            {SCF_FAPI_UL_DCI_REQUEST, "UL_DCI.request"},
            {SCF_FAPI_TX_DATA_REQUEST, "TX_Data.request"},
            {SCF_FAPI_DL_BFW_CVI_REQUEST, "DLBFW_CVI.request"},
            {SCF_FAPI_UL_BFW_CVI_REQUEST, "ULBFW_CVI.request"},
        };

        static std::unordered_map<scf_fapi_message_id_e, std::string> tx_msg_type_names
        {
            {SCF_FAPI_PARAM_RESPONSE, "PARAM.response"},
            {SCF_FAPI_CONFIG_RESPONSE, "CONFIG.response"},
            {SCF_FAPI_STOP_INDICATION, "STOP.indication"},
            {SCF_FAPI_ERROR_INDICATION, "ERROR.indication"},
            {SCF_FAPI_SLOT_INDICATION, "SLOT.indication"},
            {SCF_FAPI_RX_DATA_INDICATION, "RX_Data.indication"},
            {SCF_FAPI_CRC_INDICATION, "CRC.indication"},
            {SCF_FAPI_UCI_INDICATION, "UCI.indication"},
            {SCF_FAPI_SRS_INDICATION, "SRS.indication"},
            {SCF_FAPI_RACH_INDICATION, "RACH.indication"},
            {SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION, "SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION)"},
            {SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION, "SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION)"},
        };

        carrier_id_ = std::to_string(carrier_id);
        auto &metrics_manager = AerialMetricsRegistrationManager::getInstance();

        for (auto& counter : fapi_tx_packets) {
            counter.second = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_fapi_tx_packets_total", "Aerial cuPHY-CP total number of messages L1 transmits to L2", {{METRIC_CELL_KEY, carrier_id_}, {METRIC_MESSAGE_TYPE_KEY, tx_msg_type_names[counter.first]}});
        }

        for (auto& counter : fapi_rx_packets) {
            counter.second = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_fapi_rx_packets_total", "Aerial cuPHY-CP total number of messages L1 receives from L2", {{METRIC_CELL_KEY, carrier_id_}, {METRIC_MESSAGE_TYPE_KEY, rx_msg_type_names[counter.first]}});
        }

#endif
    }

    void metrics::incr_rx_packet_count(scf_fapi_message_id_e msg_type)
    {
#ifdef AERIAL_METRICS
        if (fapi_rx_packets.find(msg_type) != fapi_rx_packets.end() && fapi_rx_packets[msg_type] != nullptr) {
            fapi_rx_packets[msg_type]->Increment();
        }
#endif
    }

    void metrics::incr_tx_packet_count(scf_fapi_message_id_e msg_type)
    {
#ifdef AERIAL_METRICS
        if (fapi_tx_packets.find(msg_type) != fapi_tx_packets.end() && fapi_tx_packets[msg_type] != nullptr) {
            fapi_tx_packets[msg_type]->Increment();
        }
#endif
    }
}
