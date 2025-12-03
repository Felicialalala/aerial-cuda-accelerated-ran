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

#if !defined(SCF_5G_FAPI_METRICS_HPP_INCLUDED_)
#define SCF_5G_FAPI_METRICS_HPP_INCLUDED_

#include <cstring>
#include <string>
#include <cstdio>
#include "scf_5g_fapi.h"

#ifdef AERIAL_METRICS
#include "aerial_metrics.hpp"
using namespace aerial_metrics;
#endif

namespace scf_5g_fapi
{
    class metrics {
        public:
        void update_carrier_id(int32_t carrier_id);
        void incr_rx_packet_count(scf_fapi_message_id_e msg_type);
        void incr_tx_packet_count(scf_fapi_message_id_e msg_type);

        private:
        std::string carrier_id_;

#ifdef AERIAL_METRICS
        std::unordered_map<scf_fapi_message_id_e, prometheus::Counter*> fapi_rx_packets
        {
            {SCF_FAPI_PARAM_REQUEST, nullptr},
            {SCF_FAPI_CONFIG_REQUEST, nullptr},
            {SCF_FAPI_START_REQUEST, nullptr},
            {SCF_FAPI_STOP_REQUEST, nullptr},
            {SCF_FAPI_DL_TTI_REQUEST, nullptr},
            {SCF_FAPI_UL_TTI_REQUEST, nullptr},
            {SCF_FAPI_UL_DCI_REQUEST, nullptr},
            {SCF_FAPI_TX_DATA_REQUEST, nullptr},
            {SCF_FAPI_DL_BFW_CVI_REQUEST, nullptr},
            {SCF_FAPI_UL_BFW_CVI_REQUEST, nullptr},
        };

        std::unordered_map<scf_fapi_message_id_e, prometheus::Counter*> fapi_tx_packets
        {
            {SCF_FAPI_PARAM_RESPONSE, nullptr},
            {SCF_FAPI_CONFIG_RESPONSE, nullptr},
            {SCF_FAPI_STOP_INDICATION, nullptr},
            {SCF_FAPI_ERROR_INDICATION, nullptr},
            {SCF_FAPI_SLOT_INDICATION, nullptr},
            {SCF_FAPI_RX_DATA_INDICATION, nullptr},
            {SCF_FAPI_CRC_INDICATION, nullptr},
            {SCF_FAPI_UCI_INDICATION, nullptr},
            {SCF_FAPI_SRS_INDICATION, nullptr},
            {SCF_FAPI_RACH_INDICATION, nullptr},
        };

        static constexpr char METRIC_MESSAGE_TYPE_KEY[] = "msg_type";
        static constexpr char METRIC_CELL_KEY[] = "cell";
#endif
    };
}

#endif //SCF_5G_FAPI_METRICS_HPP_INCLUDED_