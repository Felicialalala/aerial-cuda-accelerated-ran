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

#ifndef AERIAL_FH_METRICS_HPP__
#define AERIAL_FH_METRICS_HPP__

#include "aerial-fh-driver/api.hpp"
#include "defaults.hpp"
#include "utils.hpp"

#ifdef AERIAL_METRICS
#include "aerial_metrics.hpp"
using namespace aerial_metrics;
#endif

namespace aerial_fh
{
class Peer;
class Nic;

enum class PeerMetric
{
    kCPlaneTxBytes = 0,
    kCPlaneTxPackets,
    kUPlaneTxBytes,
    kUPlaneTxPackets,
    kUPlaneRxBytes,
    kUPlaneRxPackets,
    kPeerMetricMax
};

class PeerMetrics {
public:
    PeerMetrics(Peer* peer, PeerId id);
    void update(PeerMetric counter, uint64_t value);

protected:
    Peer*       peer_{};
    std::string peer_key_name_{};

#ifdef AERIAL_METRICS
    std::array<prometheus::Counter*, (size_t)PeerMetric::kPeerMetricMax> counters_{};
#endif
};

using NicXMetricId = uint64_t;

class NicMetrics {
public:
    NicMetrics(Nic* nic);
    void cache_metric_ids();
    void update();

protected:
    Nic* nic_;

#ifdef AERIAL_METRICS
    // Generic counters
    prometheus::Counter* rx_missed_;
    prometheus::Counter* rx_errors_;
    prometheus::Counter* rx_nombufs_;
    prometheus::Counter* tx_errors_;

    // Accurate send scheduling metrics
    prometheus::Counter* tx_pp_missed_interrupt_errors_;
    prometheus::Counter* tx_pp_rearm_queue_errors_;
    prometheus::Counter* tx_pp_clock_queue_errors_;
    prometheus::Counter* tx_pp_timestamp_past_errors_;
    prometheus::Counter* tx_pp_timestamp_future_errors_;
    prometheus::Gauge*   tx_pp_jitter_;
    prometheus::Gauge*   tx_pp_wander_;
#endif

    std::array<NicXMetricId, kMetricNicXCount> xstat_id_cache;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_METRICS_HPP__