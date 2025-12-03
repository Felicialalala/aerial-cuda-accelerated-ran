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

#include "metrics.hpp"

#include "dpdk.hpp"
#include "nic.hpp"
#include "peer.hpp"
#include "utils.hpp"

#define TAG "FH.METRICS"

namespace aerial_fh
{
PeerMetrics::PeerMetrics(Peer* peer, PeerId id) :
    peer_{peer},
    peer_key_name_{std::to_string(id)}
{
#ifdef AERIAL_METRICS
    auto& metrics_manager = AerialMetricsRegistrationManager::getInstance();

    std::map<std::string, std::string> labels{{kMetricPeerKey, peer_key_name_}};

    counters_[static_cast<size_t>(PeerMetric::kCPlaneTxBytes)]   = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_cplane_tx_bytes_total", "Aerial cuPHY-CP C-plane TX bytes", labels);
    counters_[static_cast<size_t>(PeerMetric::kCPlaneTxPackets)] = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_cplane_tx_packets_total", "Aerial cuPHY-CP C-plane TX packets", labels);
    counters_[static_cast<size_t>(PeerMetric::kUPlaneTxBytes)]   = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_tx_bytes_total", "Aerial cuPHY-CP U-plane TX bytes", labels);
    counters_[static_cast<size_t>(PeerMetric::kUPlaneTxPackets)] = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_tx_packets_total", "Aerial cuPHY-CP U-plane TX packets", labels);
    counters_[static_cast<size_t>(PeerMetric::kUPlaneRxBytes)]   = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_rx_bytes_total", "Aerial cuPHY-CP U-plane RX bytes", labels);
    counters_[static_cast<size_t>(PeerMetric::kUPlaneRxPackets)] = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_rx_packets_total", "Aerial cuPHY-CP U-plane RX packets", labels);
#endif
}

void PeerMetrics::update(PeerMetric counter, uint64_t value)
{
#ifdef AERIAL_METRICS
    counters_[static_cast<size_t>(counter)]->Increment(value);
#endif
}

NicMetrics::NicMetrics(Nic* nic) :
    nic_{nic}
{
#ifdef AERIAL_METRICS
    auto  name            = nic_->get_name();
    auto& metrics_manager = AerialMetricsRegistrationManager::getInstance();

    std::map<std::string, std::string> labels{{kMetricNicKey, name}};

    rx_missed_  = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_rx_dropped_packets_total", "Aerial cuPHY-CP RX packets dropped by the HW", labels);
    rx_errors_  = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_rx_failed_packets_total", "Aerial cuPHY-CP erroneous RX packets", labels);
    rx_nombufs_ = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_rx_nombuf_packets_total", "Aerial cuPHY-CP RX mbuf allocation failures", labels);
    tx_errors_  = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_failed_packets_total", "Aerial cuPHY-CP failed TX packets", labels);

    tx_pp_missed_interrupt_errors_ = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_accu_sched_missed_interrupt_errors_total", "Aerial cuPHY-CP accurate TX scheduling missed service interrupts", labels);
    tx_pp_rearm_queue_errors_      = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_accu_sched_rearm_queue_errors_total", "Aerial cuPHY-CP TX scheduling rearm queue errors", labels);
    tx_pp_clock_queue_errors_      = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_accu_sched_clock_queue_errors_total", "Aerial cuPHY-CP TX scheduling clock queue errors", labels);
    tx_pp_timestamp_past_errors_   = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_accu_sched_timestamp_past_errors_total", "Aerial cuPHY-CP TX scheduling timestamp in the past", labels);
    tx_pp_timestamp_future_errors_ = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_net_tx_accu_sched_timestamp_future_errors_total", "Aerial cuPHY-CP TX scheduling timestamp in too distant future", labels);
    tx_pp_jitter_                  = &metrics_manager.addMetric<prometheus::Gauge>("aerial_cuphycp_net_tx_accu_sched_clock_queue_jitter_ns", "Aerial cuPHY-CP TX scheduling timestamp jitter", labels);
    tx_pp_wander_                  = &metrics_manager.addMetric<prometheus::Gauge>("aerial_cuphycp_net_tx_accu_sched_clock_queue_wander_ns", "Aerial cuPHY-CP TX scheduling timestamp wander", labels);
#endif
}

void NicMetrics::cache_metric_ids()
{
#ifdef AERIAL_METRICS
    auto name    = nic_->get_name();
    auto port_id = nic_->get_port_id();
    int  ret     = 0;

    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_missed_interrupt_errors", &xstat_id_cache[0]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_rearm_queue_errors", &xstat_id_cache[1]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_clock_queue_errors", &xstat_id_cache[2]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_timestamp_past_errors", &xstat_id_cache[3]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_timestamp_future_errors", &xstat_id_cache[4]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_jitter", &xstat_id_cache[5]);
    ret |= rte_eth_xstats_get_id_by_name(port_id, "tx_pp_wander", &xstat_id_cache[6]);

    if(ret)
    {
        THROW_FH(ENOENT, StringBuilder() << "Failed to create NIC " << name << " extended metrics");
    }

#endif
}

void NicMetrics::update()
{
#ifdef AERIAL_METRICS
    auto name    = nic_->get_name();
    auto port_id = nic_->get_port_id();

    rte_eth_stats stats;

    auto ret = rte_eth_stats_get(port_id, &stats);
    if(unlikely(ret))
    {
        THROW_FH(ENOENT, StringBuilder() << "Failed to get NIC " << name << " metrics");
    }

    rx_missed_->Increment(stats.imissed - rx_missed_->Value());
    rx_errors_->Increment(stats.ierrors - rx_errors_->Value());
    rx_nombufs_->Increment(stats.rx_nombuf - rx_nombufs_->Value());
    tx_errors_->Increment(stats.oerrors - tx_errors_->Value());

    std::array<uint64_t, kMetricNicXCount> xstat_values;

    ret = rte_eth_xstats_get_by_id(port_id, xstat_id_cache.data(), xstat_values.data(), kMetricNicXCount);
    if(unlikely(ret != kMetricNicXCount))
    {
        THROW_FH(ENOENT, StringBuilder() << "Failed to update NIC " << name << " extended metrics");
    }

    tx_pp_missed_interrupt_errors_->Increment(xstat_values[0] - tx_pp_missed_interrupt_errors_->Value());
    tx_pp_rearm_queue_errors_->Increment(xstat_values[1] - tx_pp_rearm_queue_errors_->Value());
    tx_pp_clock_queue_errors_->Increment(xstat_values[2] - tx_pp_clock_queue_errors_->Value());
    tx_pp_timestamp_past_errors_->Increment(xstat_values[3] - tx_pp_timestamp_past_errors_->Value());
    tx_pp_timestamp_future_errors_->Increment(xstat_values[4] - tx_pp_timestamp_future_errors_->Value());
    tx_pp_jitter_->Set(xstat_values[5]);
    tx_pp_wander_->Set(xstat_values[6]);
#endif
}
} // namespace aerial_fh
