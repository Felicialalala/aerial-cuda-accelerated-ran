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

#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>

#include "trafficGenerator.hpp"

class TrafficService
{
public:
    TrafficService(TrafficConfig& config, cumac::cumacCellGrpUeStatus* ue_status)
    {
        generator = std::make_unique<TrafficGenerator>(config);
        radio_rsrc = std::make_unique<RadioResource>(ue_status);
        generator->Attach(radio_rsrc.get());
    }
    TrafficService(TrafficConfig& config, cumac::cumacCellGrpUeStatus* ue_status,cumac::cumacCellGrpUeStatus* ue_status_gpu)
    {
        generator = std::make_unique<TrafficGenerator>(config);
        radio_rsrc = std::make_unique<RadioResourceGpu>(ue_status,ue_status_gpu);
        generator->Attach(radio_rsrc.get());
    }
    void Update(int step=1)
    {
        if (step <= 0) {
            return;
        }
        const int effective_step = step;
        for (int step_idx = 0; step_idx < effective_step; ++step_idx) {
            generator->Generate(1);
            radio_rsrc->AdvanceToTti(generator->GetLastTti());
            generator->Send();
        }
    }
    void Seed(int seed)
    {
        generator->Seed(seed);
    }
    void SetSlotDurationMs(double slot_duration_ms)
    {
        if (slot_duration_ms > 0.0) {
            this->slot_duration_ms = slot_duration_ms;
        }
        radio_rsrc->SetSlotDurationMs(slot_duration_ms);
    }
    void SetPacketTtlTti(int packet_ttl_tti)
    {
        radio_rsrc->SetPacketTtlTti(packet_ttl_tti);
    }
    void SetPacketTtlMs(double packet_ttl_ms)
    {
        if (packet_ttl_ms <= 0.0 || slot_duration_ms <= 0.0) {
            radio_rsrc->SetPacketTtlTti(0);
            return;
        }
        const int packet_ttl_tti = static_cast<int>(std::ceil(packet_ttl_ms / slot_duration_ms));
        radio_rsrc->SetPacketTtlTti(std::max(1, packet_ttl_tti));
    }
    int GetPacketTtlTti() const
    {
        return radio_rsrc->GetPacketTtlTti();
    }
    unsigned long long GetTotalGeneratedBytes() const
    {
        return generator->GetTotalGeneratedBytes();
    }
    unsigned long long GetTotalGeneratedPkts() const
    {
        return generator->GetTotalGeneratedPkts();
    }
    unsigned long long GetTotalAcceptedBytes() const
    {
        return radio_rsrc->GetTotalAcceptedBytes();
    }
    unsigned long long GetTotalDroppedBytes() const
    {
        return radio_rsrc->GetTotalDroppedBytes();
    }
    unsigned long long GetTotalFlowQueuedBytes() const
    {
        return radio_rsrc->GetTotalQueuedBytes();
    }
    unsigned long long GetTotalExpiredBytes() const
    {
        return radio_rsrc->GetTotalExpiredBytes();
    }
    unsigned long long GetTotalExpiredPackets() const
    {
        return radio_rsrc->GetTotalExpiredPackets();
    }
    unsigned long long GetLastExpiredBytes() const
    {
        return radio_rsrc->GetLastExpiredBytes();
    }
    unsigned long long GetLastExpiredPackets() const
    {
        return radio_rsrc->GetLastExpiredPackets();
    }
    void GetPerFlowStats(std::vector<unsigned long long>& generated_bytes,
                         std::vector<unsigned long long>& accepted_bytes,
                         std::vector<unsigned long long>& dropped_bytes,
                         std::vector<unsigned long long>& queued_bytes) const
    {
        radio_rsrc->GetPerFlowStats(generated_bytes, accepted_bytes, dropped_bytes, queued_bytes);
    }
    void GetPerFlowExpiryStats(std::vector<unsigned long long>& expired_bytes,
                               std::vector<unsigned long long>& expired_packets) const
    {
        radio_rsrc->GetPerFlowExpiryStats(expired_bytes, expired_packets);
    }
    void RecordMacServedBytes(const std::vector<unsigned long long>& served_bytes, int current_tti)
    {
        radio_rsrc->RecordServedBytes(served_bytes, current_tti);
    }
    void GetPacketDelayStats(PacketDelaySummary& total,
                             std::vector<PacketDelaySummary>& per_flow) const
    {
        radio_rsrc->GetPacketDelayStats(total, per_flow);
    }
    void GetPacketHeadStats(std::vector<PacketHeadSummary>& per_flow) const
    {
        radio_rsrc->GetPacketHeadStats(per_flow, generator->GetLastTti());
    }
    int GetCurrentTti() const
    {
        return generator->GetLastTti();
    }
private:
    double slot_duration_ms = 0.5;
    std::unique_ptr<TrafficGenerator> generator;
    std::unique_ptr<RadioResource> radio_rsrc;
};
