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

class FlowData
{
public:
    int flow_id;
    int num_bytes;
    int last_arrival;
};

class FlowType
{
private:
    // Keep statistics about traffic (e.g. arrival times, buffer sizes)
    // Also store QoS config
    int num_bytes = 0;
    int flow_id; // TODO this is never configured
    unsigned long long total_generated_bytes = 0;
    unsigned long long total_accepted_bytes = 0;
    unsigned long long total_dropped_bytes = 0;
public:
    constexpr static int MAX_BYTES = 1e9;
    void Enqueue(FlowData& flow_data){
        int incoming = flow_data.num_bytes > 0 ? flow_data.num_bytes : 0;
        total_generated_bytes += static_cast<unsigned long long>(incoming);
        int free_bytes = MAX_BYTES - num_bytes;
        int accepted = incoming;
        if (accepted > free_bytes) {
            accepted = free_bytes > 0 ? free_bytes : 0;
        }
        int dropped = incoming - accepted;
        num_bytes += accepted;
        total_accepted_bytes += static_cast<unsigned long long>(accepted);
        total_dropped_bytes += static_cast<unsigned long long>(dropped);
    }
    int MoveBytes(){
        auto tmp = num_bytes;
        num_bytes = 0;
        return tmp;
    }
    unsigned long long GetTotalGeneratedBytes() const { return total_generated_bytes; }
    int GetQueuedBytes() const { return num_bytes; }
    unsigned long long GetTotalAcceptedBytes() const { return total_accepted_bytes; }
    unsigned long long GetTotalDroppedBytes() const { return total_dropped_bytes; }
};
