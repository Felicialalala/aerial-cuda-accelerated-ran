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

#ifndef AERIAL_FH_FLOW_HPP__
#define AERIAL_FH_FLOW_HPP__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"
#include "queue.hpp"
#include "aerial-fh-driver/fh_mutex.hpp"

#include <atomic>

namespace aerial_fh
{
class Fronthaul;
class Nic;
class Peer;

class SequenceIdGenerator {
    /// Sequence ID counter
    /// @note NOT thread-safe: Change to std::atomic<uint8_t> if multiple threads
    ///       will call next() on the same SequenceIdGenerator instance concurrently
    uint8_t ecpriSeqId_{0};

public:
    uint8_t next();
    uint8_t next(uint8_t num);
};

struct PacketHeaderTemplate
{
    rte_ether_hdr  eth;
    rte_vlan_hdr   vlan;
    oran_ecpri_hdr ecpri;
} __attribute__((packed));

class Flow {
public:
    Flow(Peer* peer, FlowInfo* flow_info);
    ~Flow();

    Peer*                 get_peer() const;
    Nic*                  get_nic() const;
    Fronthaul*            get_fronthaul() const;
    FlowInfo const&       get_info() const;
    PacketHeaderTemplate& get_packet_header_template();
    SequenceIdGenerator&  get_sequence_id_generator_uplink();
    SequenceIdGenerator&  get_sequence_id_generator_downlink();
    void                  update(FlowInfo const* flow_info);

    /* RX API */
    void receive(MsgReceiveInfo* info, size_t* num_msgs);
    void receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout);

    /* GPU-init comm */
    void                    setup_packet_header_gpu();
    uint8_t*                get_packet_header_gpu();
    uint8_t*                get_next_header_gpu(uint32_t num_pkts);
    uint32_t                get_packet_header_gpu_lkey();
    PacketHeaderTemplate *get_header_template() { return &pkt_hdr_template_; }
    uint32_t get_next_header_stride_gpu(uint32_t num_pkts);

protected:
    Peer*                 peer_{};
    FlowInfo              info_;
    PacketHeaderTemplate  pkt_hdr_template_;
    SequenceIdGenerator   ecpriSeqId_generator_uplink_;
    SequenceIdGenerator   ecpriSeqId_generator_downlink_;
    Rxq*                  rxq_{};
    RxFlowRulesUnique     rx_flow_rules_;
    std::string           flow_type_{};
    aerial_fh::FHMutex            mtx_{};


    int                   flow_number;
    uint8_t *             pkt_hdr_gpu_;
    uint32_t              pkt_hdr_gpu_lkey_;
    uint32_t              pkt_header_idx_;
    size_t packet_size_rnd = 0;

    void setup_packet_header_template();
    void create_rx_rules();
    void create_rx_rule();
    void create_rx_rule_with_cpu_mirroring();
    void create_rx_rule_for_uplane();
    void setup_flow_index();
    void request_nic_resources();
    void free_nic_resources();
private:
    int GPU_device_count = 0;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_TIME_HPP__
