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

#include "cell.hpp"

namespace fh_gen
{

Cell::Cell(FhGenerator* _fhgen, GpuDevice* _gDev, const PeerInfo& _peer_info) : gDev(_gDev), fhgen(_fhgen), peer_info(_peer_info)
{
}

Cell::~Cell()
{
}

void Cell::allocate_buffers()
{
    next_slot_on_time_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    cudaMemset((uint32_t*)next_slot_on_time_rx_packets->addr(), 0, sizeof(uint32_t));
    next_slot_early_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    cudaMemset((uint32_t*)next_slot_early_rx_packets->addr(), 0, sizeof(uint32_t));
    next_slot_late_rx_packets.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    cudaMemset((uint32_t*)next_slot_late_rx_packets->addr(), 0, sizeof(uint32_t));

    next_slot_rx_packets_ts.reset(new dev_buf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_ALL_SYMBOLS * sizeof(uint64_t), gDev));
    cudaMemset((uint64_t*)next_slot_rx_packets_ts->addr(), 0, ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM * ORAN_ALL_SYMBOLS * sizeof(uint64_t));
    next_slot_rx_packets_count.reset(new dev_buf(ORAN_ALL_SYMBOLS * sizeof(uint32_t), gDev));
    cudaMemset((uint32_t*)next_slot_rx_packets_count->addr(), 0, ORAN_ALL_SYMBOLS * sizeof(uint32_t));
    next_slot_num_prb.reset(new dev_buf(1 * sizeof(uint32_t), gDev));
    cudaMemset((uint32_t*)next_slot_num_prb->addr(), 0, sizeof(uint32_t));
}

}