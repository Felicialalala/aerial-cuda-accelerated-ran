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

#ifndef FH_GENERATOR_CELL_HPP__
#define FH_GENERATOR_CELL_HPP__
#include "yaml_parser.hpp"
#include "gpudevice.hpp"

namespace fh_gen
{
class FhGenerator;

class Cell
{
public:
    Cell(FhGenerator* _fhgen, GpuDevice* _gDev, const PeerInfo& _peer_info);
    ~Cell();
    void allocate_buffers();
    PeerInfo    peer_info;
    FhGenerator*                       fhgen;

    //Packet timing stats for the next Uplink slot (overflow from current slot)
    std::unique_ptr<dev_buf>              next_slot_on_time_rx_packets;
    std::unique_ptr<dev_buf>              next_slot_early_rx_packets;
    std::unique_ptr<dev_buf>              next_slot_late_rx_packets;
    std::unique_ptr<dev_buf>              next_slot_rx_packets_ts;
    std::unique_ptr<dev_buf>              next_slot_rx_packets_count;
    std::unique_ptr<dev_buf>              next_slot_num_prb;
    GpuDevice *   gDev;
};

}
#endif