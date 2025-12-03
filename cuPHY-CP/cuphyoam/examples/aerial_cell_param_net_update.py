# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Python implementation for changing a Cell's FH configuration"""

import logging

import sys
sys.path.append('.')

import grpc
import os
import time

import aerial_common_pb2
import aerial_common_pb2_grpc


def UpdateCellParam(channel, cell_id, dst_mac_addr, vlan_tci):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.UpdateCellParams(aerial_common_pb2.CellParamUpdateRequest(cell_id=cell_id, update_network_cfg=True, dst_mac_addr=dst_mac_addr, vlan_tci=vlan_tci))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    if len(sys.argv) < 4:
       print(f"Usage: {os.path.basename(__file__)} cell_id dst_mac_addr vlan_tci")
       sys.exit(1)

    cell_id = int(sys.argv[1])
    dst_mac_addr = sys.argv[2]
    vlan_tci = int(sys.argv[3], 16)

    print(f"Setting Cell {cell_id} destination MAC address to {dst_mac_addr} and VLAN TCI to {vlan_tci}")

    with grpc.insecure_channel('localhost:50051') as channel:
        UpdateCellParam(channel, cell_id, dst_mac_addr, vlan_tci)
