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

"""A Python implementation for sending L2 Cell ctrl(config/start/stop) requests"""

import logging

import argparse
import sys
import grpc
import os
import time

sys.path.append('.')

# Add $cuBB_SDK/build/cuPHY-CP/cuphyoam to python search directory
cuBB_SDK = os.getenv('cuBB_SDK')
if (not cuBB_SDK is None):
    sys.path.append(cuBB_SDK + "/build/cuPHY-CP/cuphyoam")

import aerial_common_pb2
import aerial_common_pb2_grpc


def SendCellCtrlCmd(channel, cell_id, cell_ctrl_cmd, target_cell_id):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.SendCellCtrlCmd(aerial_common_pb2.CellCtrlCmdRequest(cell_id=cell_id, cell_ctrl_cmd=cell_ctrl_cmd, target_cell_id=target_cell_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_cell_ctrl_cmd.py', description="Send Cell ctrl(config/start/stop) cmd via OAM")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost", required=False)
    parser.add_argument("--port", help="OAM server port", type=int, default=50052, required=False)
    parser.add_argument("--cell_id", help="Cell id to send command to. Support range: [0~19]", type=int, required=True)
    parser.add_argument("--cmd", help="0: stop, 1: start, 2: Re-config, 3: Init config", type=int, required=True)
    parser.add_argument("--target_cell_id", help="RU id to re-attach to(optional, only valid with config command for re-attaching test)", type=int, default=-1)


    args=parser.parse_args()

    if args.cell_id < 0 or args.cell_id >= 20:
         print("Invalid cell_id. Current support range is [0~19]...")
         sys.exit()

    if args.cmd < 0 or args.cmd > 3:
         print("Invalid cmd. Current support value is 0: stop, 1: start, 2: Re-config, 3: Init config")
         sys.exit()

    if args.target_cell_id < -1 or args.target_cell_id >= 20:
         print("Invalid target_cell_id id. Current support range is [-1~19]...")
         sys.exit()

    server_ip = args.server_ip
    port = args.port
    cell_id = args.cell_id
    cell_ctrl_cmd = args.cmd
    target_cell_id = args.target_cell_id

    server_addr = server_ip + ":" + str(port)
    print(f"Send Cell Ctrl Cmd {cell_ctrl_cmd} to Cell {cell_id}, L2 ip: {server_addr}, optional target_cell_id: {target_cell_id}")

    with grpc.insecure_channel(server_addr) as channel:
    #with grpc.insecure_channel('localhost:50052') as channel:
    #with grpc.insecure_channel('10.32.204.69:50052') as channel:
        SendCellCtrlCmd(channel, cell_id, cell_ctrl_cmd, target_cell_id)
