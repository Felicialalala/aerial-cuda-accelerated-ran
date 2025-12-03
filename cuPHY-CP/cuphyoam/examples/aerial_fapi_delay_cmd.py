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

import logging

import argparse
import sys
sys.path.append('.')

import grpc
import os
import time

# Add $cuBB_SDK/build/cuPHY-CP/cuphyoam to python search directory
cuBB_SDK = os.getenv('cuBB_SDK')
if (not cuBB_SDK is None):
    sys.path.append(cuBB_SDK + "/build/cuPHY-CP/cuphyoam")

import aerial_common_pb2
import aerial_common_pb2_grpc


def SendFapiDelayCmd(channel, cell_id, slot, fapi_mask, delay_us):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.SendFapiDelayCmd(aerial_common_pb2.FapiDelayCmdRequest(cell_id=cell_id, slot=slot, fapi_mask=fapi_mask, delay_us=delay_us))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_fapi_delay_cmd.py', description="Send FAPI delay cmd via OAM")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server IP address", type=str, default="localhost", required=False)
    parser.add_argument("--cell_id", help="Cell id to send command to", type=int, default=255, required=False)
    parser.add_argument("--slot", help="Slot number to delay", type=int, default=0, required=True)
    parser.add_argument("--fapi_mask", help="Bit mask of FAPI messages to delay", type=int, default=255, required=False)
    parser.add_argument("--delay_us", help="Delay time in microseconds", type=int, default=500, required=True)


    args=parser.parse_args()

    server_ip = args.server_ip
    cell_id = args.cell_id
    slot = args.slot
    fapi_mask = args.fapi_mask
    delay_us = args.delay_us

    print(f"Send FAPI delay cmd: {server_ip} cell_id={cell_id} slot={slot} fapi_mask={fapi_mask} delay_us={delay_us}")

    with grpc.insecure_channel(server_ip+":50052") as channel:
        SendFapiDelayCmd(channel, cell_id, slot, fapi_mask, delay_us)
