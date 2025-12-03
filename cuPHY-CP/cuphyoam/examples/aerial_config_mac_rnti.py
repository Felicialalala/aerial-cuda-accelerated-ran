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


# Note: define different cmd_id for different commands
CMD_ID_CONFIG_RNTI = 1

def SendGenericAsyncCmd(channel, cmd_id, param_int1=0, param_int2=0, param_str=""):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.SendGenericAsyncCmd(aerial_common_pb2.GenericAsyncRequest(cmd_id=cmd_id, param_int1=param_int1, param_int2=param_int2, param_str=param_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_config_mac_rnti.py', description="Configure testMAC PUSCH RNTI mode")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server IP address", type=str, default="localhost", required=False)
    parser.add_argument("--rnti_mode", help="Set RNTI mode: 0 - normal; 1 - negative test mode", type=int, required=True)

    args=parser.parse_args()
    server_ip = args.server_ip
    rnti_mode = args.rnti_mode

    # Set cmd_id
    cmd_id = CMD_ID_CONFIG_RNTI

    print(f"Configure testMAC PUSCH RNTI mode: {server_ip} cmd_id={cmd_id} rnti_mode={rnti_mode}")

    with grpc.insecure_channel(server_ip+":50052") as channel:
        SendGenericAsyncCmd(channel, cmd_id, rnti_mode)
