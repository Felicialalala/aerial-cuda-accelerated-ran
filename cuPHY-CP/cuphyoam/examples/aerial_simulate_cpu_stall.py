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

"""A Python implementation for sending simulated cpu stall request"""

import logging

import argparse
import sys
sys.path.append('.')

import grpc
import os
import platform
import time

cuBB_SDK=os.environ.get('cuBB_SDK','/opt/nvidia/cuBB')
BUILD=f'build.{platform.machine()}'
if not os.path.exists(BUILD):
    BUILD='build'

sys.path.append(f'{cuBB_SDK}/{BUILD}/cuPHY-CP/cuphyoam')

import aerial_common_pb2
import aerial_common_pb2_grpc


def SimulateCPUStall(channel, thread_id, task_id,usleep_duration):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.SimulateCPUStall(aerial_common_pb2.SimulatedCPUStallRequest(thread_id=thread_id, task_id=task_id,usleep_duration=usleep_duration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_simulate_cpu_stall.py', description="Send Simulated CPU Stall cmd via OAM")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost", required=True)
    parser.add_argument("--thread_id", help="0: l2a msg thread, 1: l2a tick thread, 2: DL worker thread, 3: UL worker thread", type=int, required=True)
    parser.add_argument("--task_id", help="DL/UL worker thread task IDs DL <0: pdsch 1: control 2: fh_cb 3:compression 4:gpu_comm 5:buf_cleanup 6:DLBfw 7:Cplane> UL 0:pusch/pucch 1:EarlyUciInd 2:prach 3:srs 4:orderKernel 5:Aggr3 6:ULBfw 7:Cplane", type=int, default="-1",required=False)
    parser.add_argument("--usleep_duration", help="Simulated cpu stall duration", type=int, required=True)

    args=parser.parse_args()

    server_ip = args.server_ip
    thread_id = args.thread_id
    task_id = args.task_id
    usleep_duration = args.usleep_duration    

    if thread_id == 0 :
        thread_name = "L2A msg thread"
    elif thread_id == 1 :
        thread_name = "L2A tick thread"
    elif thread_id == 2 :
        thread_name = "DL worker thread"
    elif thread_id == 3 :
        thread_name = "UL worker thread"

    print(f"Send Simulated CPU Stall Cmd to " + thread_name + f", task ID: {task_id} stall duration: {usleep_duration}")

    with grpc.insecure_channel(server_ip+":50051") as channel:
        SimulateCPUStall(channel, thread_id, task_id,usleep_duration)
