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

"""A Python implementation for retrieving the current SFN from Aerial gNB"""

import logging
import multiprocessing
import os
import sys
sys.path.append('.')

import grpc
import time

import aerial_common_pb2
import aerial_common_pb2_grpc
import pdb

PRINT_RESPONSE = 0

def GetTestStream(channel,msgSizeInBytes,total_vec):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    data = bytes([k % 256 for k in range(msgSizeInBytes)])
    streamResponse = stub.GetTestStream(aerial_common_pb2.TestStreamRequest(data=data,num_bytes=msgSizeInBytes))
    response_count = 0
    total_bytes = 0
    total_msgs = 0
    for response in streamResponse:
        response_count = response_count + 1
        if (PRINT_RESPONSE):
            #pdb.set_trace()
            line = ""
            c = 0;
            for d in response.data:
                line = line + f"{d:02x} "
                c = c + 1
                if (c % 16 == 0):
                    print(f"{response_count:05d}:{(c-16):05d}-{(c-1):05d}  {line}")
                    line = ""

            if (c % 16 != 0):
                print(f"{response_count:05d}:{(c-(c%16)):05d}-{(c-1):05d}  {line}")
            print("")

        total_bytes += len(response.data)
        total_msgs += 1
        with total_vec.get_lock():
            total_vec[0] = total_bytes
            total_vec[1] = total_msgs

def task(grpcServer, msgSizeInBytes, total_vec):
    with grpc.insecure_channel(grpcServer) as channel:
        GetTestStream(channel,msgSizeInBytes,total_vec)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    if len(sys.argv) < 4:
       print(f"Usage: {os.path.basename(__file__)} grpcServer numClients msgSizeInBytes")
       sys.exit(1)

    grpcServer = sys.argv[1]
    numClients = int(sys.argv[2])
    msgSizeInBytes = int(sys.argv[3])

    print(f"Spawning {numClients} clients with msg size {msgSizeInBytes} bytes")

    pVec = []
    totalVecVec = []
    for k in range(numClients):
        v = multiprocessing.Array('i', [0,0], lock=True)
        p = multiprocessing.Process(target=task, args=(grpcServer, msgSizeInBytes, v))
        p.start()
        pVec.append(p)
        totalVecVec.append(v)

    prev_total_bytes = 0
    prev_total_msgs = 0
    time_start = time.perf_counter()
    while (1):
        cur_total_bytes = 0
        cur_total_msgs = 0
        time.sleep(5)
        for v in totalVecVec:
            cur_total_bytes = cur_total_bytes + v[0]
            cur_total_msgs = cur_total_msgs + v[1]
        time_end = time.perf_counter()
        time_delta = (time_end-time_start)
        bytes_delta = cur_total_bytes - prev_total_bytes
        rate_Mbps = bytes_delta*8 / time_delta / 1e6
        msgs_per_sec = (cur_total_msgs - prev_total_msgs) / time_delta
        print(f"Received {cur_total_msgs} messages, {cur_total_bytes} total bytes, {rate_Mbps} Mbps, {msgs_per_sec} msgs/sec")
        time_start = time_end
        prev_total_bytes = cur_total_bytes
        prev_total_msgs = cur_total_msgs

    for p in pVec:
        p.join()
