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

import sys
sys.path.append('.')

import grpc
import time

import aerial_common_pb2
import aerial_common_pb2_grpc


def GetCpuUtilization(channel):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.GetCpuUtilization(aerial_common_pb2.GenericRequest(name=''))
    return response.core


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    with grpc.insecure_channel('localhost:50051') as channel:
        while 1:
            cores = GetCpuUtilization(channel)
            s = ""
            for core in cores:
                utilization_percent = core.utilization_x1000/10.
                s = s + f" Core {core.core_id:3d}: {utilization_percent:5.2f}%"
            logging.info(s)

            time.sleep(1)
