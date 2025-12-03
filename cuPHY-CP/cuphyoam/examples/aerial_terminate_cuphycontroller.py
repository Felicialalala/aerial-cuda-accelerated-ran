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

"""A Python implementation for terminating cuphycontroller using a gRPC message"""

import logging

import sys
sys.path.append('.')

import grpc
import time

import aerial_common_pb2
import aerial_common_pb2_grpc


def TerminateCuphycontroller(channel):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.TerminateCuphycontroller(aerial_common_pb2.GenericRequest(name=''))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    with grpc.insecure_channel('localhost:50051') as channel:
        logging.info(f"Terminating cuphycontroller...")
        try:
            TerminateCuphycontroller(channel)
        except grpc._channel._InactiveRpcError:
            logging.info(f"cuphycontroller terminated successfully!")
