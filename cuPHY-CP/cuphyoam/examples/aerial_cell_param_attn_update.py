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

"""A Python implementation for changing a Cell's attenuation configuration"""

import logging

import sys
sys.path.append('.')

import grpc
import os
import time

import aerial_common_pb2
import aerial_common_pb2_grpc


def UpdateCellParam(channel, cell_id, attenuation_db):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.UpdateCellParams(aerial_common_pb2.CellParamUpdateRequest(cell_id=cell_id, update_cell_attenuation_cfg=True, attenuation_db=attenuation_db))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    if len(sys.argv) < 3:
       print(f"Usage: {os.path.basename(__file__)} cell_id attenuation_db")
       sys.exit(1)

    cell_id = int(sys.argv[1])
    attenuation_db = float(sys.argv[2])

    print(f"Setting Cell {cell_id} additional attenuation to {attenuation_db}")

    with grpc.insecure_channel('localhost:50051') as channel:
        UpdateCellParam(channel, cell_id, attenuation_db)
