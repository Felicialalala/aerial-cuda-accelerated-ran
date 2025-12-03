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

"""A Python implementation for changing a Cell's configuration"""

import logging

import argparse
import sys
sys.path.append('.')

import grpc
import os
import time
import json
import re
import yaml

import aerial_common_pb2
import aerial_common_pb2_grpc

from enum import IntEnum
from sys import maxsize

class ChannelType(IntEnum):
    NONE = maxsize  # Python equivalent of UINT32_MAX
    PDSCH_CSIRS = 0
    PDSCH = 1
    CSI_RS = 2
    PDSCH_DMRS = 3
    PBCH = 4
    SSB_PBCH_DMRS = 5
    PDCCH_DL = 6
    PDCCH_UL = 7
    PDCCH_DMRS = 8
    PUSCH = 9
    PUCCH = 10
    PRACH = 11
    SRS = 12
    BFW = 13
    CHANNEL_MAX = 14


# Mapping from YAML keys to ChannelType enum
yaml_key_to_channel_type = {
    'eAxC_id_ssb_pbch': [ChannelType.PBCH],
    'eAxC_id_pdcch': [ChannelType.PDCCH_DL, ChannelType.PDCCH_UL],
    'eAxC_id_pdsch': [ChannelType.PDSCH, ChannelType.PDSCH_CSIRS],
    'eAxC_id_csirs': [ChannelType.CSI_RS],
    'eAxC_id_pusch': [ChannelType.PUSCH],
    'eAxC_id_pucch': [ChannelType.PUCCH],
    'eAxC_id_srs': [ChannelType.SRS],
    'eAxC_id_prach': [ChannelType.PRACH]
}

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def UpdateCellParam(channel, cell_id, ch_eaxcid_map):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.UpdateCellParamsSyncCall(aerial_common_pb2.CellParamUpdateRequest(cell_id=cell_id, eaxcid_update=True, ch_eaxcid_map=ch_eaxcid_map))
    status = "Ok" if not response.resp.status_code else "Failed"
    print("Response status:", status)
    if response.resp.status_code :
        print("Response error_msgs:", response.resp.error_msgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_cell_eaxcids_update.py', description="Send Cell eAxCIds update cmd via OAM")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost", required=False)
    parser.add_argument("--cell_id", help="Cell id to send command to. Current support range is [1~16]", type=int, default=1, required=False)
    parser.add_argument("--channel_eaxcids_yaml_file", help="Cell channel eaxcids yaml file", required=True)

    args=parser.parse_args()

    server_ip = args.server_ip
    cell_id = args.cell_id

    if args.cell_id <= 0 or args.cell_id > 20:
         print("Invalid mplane id. Current support range is [1~20]...")
         sys.exit()

    print(f"Send Cell Prams update Cmd to Cell {cell_id}")

    channel_eaxcids_yaml_file = args.channel_eaxcids_yaml_file
    yaml_data = parse_yaml_file(channel_eaxcids_yaml_file)

    # Create ch_eaxcid_map message
    ch_eaxcid_map = {}

    # Populate the ch_eaxcid_map from YAML data
    print("CellParamUpdateRequest contents:")
    for key, value in yaml_data.items():
        if key in yaml_key_to_channel_type:
            channel_types = yaml_key_to_channel_type[key]
            int_vector = aerial_common_pb2.IntegerVector()
            int_vector.values.extend(value)
            for value in channel_types:
                ch_eaxcid_map[value] = int_vector
            print(f"{key}: {list(int_vector.values)}")
        else:
            print(f"Warning: No mapping found for key {key}")


    if len(ch_eaxcid_map) == 0 :
        print("No eAxCIds to be updated, exit...")
        sys.exit()

    with grpc.insecure_channel(server_ip+":50051") as channel:
        UpdateCellParam(channel, cell_id, ch_eaxcid_map)
