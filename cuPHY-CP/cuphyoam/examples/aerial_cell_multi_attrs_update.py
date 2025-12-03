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

import aerial_common_pb2
import aerial_common_pb2_grpc

def isMac(string):
    preg = re.compile('^([a-fA-F0-9]{2}:){5}[a-fA-F0-9]{2}$')
    ret = preg.match(string)
    if ret is None:
        return False
    else:
        return True

def encode_mac_address(mac):
    mac = mac.replace(":", "")
    return int(mac, 16)


def encode_pcie_address(address):
    # Split the address into its components
    domain, bus, device_function = address.split(':')

    # Split the device and function components into their own values
    device, function = device_function.split('.')

    # Convert the components to integers
    domain = int(domain, 16)
    bus = int(bus, 16)
    device = int(device, 16)
    function = int(function, 16)

    # Calculate the integer value of the address
    address_int = (domain << 20) | (bus << 12) | (device << 4) | function

    return address_int


def UpdateCellParam(channel, cell_id, attrs):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    response = stub.UpdateCellParamsSyncCall(aerial_common_pb2.CellParamUpdateRequest(cell_id=cell_id, multi_attrs_cfg=True, attrs=attrs))
    status = "Ok" if not response.resp.status_code else "Failed"
    print("Response status:", status)
    if response.resp.status_code :
        print("Response error_msgs:", response.resp.error_msgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_cell_multi_attrs_update.py', description="Send Cell multi params update cmd via OAM")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost", required=False)
    parser.add_argument("--cell_id", help="Cell id to send command to. Current support range is [1~20]", type=int, default=1, required=False)
    parser.add_argument("--ru_type", help="ru_type: 1(SINGLE_SECT_MODE), 2(MULTI_SECT_MODE) or 3(OTHER_MODE) ", type=int)
    parser.add_argument("--nic", help="Choose which NIC port to use in dual FH ports case", type=str)
    parser.add_argument("--dst_mac_addr", help="dst mac addr", type=str)
    parser.add_argument("--vlan_id", help="vlan id", type=int)
    parser.add_argument("--pcp", help="pcp", type=int)
    parser.add_argument("--dl_comp_meth", help="dl_comp_meth: 0 for Fix point , or 1 for BFP", type=int)
    parser.add_argument("--dl_bit_width", help="dl_bit_width: 16 for Fix point, 9, 14, 16 for BFP", type=int)
    parser.add_argument("--ul_comp_meth", help="ul_comp_meth: 0 for Fix point, or 1 for BFP", type=int)
    parser.add_argument("--ul_bit_width", help="ul_bit_width: 16 for Fix point, 9, 14, 16 for BFP", type=int)
    parser.add_argument("--exponent_dl", help="exponent_dl", type=int)
    parser.add_argument("--exponent_ul", help="exponent_ul", type=int)
    parser.add_argument("--max_amp_ul", help="max_amp_ul", type=int)
    parser.add_argument("--pusch_prb_stride", help="pusch_prb_stride", type=int)
    parser.add_argument("--prach_prb_stride", help="prach_prb_stride", type=int)
    parser.add_argument("--section_3_time_offset", help="section_3_time_offset", type=int)
    parser.add_argument("--fh_distance_range", help="fh_distance_range: 0(for 0~30km), or 1(for 20~50km)", type=int)
    parser.add_argument("--gps_alpha", help="gps_alpha", type=int)
    parser.add_argument("--gps_beta", help="gps_beta", type=int)
    parser.add_argument("--ul_gain_calibration", help="ul_gain_calibration", type=float)
    parser.add_argument("--lower_guard_bw", help="lower_guard_bw", type=int)
    parser.add_argument("--ref_dl", help="ref_dl", type=int)

    args=parser.parse_args()

    server_ip = args.server_ip
    cell_id = args.cell_id

    if args.cell_id <= 0 or args.cell_id > 20:
         print("Invalid mplane id. Current support range is [1~20]...")
         sys.exit()

    print(f"Send Cell Prams update Cmd to Cell {cell_id}")

    attrs = {}
    if args.dst_mac_addr is not None:
        if not isMac(args.dst_mac_addr) :
             print("Ill formatted mac address, exit...")
             sys.exit()
        if args.vlan_id is None:
             print("No vlan id provided, exit...")
             sys.exit()
        if args.pcp is None:
             print("No pcp provided, exit...")
             sys.exit()

        attrs["dst_mac_addr"] = args.dst_mac_addr
        attrs["vlan_id"] = args.vlan_id
        attrs["pcp"] = args.pcp

    if args.ru_type is not None:
        attrs["ru_type"] = args.ru_type
        if args.ru_type < 1 or args.ru_type > 3:
             print("Invalid ru_type, should be either 1(SINGLE_SECT_MODE), 2(MULTI_SECT_MODE) or 3(OTHER_MODE) exit...")
             sys.exit()

    if args.dl_comp_meth is not None or args.dl_bit_width is not None:
        if args.dl_comp_meth is None or args.dl_bit_width is None:
            print("'dl_comp_meth' and 'dl_bit_width' have to updated together, one of them is missing...")
            sys.exit()
        attrs["dl_comp_meth"] = args.dl_comp_meth
        attrs["dl_bit_width"] = args.dl_bit_width
        if args.dl_comp_meth < 0 or args.dl_comp_meth > 1:
            print("Invalid dl_comp_meth, should be either 0(Fix point)) or 1(BFP) exit...")
            sys.exit()
        if args.dl_comp_meth == 0:
            if args.dl_bit_width != 16:
                print("Invalid dl_bit_width for fix point, only 16 is supported currently. Exit...")
                sys.exit()
        if args.dl_comp_meth == 1:
            if args.dl_bit_width != 9 and args.dl_bit_width != 14 and args.dl_bit_width != 16 :
                print("Invalid dl_bit_width for BFP, the supported values are: 9, 14, 16. Exit...")
                sys.exit()

    if args.ul_comp_meth is not None or args.ul_bit_width is not None:
        if args.ul_comp_meth is None or args.ul_bit_width is None:
            print("'ul_comp_meth' and 'ul_bit_width' have to updated together, one of them is missing...")
            sys.exit()
        attrs["ul_comp_meth"] = args.ul_comp_meth
        attrs["ul_bit_width"] = args.ul_bit_width
        if args.ul_comp_meth < 0 or args.ul_comp_meth > 1:
            print("Invalid ul_comp_meth, should be either 0(Fix point)), 1(BFP) exit...")
            sys.exit()
        if args.ul_comp_meth == 0:
            if args.ul_bit_width != 16:
                print("Invalid ul_bit_width for fix point, only 16 is supported currently. Exit...")
                sys.exit()
        if args.ul_comp_meth == 1:
            if args.ul_bit_width != 9 and args.ul_bit_width != 14 and args.ul_bit_width != 16 :
                print("Invalid ul_bit_width for BFP, the supported values are: 9, 14, 16. Exit...")
                sys.exit()

    if args.exponent_dl is not None:
        attrs["exponent_dl"] = args.exponent_dl
    if args.exponent_ul is not None:
        attrs["exponent_ul"] = args.exponent_ul
    if args.max_amp_ul is not None:
        attrs["max_amp_ul"] = args.max_amp_ul
    if args.pusch_prb_stride is not None:
        attrs["pusch_prb_stride"] = args.pusch_prb_stride
    if args.prach_prb_stride is not None:
        attrs["prach_prb_stride"] = args.prach_prb_stride
    if args.section_3_time_offset is not None:
        attrs["section_3_time_offset"] = args.section_3_time_offset
    if args.fh_distance_range is not None:
        attrs["fh_distance_range"] = args.fh_distance_range
        if args.fh_distance_range != 0 and args.fh_distance_range != 1:
             print("Invalid fh_distance_range, should be either 0(for 0~30km), or 1(for 20~50km), exit...")
             sys.exit()
    if args.gps_alpha is not None:
        attrs["gps_alpha"] = args.gps_alpha
    if args.gps_beta is not None:
        attrs["gps_beta"] = args.gps_beta
    if args.ul_gain_calibration is not None:
        attrs["ul_gain_calibration"] = args.ul_gain_calibration
    if args.lower_guard_bw is not None:
        attrs["lower_guard_bw"] = args.lower_guard_bw
    if args.ref_dl is not None:
        attrs["ref_dl"] = args.ref_dl
    if args.nic is not None:
        attrs["nic"] = args.nic

    if len(attrs) == 0 :
        print("No params to be updated, exit...")
        sys.exit()

    print(json.dumps(attrs, indent = 4))

    if args.dst_mac_addr is not None:
        attrs["dst_mac_addr"] = encode_mac_address(args.dst_mac_addr)

    if args.nic is not None:
        attrs["nic"] = encode_pcie_address(args.nic)

    with grpc.insecure_channel(server_ip+":50051") as channel:
        UpdateCellParam(channel, cell_id, attrs)
