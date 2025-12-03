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

"""A Python implementation for dumping the next PUSCH CRC error from Aerial gNB"""

import logging

import sys

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


def ulPcapCaptureNextCrc(channel, cell_id, cmd, use_cell_mask, cell_mask):
    try:
        stub = aerial_common_pb2_grpc.CommonStub(channel)
        response = stub.SendCellUlPcapCmd(
            aerial_common_pb2.PcapRequest(
                cell_id=cell_id, 
                cmd=cmd, 
                use_cell_mask=use_cell_mask, 
                cell_mask=cell_mask
            )
        )
        logging.info(f"PCAP capture {'enabled' if cmd == 1 else 'disabled'} successfully")
        return 0
    except grpc.RpcError as e:
        status_code = e.code()
        details = e.details()
        logging.error(f"RPC failed: {status_code} - {details}")
        if status_code == grpc.StatusCode.UNAVAILABLE:
            logging.error("Server is unavailable. Check if the server is running and accessible")
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            logging.error("Invalid arguments provided. Check cell_id or command values")
        return -1
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='aerial_ul_pcap_capture_next_crc.py', description="Send trigger to arm pcap capture mechanism for next CRC error")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost")
    parser.add_argument("--port", help="OAM server port", type=int, default=50051, required=False)
    parser.add_argument("--cell_id", help="Cell id to send command to. Support range: [0~19]", type=int, required=False)
    parser.add_argument("--cmd", help="0: disable, 1: enable", type=int, required=False)
    parser.add_argument("--cell_mask", help="Cell bitmask to set", type=int, required=False)

    args=parser.parse_args()

    server_ip = args.server_ip
    port = args.port
    server_addr = server_ip + ":" + str(port)

    # Validate arguments
    if args.cell_mask is not None:
        use_cell_mask = 1
        cell_mask = args.cell_mask
        cell_id = -1
        cmd = -1
    else:
        if args.cell_id is None or args.cmd is None:
            parser.error("Both --cell_id and --cmd are required when not using --cell_mask")
        use_cell_mask = 0
        cell_mask = 0
        cell_id = args.cell_id
        cmd = args.cmd


    try:
        server_addr = f"{args.server_ip}:{args.port}"
        logging.info(f"Connecting to server at {server_addr}")
        
        with grpc.insecure_channel(server_addr) as channel:
            result = ulPcapCaptureNextCrc(
                channel=channel,
                cell_id=cell_id,
                cmd=cmd,
                use_cell_mask=use_cell_mask,
                cell_mask=cell_mask
            )
            sys.exit(result)
            
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to execute command: {str(e)}")
        sys.exit(1)
