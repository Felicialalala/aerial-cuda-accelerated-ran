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

"""A Python implementation for sending one-time all-zero U-plane data for Aerial gNB"""

import logging
import sys
import argparse
import grpc
import os

sys.path.append('.')

# Add $cuBB_SDK/build/cuPHY-CP/cuphyoam to python search directory
cuBB_SDK = os.getenv('cuBB_SDK')
if (not cuBB_SDK is None):
    sys.path.append(cuBB_SDK + "/build/cuPHY-CP/cuphyoam")

import aerial_common_pb2
import aerial_common_pb2_grpc

# Define error codes
class ErrorCodes:
    SUCCESS = 0
    SERVER_UNAVAILABLE = 1
    INVALID_ARGUMENTS = 2
    RPC_ERROR = 3
    UNEXPECTED_ERROR = 4

def check_server_connection(server_addr, timeout=5):
    """Check if gRPC server is available"""
    try:
        channel = grpc.insecure_channel(server_addr)
        future = grpc.channel_ready_future(channel)
        future.result(timeout=timeout)
        channel.close()
        return ErrorCodes.SUCCESS
    except grpc.FutureTimeoutError:
        logging.error(f"Server at {server_addr} is not responding (timeout after {timeout}s)")
        return ErrorCodes.SERVER_UNAVAILABLE
    except Exception as e:
        logging.error(f"Failed to connect to server at {server_addr}: {str(e)}")
        return ErrorCodes.SERVER_UNAVAILABLE

# ... copyright header and imports remain the same ...

def sendZeroUplane(channel, cell_id, use_cell_mask, cell_mask):
    try:
        stub = aerial_common_pb2_grpc.CommonStub(channel)
        
        # Format request based on mask or cell_id
        if use_cell_mask:
            logging.info(f"Using cell mask: 0x{cell_mask:x}")
            request = aerial_common_pb2.ZeroUplaneRequest(
                cell_id=-1,
                use_cell_mask=1,
                cell_mask=cell_mask,
                channel_id=args.channel_id
            )
        else:
            if not (0 <= cell_id <= 19):
                logging.error(f"Invalid cell_id {cell_id}. Must be between 0 and 19")
                return ErrorCodes.INVALID_ARGUMENTS
                
            logging.info(f"Sending zero U-plane for cell {cell_id}")
            request = aerial_common_pb2.ZeroUplaneRequest(
                cell_id=cell_id,
                use_cell_mask=0,
                cell_mask=0,
                channel_id=args.channel_id
            )

        response = stub.SendZeroUplane(request)
        
        if use_cell_mask:
            logging.info(f"Zero U-plane sent for cells with mask 0x{cell_mask:x}")
        else:
            logging.info(f"Zero U-plane sent for cell {cell_id}")
        return ErrorCodes.SUCCESS

    except grpc.RpcError as e:
        status_code = e.code()
        details = e.details()
        logging.error(f"RPC failed: {status_code} - {details}")
        
        if status_code == grpc.StatusCode.UNAVAILABLE:
            logging.error("Server is unavailable. Check if the server is running and accessible")
            return ErrorCodes.SERVER_UNAVAILABLE
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            if use_cell_mask:
                logging.error(f"Invalid cell mask: 0x{cell_mask:x}")
            else:
                logging.error(f"Invalid cell_id: {cell_id}")
            return ErrorCodes.INVALID_ARGUMENTS
        return ErrorCodes.RPC_ERROR
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ErrorCodes.UNEXPECTED_ERROR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='aerial_ul_zero_uplane.py',
        description="Send one-time all-zero U-plane data"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost")
    parser.add_argument("--port", help="OAM server port", type=int, default=50052)
    
    # Mutually exclusive group for cell_id vs cell_mask
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell_id", help="Cell id [0-19]", type=int)
    group.add_argument("--cell_mask", help="Cell bitmask to set", type=lambda x: int(x, 0))
    
    # Channel ID with default and range validation
    def channel_id_type(x):
        try:
            x = int(x)
            if not 1 <= x <= 4:
                raise argparse.ArgumentTypeError("Channel ID must be between 1 and 4")
            return x
        except ValueError:
            raise argparse.ArgumentTypeError("Channel ID must be an integer")
            
    parser.add_argument("--channel_id", 
                       help="Channel id [1-4]", 
                       type=channel_id_type,
                       default=1)

    args = parser.parse_args()

    # Validate arguments
    if args.cell_mask is not None:
        use_cell_mask = 1
        cell_mask = args.cell_mask
        cell_id = -1
    else:
        use_cell_mask = 0
        cell_mask = 0
        cell_id = args.cell_id

    try:
        server_addr = f"{args.server_ip}:{args.port}"
        logging.info(f"Checking server connection at {server_addr}")
        
        # Check server connection first
        result = check_server_connection(server_addr)
        if result != ErrorCodes.SUCCESS:
            sys.exit(result)
            
        logging.info("Server is available, proceeding with request")
        
        with grpc.insecure_channel(server_addr) as channel:
            result = sendZeroUplane(
                channel=channel,
                cell_id=cell_id,
                use_cell_mask=use_cell_mask,
                cell_mask=cell_mask
            )
            sys.exit(result)
            
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)
    except Exception as e:
        logging.error(f"Failed to execute command: {str(e)}")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)