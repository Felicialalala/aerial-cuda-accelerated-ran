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
import grpc
import aerial_common_pb2
import aerial_common_pb2_grpc

class ErrorCodes:
    SUCCESS = 0
    SERVER_UNAVAILABLE = 1
    INVALID_ARGUMENTS = 2
    RPC_ERROR = 3
    UNEXPECTED_ERROR = 4

def check_server_connection(server_addr):
    try:
        with grpc.insecure_channel(server_addr) as channel:
            stub = aerial_common_pb2_grpc.CommonStub(channel)
            # Simple ping with 1 second timeout
            grpc.channel_ready_future(channel).result(timeout=1)
            return ErrorCodes.SUCCESS
    except grpc.FutureTimeoutError:
        logging.error(f"Server at {server_addr} is not available")
        return ErrorCodes.SERVER_UNAVAILABLE
    except Exception as e:
        logging.error(f"Error checking server connection: {str(e)}")
        return ErrorCodes.UNEXPECTED_ERROR

def FlushUlPcap(channel, cell_id):
    try:
        stub = aerial_common_pb2_grpc.CommonStub(channel)
        
        # Validate input parameters
        if not (0 <= cell_id <= 19):
            logging.error(f"Invalid cell_id {cell_id}. Must be between 0 and 19")
            return ErrorCodes.INVALID_ARGUMENTS

        # Format request
        request = aerial_common_pb2.FlushUlPcapRequest(
            cell_id=cell_id
        )

        # Send request
        response = stub.FlushUlPcap(request)
        
        logging.info(f"UL PCAP flush command sent for cell {cell_id}")
        return ErrorCodes.SUCCESS

    except grpc.RpcError as e:
        status_code = e.code()
        details = e.details()
        logging.error(f"RPC failed: {status_code} - {details}")
        
        if status_code == grpc.StatusCode.UNAVAILABLE:
            logging.error("Server is unavailable. Check if the server is running and accessible")
            return ErrorCodes.SERVER_UNAVAILABLE
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            logging.error(f"Invalid arguments provided")
            return ErrorCodes.INVALID_ARGUMENTS
        return ErrorCodes.RPC_ERROR
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ErrorCodes.UNEXPECTED_ERROR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='aerial_ul_pcap_flush.py',
        description="Send UL PCAP flush command"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost")
    parser.add_argument("--port", help="OAM server port", type=int, default=50051)
    parser.add_argument("--cell_id", help="Cell id [0-19]", type=int, required=True)

    args = parser.parse_args()

    try:
        server_addr = f"{args.server_ip}:{args.port}"
        logging.info(f"Checking server connection at {server_addr}")
        
        # Check server connection first
        result = check_server_connection(server_addr)
        if result != ErrorCodes.SUCCESS:
            sys.exit(result)
            
        logging.info("Server is available, proceeding with request")
        
        with grpc.insecure_channel(server_addr) as channel:
            result = FlushUlPcap(
                channel=channel,
                cell_id=args.cell_id
            )
            sys.exit(result)
            
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)
    except Exception as e:
        logging.error(f"Failed to execute command: {str(e)}")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)