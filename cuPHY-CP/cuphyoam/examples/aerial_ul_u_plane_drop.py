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
import os
import sys
import grpc

sys.path.append('.')

# Add $cuBB_SDK/build/cuPHY-CP/cuphyoam to python search directory
cuBB_SDK = os.getenv('cuBB_SDK')
if (not cuBB_SDK is None):
    sys.path.append(cuBB_SDK + "/build/cuPHY-CP/cuphyoam")

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

def SimulateULUPlaneDrop(channel, cell_id, channel_id, drop_rate, single_drop, drop_slot=None, frame_id=None, subframe_id=None, slot_id=None):
    try:
        stub = aerial_common_pb2_grpc.CommonStub(channel)
        
        # Validate input parameters
        if not (0 <= cell_id <= 19):
            logging.error(f"Invalid cell_id {cell_id}. Must be between 0 and 19")
            return ErrorCodes.INVALID_ARGUMENTS
            
        if not (1 <= channel_id <= 4):
            logging.error(f"Invalid channel_id {channel_id}. Must be between 1 and 4")
            return ErrorCodes.INVALID_ARGUMENTS
            
        if not (0 <= drop_rate <= 50):
            logging.error(f"Invalid drop_rate {drop_rate}. Must be between 0 and 50")
            return ErrorCodes.INVALID_ARGUMENTS
            
        if not (single_drop in [0, 1]):
            logging.error(f"Invalid single_drop {single_drop}. Must be 0 or 1")
            return ErrorCodes.INVALID_ARGUMENTS

        # Validate slot parameters only if single_drop is enabled
        if single_drop:
            if drop_slot is None or frame_id is None or subframe_id is None or slot_id is None:
                logging.error("When single_drop is enabled, drop_slot, frame, subframe, and slot parameters are required")
                return ErrorCodes.INVALID_ARGUMENTS

            if not (drop_slot in [0, 1]):
                logging.error(f"Invalid drop_slot {drop_slot}. Must be 0 or 1")
                return ErrorCodes.INVALID_ARGUMENTS

            if not (0 <= frame_id <= 255):
                logging.error(f"Invalid frame_id {frame_id}. Must be between 0 and 255")
                return ErrorCodes.INVALID_ARGUMENTS

            if not (0 <= subframe_id <= 9):
                logging.error(f"Invalid subframe_id {subframe_id}. Must be between 0 and 9")
                return ErrorCodes.INVALID_ARGUMENTS

            if not (0 <= slot_id <= 1):
                logging.error(f"Invalid slot_id {slot_id}. Must be 0 or 1")
                return ErrorCodes.INVALID_ARGUMENTS

        # Format request with optional parameters
        request_params = {
            'cell_id': cell_id,
            'channel_id': channel_id,
            'drop_rate': drop_rate,
            'single_drop': single_drop
        }

        # Add optional parameters only if they are provided
        if single_drop:
            request_params.update({
                'drop_slot': drop_slot,
                'frame_id': frame_id,
                'subframe_id': subframe_id,
                'slot_id': slot_id
            })

        request = aerial_common_pb2.SimulateULUPlaneDropRequest(**request_params)

        # Send request
        response = stub.SimulateULUPlaneDrop(request)
        
        # Log appropriate message based on mode
        if single_drop:
            logging.info(f"UL U-plane drop command sent for cell {cell_id}, channel {channel_id}")
            logging.info(f"Single drop mode: frame {frame_id}, subframe {subframe_id}, slot {slot_id}")
        else:
            logging.info(f"UL U-plane drop command sent for cell {cell_id}, channel {channel_id}")
            logging.info(f"Continuous drop mode with rate {drop_rate}%")
        
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
        prog='aerial_ul_u_plane_drop.py',
        description="Send UL U-plane drop command"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Required params
    parser.add_argument("--server_ip", help="OAM server ip address", type=str, default="localhost")
    parser.add_argument("--port", help="OAM server port", type=int, default=50052)
    parser.add_argument("--cell_id", help="Cell id [0-19]", type=int, required=True)
    parser.add_argument("--channel_id", help="Channel id [1-4] (1:PUSCH, 2:PRACH, 3:PUCCH, 4:SRS)", type=int, default=1)
    parser.add_argument("--drop_rate", help="Drop rate [0-50] (0 to disable)", type=int, default=0)
    parser.add_argument("--single_drop", help="Drop only one packet [0-1]", type=int, default=0)
    
    # Optional params (required when single_drop=1)
    parser.add_argument("--drop_slot", help="Drop entire slot [0-1] (required with single_drop)", type=int)
    parser.add_argument("--frame", help="Frame ID [0-255] (required with single_drop)", type=int)
    parser.add_argument("--subframe", help="Subframe ID [0-9] (required with single_drop)", type=int)
    parser.add_argument("--slot", help="Slot ID [0-1] (required with single_drop)", type=int)

    args = parser.parse_args()

    try:
        # Validate that required parameters are provided when single_drop is enabled
        if args.single_drop:
            if any(param is None for param in [args.drop_slot, args.frame, args.subframe, args.slot]):
                logging.error("When single_drop is enabled, drop_slot, frame, subframe, and slot parameters are required")
                sys.exit(ErrorCodes.INVALID_ARGUMENTS)

        server_addr = f"{args.server_ip}:{args.port}"
        logging.info(f"Checking server connection at {server_addr}")
        
        # Check server connection first
        result = check_server_connection(server_addr)
        if result != ErrorCodes.SUCCESS:
            sys.exit(result)
            
        logging.info("Server is available, proceeding with request")
        
        with grpc.insecure_channel(server_addr) as channel:
            result = SimulateULUPlaneDrop(
                channel=channel,
                cell_id=args.cell_id,
                channel_id=args.channel_id,
                drop_rate=args.drop_rate,
                single_drop=args.single_drop,
                drop_slot=args.drop_slot,
                frame_id=args.frame,
                subframe_id=args.subframe,
                slot_id=args.slot
            )
            sys.exit(result)
            
    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)
    except Exception as e:
        logging.error(f"Failed to execute command: {str(e)}")
        sys.exit(ErrorCodes.UNEXPECTED_ERROR)