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
sys.path.append('../../../build/cuPHY-CP/cuphyoam')

import grpc
import time

import aerial_common_pb2
import aerial_common_pb2_grpc


RX_data_indication = 0x85

PRINT_RESPONSE = True
global_count = 0
databuf = []
msgbuf = []


def GetFAPIStream(channel):
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    L2response = stub.GetFAPIStream(aerial_common_pb2.FAPIStreamRequest(client_id=1,total_msgs_requested=10))
    print_flag = 1
    msg_count = 0

    for response in L2response:
        print(f"Msg {msg_count} : Length of response msgbuf {len(response.msg_buf)}")
        print(f"Msg {msg_count} : Length of response databuf {len(response.data_buf)} ")
        print(f"Msg {msg_count} : Type : {hex(int.from_bytes(response.msg_buf[2:4],'little'))}")
        print("---------------------------------------------------------------------")
        

        if len(response.data_buf): #int.from_bytes(response.msg_buf[2:4],"little") == RX_data_indication
            
            print(f"Msg {msg_count} is an UL Slot, DETAILS :\n")
            time.sleep(1)
            msg_body_length = int.from_bytes(response.msg_buf[4:8],"little")
            # print(f"Length of response msg body {msg_body_length}")
            # print(f"Length of response data buf {len(response.data_buf)}")
            msgbuf.append(response.msg_buf)
            databuf.append(response.data_buf)

            sfn = int.from_bytes(response.msg_buf[8:10],"little") 
            slot = int.from_bytes(response.msg_buf[10:12],"little")
            num_pdu = int.from_bytes(response.msg_buf[12:14],"little")

            print(f"SFN: {sfn} \t SLOT: {slot} \t #PDU: {num_pdu}")
            print("---------------------------------------------------------------------")

            pdu_messages = response.msg_buf[14:]
            # print(f"pdu_msg_length: {len(pdu_messages)}")
            for pdu in range(num_pdu):
                handle =int.from_bytes( pdu_messages[0:4],"little")
                rnti = int.from_bytes(pdu_messages[4:6],"little")
                harqID = int( pdu_messages[6])
                pdu_length =int.from_bytes( pdu_messages[7:11],"little")
                UL_CQI =int(pdu_messages[11])
                timing_advance = int.from_bytes(pdu_messages[12:14],"little")
                RSSI = int.from_bytes(pdu_messages[14:16],"little")

                if PRINT_RESPONSE:
                    
                    print(f"Handle: \t{handle}")
                    print(f"RNTI: \t\t{rnti}")
                    print(f"harqID: \t{harqID}")
                    print(f"pdu_length: \t{pdu_length}")
                    print(f"UL_CQI: \t{UL_CQI}")
                    print(f"timing_advance: {timing_advance}")
                    print(f"RSSI: \t\t{RSSI}")
                    print("---------------------------------------------------------------------")
                    
                    # print("payload")
                    # i = 1
                    # if print_flag:
                    #     while 14*i < len(response.data_buf):
                    #         print([int(byte) for byte in response.data_buf[14*(i-1):14*i]])
                    #         i+=1
                    #     print_flag= 0
        msg_count += 1

    return msg_count


if __name__ == '__main__':
    response_count = 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    with grpc.insecure_channel('localhost:50051') as channel:
        msgs_count = GetFAPIStream(channel)
        print(f"Received {msgs_count} messages")
        logging.info(f"Received {msgs_count} messages")
