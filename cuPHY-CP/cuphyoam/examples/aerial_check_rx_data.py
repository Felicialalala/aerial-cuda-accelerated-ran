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

"""Live continuous receive and checking of rx_data.inidcation msg and associated PDU"""

import logging

import sys
sys.path.append('.')
sys.path.append('../../../build/cuPHY-CP/cuphyoam')

import grpc
import time
import binascii
import aerial_common_pb2
import aerial_common_pb2_grpc

msgtype_text_table_0x00 = ["PARAM.request", "PARAM.response", "CONFIG.request", "CONFIG.response", "START.request", "STOP.request", "STOP.indication", "ERROR.indication"]
msgtype_text_table_0x80 = ["DL_TTI.request", "UL_TTI.request", "SLOT.indication", "UL_DCI.request", "TX_Data.request", "Rx_Data.indication", "CRC.indication", "UCI.indication", "SRS.indication", "RACH.indication"]
RX_data_indication = 0x85

def bytesdump(title, bytes, columns):
    print(title)
    i = 0
    while i < len(bytes):
        byteshex = binascii.hexlify( bytes[i:(i+columns)], ' ')   # space separated
        print(" " + format(i,'04X') + " : " + str(byteshex, "utf-8"))  # utf-8 to remove the b in front
        i+=columns

# msg_buf packet format
# msg header
#   [0]   = nummsg
#   [1]   = opaque
#   [2-3] = msgtype
#   [4-7] = msg bodylen
# -------------------
# msg body
#   [8-9] = sfn
#   [10-11] = slot
#   [12-13] = num_pdu
# -------------------
# pdu_messages = msg_buf[14..end]
# PDU 1 start at ofs i=14
#   pdu[0-3] = handle
#   pdu[4-5] = rnti
#   pdu[6]   = harqid
#   pdu[7-10]= pdu_length
#   pdu[11]  = UL_CQI
#   pdu[12-13] = TA
#   pdu[14-15] = RSSI
# -------------------
# PDU 2 start at ofs i = i + pdu_length
#   pdu[0-3] = handle
#   pdu[4-5] = rnti
#   pdu[6]   = harqid
#   pdu[7-10]= pdu_length
#   pdu[11]  = UL_CQI
#   pdu[12-13] = TA
#   pdu[14-15] = RSSI
# ...


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    channel = grpc.insecure_channel('localhost:50051')
    stub = aerial_common_pb2_grpc.CommonStub(channel)
    L2response = stub.GetFAPIStream(aerial_common_pb2.FAPIStreamRequest(client_id=1,total_msgs_requested=-1))

    for response in L2response :
        # if len(response.data_buf):  # Does this msg have a data_buf payload attached
        if int.from_bytes(response.msg_buf[2:4],"little") == RX_data_indication :

            print("---------------------------------------------------------------------------------------------")
            # msg_buf is of type "bytes" and we now convert to hex text and space separated
            byteshex = binascii.hexlify(response.msg_buf, ' ') # space separated
            print("msg_buf = " + str(byteshex, "utf-8"))       # utf-8 to remove the b in front

            # uint8 number of messages included in PHY API message
            nummsg_bytes = response.msg_buf[0:1]
            nummsg_int = int.from_bytes(nummsg_bytes,"little")

            # uint8 opaque handle
            opaque_bytes = response.msg_buf[1:2]
            opaque_int = int.from_bytes(opaque_bytes,"little")

            # unit16 message type ID
            msgtype_bytes = response.msg_buf[2:4]
            msgtype_int = int.from_bytes(msgtype_bytes,"little")

            # Length of message body (bytes)
            msg_bodylen_bytes = response.msg_buf[4:8]
            msg_bodylen_int = int.from_bytes(msg_bodylen_bytes,"little")

            if (msgtype_int < 0x80):
                msgname = msgtype_text_table_0x00[msgtype_int]
            else:
                msgname = msgtype_text_table_0x80[msgtype_int - 0x80]

            print("nummsg=" + str(nummsg_int) + "\tmsg_type=" + hex(msgtype_int) + "(" + msgname + ")\topaque=" + str(opaque_int) + "\tmsg_bodylen=" + str(msg_bodylen_int))

            # print(f"Length of response data buf {len(response.data_buf)}")

            # Check that length of msgbuf bodylen + header matches actual size of received msgbuf 
            if (msg_bodylen_int + 8 == len(response.msg_buf)):
                print("OK: (8 + msg_bodylen) matches len(msgbuf)=" + str(len(response.msg_buf)))
            else :
                print("ERROR: header + len(msg_bodylen) and len(msgbuf) mismatch")
                exit

            if msgtype_int == RX_data_indication :
                # See 3.4.7 Rx_Data.indication page 97 of SCF222_02_5G_FAPI_SPY_SPI_Spec-MAR20.pdf
                sfn = int.from_bytes(response.msg_buf[8:10],"little")  # 442, 453, 505, 653,757, 807, 954, 1006, 21, 130, 234, 270, 284, 374, 431, 483
                slot = int.from_bytes(response.msg_buf[10:12],"little") # 4,5,15,14,
                num_pdu = int.from_bytes(response.msg_buf[12:14],"little") # always 1

                # ---------------------------------------------------------------------
                # Error checking validity of data Rx_Data.indication per spec sec 3.4.7
                # SFN
                # Slot
                # number of PDUs
                # ---------------------------------------------------------------------
                print(f"SFN={sfn} \t SLOT={slot} \t num_pdu={num_pdu}")

                if (sfn > 1023) :
                    sys.exit("ERROR: Invalid SFN value (out of bound)")

                if (slot > 159) :
                    sys.exit("ERROR: Invalid SLOT number (out of bound)")

                # Now check the rest of the packet in the for-each-pdu section
                pdu_messages = response.msg_buf[14:]
                sum_pdu_length = 0 # accumulator of total concatenated pdu length in msg_buf
                pdu_ofs = 0 # offset from [14:] as 0, incrementing at each pdu's pdu_length
                for pdu in range(num_pdu):
                    pdu_this = pdu_messages[pdu_ofs:]

                    handle = int.from_bytes(pdu_this[0:4],"little")
                    rnti = int.from_bytes(pdu_this[4:6],"little")
                    harqID = int(pdu_this[6])
                    pdu_length =int.from_bytes( pdu_this[7:11],"little")
                    UL_CQI =int(pdu_this[11])
                    timing_advance = int.from_bytes(pdu_this[12:14],"little")
                    RSSI = int.from_bytes(pdu_this[14:16],"little")
                    pdu_ofs += pdu_length

                    print(f"Handle={handle}\t RNTI={rnti}\t harqID={harqID}\t pdu_len={pdu_length}\t UL_CQI={UL_CQI}\t TA={timing_advance}\t RSSI={RSSI}")

                    # Error checking validity of data Rx_Data.indication per spec sec 3.4.7
                    # SFN
                    # Slot
                    # number of PDUs
                    # ---------------------------------------------------------------------
                    # For each PDU {
                    #   handle (uint_32)
                    #   rnti (uint16) 1..65535
                    #   harqID (uint8) 0..15 HARQ Process ID
                    #   pdu_length (uint16) 0..65535 (a length of 0 indicate CRC or decode error)
                    #   UL_CQI (uint8)  0..255 (representing SNR from -64dB to 63dB step 0.5dB, or 0xFF if field is invalid)
                    #   TA timing_advance (uint16) 0..63, or 0xFFFF indicating if field is invalid)
                    #   RSSI (uint16) 0-1280
                    #   PDU (variable) content of MAC PDU
                    # }
                    # ---------------------------------------------------------------------

                    # Add up all lengths of PDUs in msg_buf and be sure it matches data_buf
                    sum_pdu_length = sum_pdu_length + pdu_length

                    if (pdu_length == 0) :
                        print("PDU length of 0 meand CRC error per the spec")

                    if (timing_advance > 31) :
                        sys.exit("ERROR: TA value out of bound")

                    # Skip checking RSSI for now available
                    # if (RSSI > 1280) :
                    #    sys.exit("ERROR: RSSI value out of bound")

                    bytesdump("data_buf (mac pdu)", response.data_buf, 32)
                    # end of for loop of pdu

                if (sum_pdu_length != len(response.data_buf)) :
                    sys.exit("ERROR: Specified PDU length does not match actual PDU buffer length")
                else :
                    print("OK: pdu length matches data_buf=" + str(len(response.data_buf)))

    exit
