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

import json
import argparse
import yaml
import pdb
import os
import sys

NUM_SYMBOLS=14
NUM_PRBS = 273
NUM_UL_ANT = 4
NUM_DL_ANT = 4
COMPRESSION_BITS = 9

MTU = 1514
# PCIe address in the same index are physically connected
DU_NIC_ADDRS = ["0000:cc:00.0", "0000:cc:00.1"]
RU_NIC_ADDRS = ["0000:b5:00.0", "0000:b5:00.1"]

# Which port to use per cell (index for the above PCIe addr lists)
CELL_PORT_IDX = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Cell IQ width
CELL_IQ_WIDTH = [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]

CELL_VLAN = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
CELL_PCP = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

# DU Core list
DU_UL_CORE_LIST = [5]
DU_DL_CORE_LIST = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# RU Core list
RU_UL_CORE_LIST = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]
RU_DL_CORE_LIST = [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43]

RU_TYPE=""
DU_TYPE=""

R750_DU_UL_CORE_LIST = [5]
R750_DU_DL_CORE_LIST = [7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45]

SMC_DU_UL_CORE_LIST = [5]
SMC_DU_DL_CORE_LIST = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]


R750_RU_UL_CORE_LIST = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]
R750_RU_DL_CORE_LIST = [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43]

SMC_RU_UL_CORE_LIST = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
SMC_RU_DL_CORE_LIST = [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]

TEST_SLOTS = 600000

ENABLE_DLC=1
ENABLE_DLU=1
ENABLE_ULC=1
ENABLE_ULU=1

SFN_SLOT_SYNC_DU_IP=""
SFN_SLOT_SYNC_RU_IP=""
OAM_PORT_ID=50051

ULC_ONTIME_PASS_PERCENTAGE=99.99
DLC_ONTIME_PASS_PERCENTAGE=99.99
ULU_ONTIME_PASS_PERCENTAGE=99.99
DLU_ONTIME_PASS_PERCENTAGE=99.99

ULU_TX_TIME_ADVANCE_NS=280000
ULU_ENQ_TIME_ADVANCE_NS=1500000
DLU_ENQ_TIME_ADVANCE_NS=1000000
DLC_ENQ_TIME_ADVANCE_NS=1000000
ONTIME_WINDOW_NS=51000

def create_tfgrid(numCells):
    g = []
    for cellIdx in range(numCells):
        g_curCell = []
        for symIdx in range(NUM_SYMBOLS):
            g_curSym = []
            for prbIdx in range(273):
                g_curSym.append(("","",0,0,0))
            g_curCell.append(g_curSym)
        g.append(g_curCell)

    return g


def update_tfgrid(slotType, g, cellIdx, startSym, numSym, startPrb, numPrb, channel, alloc):

    for symIdx in range(startSym, startSym + numSym):
        for prbIdx in range(startPrb, startPrb + numPrb):
            if slotType == "UL":
                ant = NUM_UL_ANT
            else:
                ant = NUM_DL_ANT

            # SSB and CSIRS only sections will have 1 antenna
            if "ssb" in channel:
                ant = 1
            if "csirs" in channel:
                ant = 1
            if "csirs" not in channel:
                g[cellIdx][symIdx][prbIdx] = (channel, alloc, startSym, numSym, ant)
            elif g[cellIdx][symIdx][prbIdx][0] == "":
                g[cellIdx][symIdx][prbIdx] = (channel, alloc, startSym, numSym, ant)
    return g


def tfgrid_from_file(filename, NUM_CELLS):
    ifile = open(filename, "r")
    data = json.load(ifile)
    numSlots = 0
    g = []
    for slotIdx, slotName in enumerate(data["sequence"]):
        numSlots = numSlots + 1

        slot = data["types"][slotName]

        g_curSlot = create_tfgrid(NUM_CELLS)

        slotType = 'UL'
        if "dl" in slotName:
            slotType = 'DL'

        skip_this_slot = False
        # if slotIdx != 0 and slotIdx != 5:
        # if slotIdx != 0:
        #     skip_this_slot = True

        if skip_this_slot == False:
            # print(f"[{slotIdx}] slotName is {slotName}")
            for channel in slot:
                if channel == "//comment":
                    continue
                for cellIdx in range(NUM_CELLS):

                    if (channel == "csirs_flex_0") and ((cellIdx % 2) == 1):
                        continue
                    if (channel == "csirs_flex_1") and ((cellIdx % 2) == 0):
                        continue

                    for allocIdx, tf_alloc in enumerate(slot[channel]):
                        startSym = tf_alloc[0]
                        numSym = tf_alloc[1] - tf_alloc[0]
                        startPrb = tf_alloc[2]
                        numPrb = tf_alloc[3] - tf_alloc[2]

                        g_curSlot = update_tfgrid(
                            slotType,
                            g_curSlot,
                            cellIdx,
                            startSym,
                            numSym,
                            startPrb,
                            numPrb,
                            channel,
                            channel + "_" + str(allocIdx),
                        )

        g.append((slotType,g_curSlot))

    return numSlots, g


def tf_grid_to_prb_list(g):

    # Construct list
    prb_list = []
    NUM_CELLS = len(g[0][1])
    for slotIdx, g_curSlotAndType in enumerate(g):
        g_curType = g_curSlotAndType[0]
        g_curSlot = g_curSlotAndType[1]
        prb_list_curSlot = []
        for cellIdx in range(NUM_CELLS):
            prb_list_curCell = []

            for symIdx in range(NUM_SYMBOLS):
                # print(f"Cell {cellIdx} symIdx {symIdx}")

                prb_list_curSym = []
                prb_list_curSym_channel = []

                # Find start of allocation, find end of allocation, compute numPrb
                alloc = ""
                channel = ""
                startPrb = 0

                for curPrb in range(NUM_PRBS):
                    newChannel = g_curSlot[cellIdx][symIdx][curPrb][0]
                    newAlloc = g_curSlot[cellIdx][symIdx][curPrb][1]
                    newStartSym = g_curSlot[cellIdx][symIdx][curPrb][2]
                    newNumSym = g_curSlot[cellIdx][symIdx][curPrb][3]
                    newNumAnt = g_curSlot[cellIdx][symIdx][curPrb][4]
                    # if slotIdx == 0 and symIdx == 1:
                    #     print(
                    #         f"   Found channel {newChannel} alloc {newAlloc} curPrb {curPrb}"
                    #     )
                    if channel == "" or alloc == "":
                        startPrb = curPrb
                    if channel != "" and alloc != "":
                        if newChannel != channel:
                            numPrb = curPrb - startPrb
                            prb_list_curSym_channel.append((startPrb, numPrb, startSym, numSym, numAnt))
                            # print(
                            #     f" Slot {slotIdx} sym {symIdx} Found channel {channel} alloc {alloc} startPrb {startPrb} numPrb {numPrb}"
                            # )
                            prb_list_curSym.append(prb_list_curSym_channel)
                            prb_list_curSym_channel = []
                            # (channel, alloc, startSym, numSym, numAnt) = g_curSlot[cellIdx][symIdx][curPrb]
                            startPrb = curPrb
                        elif newAlloc != alloc:
                            numPrb = curPrb - startPrb
                            # print(
                            #     f" Slot {slotIdx} sym {symIdx} Found channel {channel} alloc {alloc} startPrb {startPrb} numPrb {numPrb}"
                            # )
                            prb_list_curSym_channel.append((startPrb, numPrb, startSym, numSym, numAnt))
                            # (channel, alloc, startSym, numSym, numAnt) = g_curSlot[cellIdx][symIdx][curPrb]
                            startPrb = curPrb

                    alloc = newAlloc
                    channel = newChannel
                    startSym = newStartSym
                    numSym = newNumSym
                    numAnt = newNumAnt
                # Final allocation
                if alloc != "":
                    # end of allocation, add to list
                    numPrb = NUM_PRBS - startPrb
                    # print(
                    #     f"   Found channel {channel} alloc {alloc} startPrb {startPrb} numPrb {numPrb} startSym {startSym} numSym {numSym}"
                    # )
                    prb_list_curSym_channel.append((startPrb, numPrb, startSym, numSym, numAnt))
                    prb_list_curSym.append(prb_list_curSym_channel)

                prb_list_curCell.append(prb_list_curSym)

            prb_list_curSlot.append(prb_list_curCell)

        prb_list.append(prb_list_curSlot)

    # print(f"prb_list is {prb_list}")
    return prb_list


class FHGen:
    def __init__(self, NUM_CELLS, NUM_SLOTS, SYSARGS):
        self.nic_addrs = DU_NIC_ADDRS
        self.ru_nic_addrs = RU_NIC_ADDRS
        self.mtu = MTU
        self.dst_mac_addr_base = "20:04:9B:9E:27:"
        self.src_mac_addr_base = "20:04:9B:9E:20:"
        self.NUM_CELLS = NUM_CELLS
        self.NUM_SLOTS = NUM_SLOTS

        self.d = {
            "command_line_args": str(SYSARGS),
            "dpdk_thread": 0,
            "dpdk_verbose_logs": 0,
            "accu_tx_sched_res_ns": 500,
            "worker_thread_sched_fifo_prio": 95,
            "slot_duration_ns": 500000,
            "slot_count": self.NUM_SLOTS,
            "test_slots": TEST_SLOTS,
            "enable_ulu": ENABLE_ULU,
            "enable_dlu": ENABLE_DLU,
            "enable_ulc": ENABLE_ULC,
            "enable_dlc": ENABLE_DLC,
            "iq_data_buffers": [{"id": 0, "buffer_size": 4194304, "cuda_device_id": 0}],
            "du_cpus": {"ul": DU_UL_CORE_LIST, "dl": DU_DL_CORE_LIST},
            "ru_cpus": {"ul": RU_UL_CORE_LIST, "dl": RU_DL_CORE_LIST},
            "ulu_tx_time_advance_ns": ULU_TX_TIME_ADVANCE_NS,
            "ulu_enq_time_advance_ns": ULU_ENQ_TIME_ADVANCE_NS,
            "dlu_enq_time_advance_ns": DLU_ENQ_TIME_ADVANCE_NS,
            "dlc_enq_time_advance_ns": DLC_ENQ_TIME_ADVANCE_NS,
            "dlc_ontime_pass_percentage": DLC_ONTIME_PASS_PERCENTAGE,
            "dlu_ontime_pass_percentage": DLU_ONTIME_PASS_PERCENTAGE,
            "ulc_ontime_pass_percentage": ULC_ONTIME_PASS_PERCENTAGE,
            "ulu_ontime_pass_percentage": ULU_ONTIME_PASS_PERCENTAGE,
        }

        nics = []
        ru_nics = []
        for nic_addr in self.nic_addrs:
            nic = {
                    "nic": nic_addr,
                    "mtu": self.mtu,
                    "cpu_mbufs": 196608,
                    "uplane_tx_handles": 64,
                    "txq_count": NUM_CELLS,
                    "txq_size": 8192,
                    "rxq_count": NUM_CELLS,
                    "rxq_size": 16384,
                    "cuda_device_id": 0,
                }
            if ENABLE_DLC == 1 or ENABLE_ULC == 1:
                nic["txq_count"] = NUM_CELLS + 2 * NUM_CELLS
            nics.append(nic)
        for nic_addr in self.ru_nic_addrs:
            nic = {
                    "nic": nic_addr,
                    "mtu": self.mtu,
                    "cpu_mbufs": 262144,
                    "uplane_tx_handles": 64,
                    "txq_count": NUM_SYMBOLS * 2,
                    "txq_size": 4096,
                    "rxq_count": NUM_CELLS,
                    "rxq_size": 2048,
                    "cuda_device_id": -1,
                }
            ru_nics.append(nic)

        self.d["nics"] = nics
        self.d["ru_nics"] = ru_nics

        sfn_slot_sync = {
            "enable": 0
        }

        if SFN_SLOT_SYNC_DU_IP != "" and SFN_SLOT_SYNC_RU_IP != "":
            sfn_slot_sync["enable"] = 1
            sfn_slot_sync["ru_server_addr"] = SFN_SLOT_SYNC_RU_IP
            sfn_slot_sync["du_server_addr"] = SFN_SLOT_SYNC_DU_IP

        self.d["sfn_slot_sync"] = sfn_slot_sync

    def config(self, prb_list, g):
        cell_list = []
        for cellIdx in range(self.NUM_CELLS):
            cellEntry = {
                "cell_id": cellIdx,
                "src_mac_addr": self.src_mac_addr_base + f"{cellIdx:02x}",
                "dst_mac_addr": self.dst_mac_addr_base + f"{cellIdx:02x}",
                "txq_count_uplane": 1,
                "vlan": CELL_VLAN[cellIdx],
                "pcp": CELL_PCP[cellIdx],
                "ud_iq_width": CELL_IQ_WIDTH[cellIdx],
                "ud_comp_meth": 1,
                "nic": self.nic_addrs[CELL_PORT_IDX[cellIdx]],
                "ru_nic": self.ru_nic_addrs[CELL_PORT_IDX[cellIdx]],
                "tcp_adv_dl_ns": 125000,
                "t1a_max_cp_ul_ns": 336000,
                "t1a_max_up_ns": 345000,
                "ta4_min_ns": 50000,
                "ta4_max_ns": 331000,
                "window_end_ns": ONTIME_WINDOW_NS,
            }
            cell_list.append(cellEntry)

        self.d["cells"] = cell_list

        flow_list = []
        for slotIdx in range(self.NUM_SLOTS):
            g_curType = g[slotIdx][0]
            num_ant = NUM_DL_ANT
            if g_curType == 'UL':
                num_ant = NUM_UL_ANT
            for symIdx in range(NUM_SYMBOLS):
                for cellIdx in range(self.NUM_CELLS):
                    for antIdx in range(num_ant):
                        cplane_list = []
                        for channel in prb_list[slotIdx][cellIdx][symIdx]:
                            section_list = []
                            for channel_section in channel:
                                if channel_section[4] <= antIdx:
                                    # print(f"Skipping slot {slotIdx} sym {symIdx} ant {antIdx} startPrb {channel_section[0]} numPrb {channel_section[1]}")
                                    continue
                                section = {
                                    "start_prb": channel_section[0],
                                    "num_prb": channel_section[1],
                                    "start_sym": channel_section[2],
                                    "num_sym": channel_section[3]
                                }
                                section_list.append(section)
                            cplane = {
                                    "slot_id": slotIdx,
                                    "symbol_id": symIdx,
                                    "section_count": len(section_list), # TODO this should be based on the number of prb ranges
                                    "data_direction": g_curType,
                                    "sections": section_list
                                }
                            if len(section_list) > 0:
                                cplane_list.append(cplane)

                        uplane_list = []
                        for channel in prb_list[slotIdx][cellIdx][symIdx]:
                            section_list = []
                            for channel_section in channel:
                                if channel_section[4] <= antIdx:
                                    continue
                                section = {
                                    "start_prb": channel_section[0],
                                    "num_prb": channel_section[1]
                                }
                                section_list.append(section)
                            uplane_entry = {
                                "slot_id": slotIdx,
                                "symbol_id": symIdx,
                                "iq_data_buffer": 0,
                                "sections": section_list
                            }
                            if len(section_list) > 0:
                                uplane_list.append(uplane_entry)

                        flowEntry = {
                            "eAxC": antIdx, #(antIdx * NUM_SYMBOLS * self.NUM_SLOTS) + (slotIdx * NUM_SYMBOLS) + symIdx,
                            "cell_id": cellIdx,
                            "cplane_tx": cplane_list,
                        }
                        if g_curType == 'DL':
                            flowEntry['uplane_tx'] = uplane_list
                        elif g_curType == 'UL':
                            flowEntry['uplane_rx'] = uplane_list
                        else:
                            raise TypeError(f"Unknown slot type {g_curType}")

                        if len(cplane_list) > 0 or len(uplane_list) > 0:
                            flow_list.append(flowEntry)

        self.d["flows"] = flow_list

    def write(self, filename):
        print(f"Writing filename {filename}")
        with open(filename, "w") as f:
            yaml.dump(self.d, f, sort_keys=False, width=float("inf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert slot pattern in JSON format to fhgen yaml configuration",formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", type=int, dest="num_cells", help="Number of Cells", default = 2)
    parser.add_argument("-i", type=str, dest="pattern_filename", help="Slot pattern filename (in JSON format)", required=True)
    parser.add_argument("-o", type=str, dest="out_filename", help="Output fhgen yaml filename, default with no params: {input pattern_filename}_{num_cells}C.yaml, default with params (if specified): {input pattern_filename}_{num_cells}C_testslots_{test_slots}.yaml")
    parser.add_argument("--du_nic_addrs", type=str, dest="du_nic_addrs", help="DU NIC addresses, comma separated list, default \"0000:cc:00.1,0000:cc:00.0\"")
    parser.add_argument("--ru_nic_addrs", type=str, dest="ru_nic_addrs", help="RU NIC addresses, comma separated list, default = \"0000:b5:00.0,0000:b5:00.1\"")
    parser.add_argument("--iq_width", type=str, dest="iq_width", help="Cell IQ width, comma separated list that will repeat for total cell count, i.e. \"9,14\" for 5C test would result in 9,14,9,14,9")
    parser.add_argument("--cell_ports", type=str, dest="cell_ports", help="Cell port index to use, comma separated list that will repeat for total cell count, i.e. \"0,1\" for 5C test would result in 0,1,0,1,0")
    parser.add_argument("--vlan", type=str, dest="vlan", help="Cell VLAN ID to use, comma separated list that will repeat for total cell count, i.e. \"0,1\" for 5C test would result in 0,1,0,1,0")
    parser.add_argument("--pcp", type=str, dest="pcp", help="Cell PCP value to use, comma separated list that will repeat for total cell count, i.e. \"0,1\" for 5C test would result in 0,1,0,1,0")
    parser.add_argument("--du_type", type=str, dest="du_type", help="DU machine type (R750, SMC) with default core settings that can be overridden by the other parameters\n R750: UL: 5 DL: 7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45\n SMC: UL: 5 DL: 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25")
    parser.add_argument("--ru_type", type=str, dest="ru_type", help="RU machine type (R750, SMC) with default core settings that can be overridden by the other parameters\n R750: UL: 4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42 DL: 5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43\n SMC: UL: 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 DL: 24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43")
    parser.add_argument("--du_ul_cores", type=str, dest="du_ul_cores", help="Custom DU UL core list, default based on SMC GH = 5")
    parser.add_argument("--du_dl_cores", type=str, dest="du_dl_cores", help="Custom DU DL core list, default based on SMC GH = 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25")
    parser.add_argument("--ru_ul_cores", type=str, dest="ru_ul_cores", help="Custom RU UL core list, default based on R750 = 4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42")
    parser.add_argument("--ru_dl_cores", type=str, dest="ru_dl_cores", help="Custom RU DL core list, default based on R750 = 5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43")
    parser.add_argument("--ul_ant", type=int, dest="ul_antennas", help="Number UL Antenna streams, default = 4")
    parser.add_argument("--dl_ant", type=int, dest="dl_antennas", help="Number DL Antenna streams, default = 4")
    parser.add_argument("--enable_ulu", type=int, dest="enable_ulu", help="Enable UL U Plane TX and RX kernel, default = 1")
    parser.add_argument("--enable_dlu", type=int, dest="enable_dlu", help="Enable DL U Plane TX, default = 1")
    parser.add_argument("--enable_ulc", type=int, dest="enable_ulc", help="Enable UL C Plane TX, default = 1")
    parser.add_argument("--enable_dlc", type=int, dest="enable_dlc", help="Enable DL C Plane TX, default = 1")
    parser.add_argument("--test_slots", type=int, dest="test_slots", help="Number test slots to run, default = 600000")
    parser.add_argument("--sfn_slot_sync_ru", type=str, dest="sfn_slot_sync_ru", help="RU system IP address")
    parser.add_argument("--sfn_slot_sync_du", type=str, dest="sfn_slot_sync_du", help="DU system IP address")
    parser.add_argument("--oam_port_id", type=str, dest="oam_port_id", help="OAM port ID, default = 50051")
    parser.add_argument("--dlc_pass_pct", type=float, dest="dlc_ontime_pass_percentage", help="DLC on-time pass percentage criteria", default = 99.99)
    parser.add_argument("--dlu_pass_pct", type=float, dest="dlu_ontime_pass_percentage", help="DLU on-time pass percentage criteria", default = 99.99)
    parser.add_argument("--ulc_pass_pct", type=float, dest="ulc_ontime_pass_percentage", help="ULC on-time pass percentage criteria", default = 99.99)
    parser.add_argument("--ulu_pass_pct", type=float, dest="ulu_ontime_pass_percentage", help="ULU on-time pass percentage criteria", default = 99.99)
    parser.add_argument("--ulu_tx_adv_ns", type=int, dest="ulu_tx_adv_ns", help="ULU TX Time advance ns, default = 280000")
    parser.add_argument("--ulu_enq_adv_ns", type=int, dest="ulu_enq_adv_ns", help="ULU SW Enqueue time advance ns, default = 1500000")
    parser.add_argument("--dlc_enq_adv_ns", type=int, dest="dlc_enq_adv_ns", help="DLC SW Enqueue time advance ns, default = 1000000")
    parser.add_argument("--dlu_enq_adv_ns", type=int, dest="dlu_enq_adv_ns", help="DLU SW Enqueue time advance ns, default = 1000000")
    parser.add_argument("--ontime_window_ns", type=int, dest="ontime_window_ns", help="Ontime window ns, default = 51000")
    parser.add_argument("--mtu", type=int, dest="mtu", help="MTU to set the NIC, default 1514")

    args = parser.parse_args()

    if args.du_type is not None:
        if args.du_type == "R750":
            print(f"DU type: {args.du_type}")
            DU_DL_CORE_LIST = R750_DU_DL_CORE_LIST
            DU_UL_CORE_LIST = R750_DU_UL_CORE_LIST
        elif args.du_type == "SMC":
            print(f"DU type: {args.du_type}")
            DU_DL_CORE_LIST = SMC_DU_DL_CORE_LIST
            DU_UL_CORE_LIST = SMC_DU_UL_CORE_LIST
        else:
            print(f"DU type not recognized: {args.du_type}")

    if args.ru_type is not None:
        if args.ru_type == "R750":
            print(f"RU type: {args.ru_type}")
            RU_DL_CORE_LIST = R750_RU_DL_CORE_LIST
            RU_UL_CORE_LIST = R750_RU_UL_CORE_LIST
        elif args.ru_type == "SMC":
            print(f"RU type: {args.ru_type}")
            RU_DL_CORE_LIST = SMC_RU_DL_CORE_LIST
            RU_UL_CORE_LIST = SMC_RU_UL_CORE_LIST
        else:
            print(f"RU type not recognized: {args.ru_type}")

    output_filename = ""
    if(args.out_filename == None):
        filename, file_ext = os.path.splitext(args.pattern_filename)
        output_filename = filename + "_" + str(args.num_cells) + "C"

    if args.iq_width is not None:
        iq_width = [int(item) for item in args.iq_width.split(',')]
        q, r = divmod(args.num_cells, len(iq_width))
        CELL_IQ_WIDTH = q * iq_width + iq_width[:r]
    else:
        iq_width = CELL_IQ_WIDTH
        q, r = divmod(args.num_cells, len(iq_width))
        CELL_IQ_WIDTH = q * iq_width + iq_width[:r]
    print(f"CELL IQ WIDTH {CELL_IQ_WIDTH}")

    if args.du_nic_addrs is not None:
        DU_NIC_ADDRS = [str(item) for item in args.du_nic_addrs.split(',')]

    if args.ru_nic_addrs is not None:
        RU_NIC_ADDRS = [str(item) for item in args.ru_nic_addrs.split(',')]
    print(f"DU_NIC_ADDRS {DU_NIC_ADDRS}")
    print(f"RU_NIC_ADDRS {RU_NIC_ADDRS}")
    print(f"Assume NIC addresses in the same index of the DU/RU list are connected physically")

    if args.mtu is not None:
        MTU = args.mtu
    print(f"MTU {MTU}")

    if args.cell_ports is not None:
        cell_ports = [int(item) for item in args.cell_ports.split(',')]
        q, r = divmod(args.num_cells, len(cell_ports))
        CELL_PORT_IDX = q * cell_ports + cell_ports[:r]
    else:
        cell_ports = CELL_PORT_IDX
        q, r = divmod(args.num_cells, len(cell_ports))
        CELL_PORT_IDX = q * cell_ports + cell_ports[:r]
    print(f"CELL PORT IDX {CELL_PORT_IDX}")

    if args.pcp is not None:
        pcp = [int(item) for item in args.pcp.split(',')]
        q, r = divmod(args.num_cells, len(pcp))
        CELL_pcp = q * pcp + pcp[:r]
    else:
        pcp = CELL_PCP
        q, r = divmod(args.num_cells, len(pcp))
        CELL_PCP = q * pcp + pcp[:r]
    print(f"CELL_PCP {CELL_PCP}")


    if args.vlan is not None:
        vlan = [int(item) for item in args.vlan.split(',')]
        q, r = divmod(args.num_cells, len(vlan))
        CELL_VLAN = q * vlan + vlan[:r]
    else:
        vlan = CELL_VLAN
        q, r = divmod(args.num_cells, len(vlan))
        CELL_VLAN = q * vlan + vlan[:r]
    print(f"CELL_VLAN {CELL_VLAN}")


    if args.du_ul_cores is not None:
        DU_UL_CORE_LIST = [int(item) for item in args.du_ul_cores.split(',')]
    print(f"DU_UL_CORE_LIST {DU_UL_CORE_LIST}")

    if args.du_dl_cores is not None:
        DU_DL_CORE_LIST = [int(item) for item in args.du_dl_cores.split(',')]
    print(f"DU_DL_CORE_LIST {DU_DL_CORE_LIST}")

    if args.ru_ul_cores is not None:
        RU_UL_CORE_LIST = [int(item) for item in args.ru_ul_cores.split(',')]
    print(f"RU_UL_CORE_LIST {RU_UL_CORE_LIST}")

    if args.ru_dl_cores is not None:
        RU_DL_CORE_LIST = [int(item) for item in args.ru_dl_cores.split(',')]
    print(f"RU_DL_CORE_LIST {RU_DL_CORE_LIST}")

    if args.ul_antennas is not None:
        NUM_UL_ANT = args.ul_antennas
        output_filename +=  "_ulant_" + str(NUM_UL_ANT)
    print(f"NUM_UL_ANT {NUM_UL_ANT}")

    if args.ul_antennas is not None:
        NUM_DL_ANT = args.ul_antennas
        output_filename +=  "_dlant_" + str(NUM_DL_ANT)
    print(f"NUM_DL_ANT {NUM_DL_ANT}")

    if args.enable_ulu is not None:
        ENABLE_ULU = args.enable_ulu
    print(f"ENABLE_ULU {ENABLE_ULU}")

    if args.enable_dlu is not None:
        ENABLE_DLU = args.enable_dlu
    print(f"ENABLE_DLU {ENABLE_DLU}")

    if args.enable_ulc is not None:
        ENABLE_ULC = args.enable_ulc
    print(f"ENABLE_ULC {ENABLE_ULC}")

    if args.enable_dlc is not None:
        ENABLE_DLC = args.enable_dlc
    print(f"ENABLE_DLC {ENABLE_DLC}")

    if args.test_slots is not None:
        TEST_SLOTS = args.test_slots
        output_filename +=  "_testslots_" + str(TEST_SLOTS)
    print(f"TEST_SLOTS {TEST_SLOTS}")

    if args.oam_port_id is not None:
        OAM_PORT_ID = args.oam_port_id
        print(f"Using OAM port ID {OAM_PORT_ID}")
    if args.sfn_slot_sync_ru is not None:
        SFN_SLOT_SYNC_RU_IP = args.sfn_slot_sync_ru + ":" + str(OAM_PORT_ID)
        print(f"SFN_SLOT_SYNC_RU_IP {SFN_SLOT_SYNC_RU_IP}")
    if args.sfn_slot_sync_du is not None:
        SFN_SLOT_SYNC_DU_IP = args.sfn_slot_sync_du + ":" + str(OAM_PORT_ID)
        print(f"SFN_SLOT_SYNC_DU_IP {SFN_SLOT_SYNC_DU_IP}")

    if args.dlc_ontime_pass_percentage is not None:
        DLC_ONTIME_PASS_PERCENTAGE = args.dlc_ontime_pass_percentage
    print(f"DLC_ONTIME_PASS_PERCENTAGE {DLC_ONTIME_PASS_PERCENTAGE}")

    if args.dlu_ontime_pass_percentage is not None:
        DLU_ONTIME_PASS_PERCENTAGE = args.dlu_ontime_pass_percentage
    print(f"DLU_ONTIME_PASS_PERCENTAGE {DLU_ONTIME_PASS_PERCENTAGE}")

    if args.ulc_ontime_pass_percentage is not None:
        ULC_ONTIME_PASS_PERCENTAGE = args.ulc_ontime_pass_percentage
    print(f"ULC_ONTIME_PASS_PERCENTAGE {ULC_ONTIME_PASS_PERCENTAGE}")

    if args.ulu_ontime_pass_percentage is not None:
        ULU_ONTIME_PASS_PERCENTAGE = args.ulu_ontime_pass_percentage
    print(f"ULU_ONTIME_PASS_PERCENTAGE {ULU_ONTIME_PASS_PERCENTAGE}")

    if args.ulu_tx_adv_ns is not None:
        ULU_TX_TIME_ADVANCE_NS = args.ulu_tx_adv_ns
    print(f"ULU_TX_TIME_ADVANCE_NS {ULU_TX_TIME_ADVANCE_NS}")

    if args.ulu_enq_adv_ns is not None:
        ULU_ENQ_TIME_ADVANCE_NS = args.ulu_enq_adv_ns
    print(f"ULU_ENQ_TIME_ADVANCE_NS {ULU_ENQ_TIME_ADVANCE_NS}")

    if args.dlc_enq_adv_ns is not None:
        DLC_ENQ_TIME_ADVANCE_NS = args.dlc_enq_adv_ns
    print(f"DLC_ENQ_TIME_ADVANCE_NS {DLC_ENQ_TIME_ADVANCE_NS}")

    if args.dlu_enq_adv_ns is not None:
        DLU_ENQ_TIME_ADVANCE_NS = args.dlu_enq_adv_ns
    print(f"DLU_ENQ_TIME_ADVANCE_NS {DLU_ENQ_TIME_ADVANCE_NS}")

    if args.ontime_window_ns is not None:
        ONTIME_WINDOW_NS = args.ontime_window_ns
    print(f"ONTIME_WINDOW_NS {ONTIME_WINDOW_NS}")

    output_filename += ".yaml"
    if(args.out_filename is not None):
        output_filename = args.out_filename
    print(f"Command line params: {' '.join(sys.argv[1:])}")
    print(f"Output file name: {output_filename}")

    print("Running...")
    num_slots, g = tfgrid_from_file(args.pattern_filename, NUM_CELLS=args.num_cells)
    prb_list = tf_grid_to_prb_list(g)
    fhgen = FHGen(NUM_CELLS=args.num_cells, NUM_SLOTS=num_slots, SYSARGS=' '.join(sys.argv[1:]))
    fhgen.config(prb_list,g)
    fhgen.write(output_filename)
    print("Execute test with:")
    print(f"sudo -E ./build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator {output_filename}")
    print("End.")