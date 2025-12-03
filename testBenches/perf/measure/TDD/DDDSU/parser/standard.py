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

from .common import check, unpack


def run(args, lines):

    latencies = {}
    clat_ul = []
    clat_ul_subslotProc = []
    clat_dl = []
    clat_dlbf = []
    clat_ulbf = []
    clat_sr1 = []
    clat_sr2 = []
    clat_ra = []
    clat_cdl = []
    clat_cul = []
    clat_ssb = []
    clat_cr = []
    clat_mac = []
    clat_mac2 = []

    # count GPU memory usage
    memoryUseMB={}
    memoryUseMB["PUSCH"] = 0
    memoryUseMB["PUCCH"] = 0
    memoryUseMB["SRS"] = 0
    memoryUseMB["PRACH"] = 0
    
    memoryUseMB["PDSCH"] = 0
    memoryUseMB["PDCCH"] = 0
    memoryUseMB["SSB"] = 0
    memoryUseMB["CSIRS"] = 0
    
    # TODO: beamforming GPU memory, BFW does not separate DL and UL
    memoryUseMB["BFW"] = 0
    
    k = 0

    is_correct = False

    while k < len(lines):
        lst = lines[k].split()

        # currently only GPU memory are traced
        if(len(lst) == 17 and lst[5] == 'cuphyMemoryFootprint' and lst[7] == 'GPU'):
            memoryUseMB[lst[13]] += float(lst[9])
        
        elif len(lst) == 4 and lst[0] == "Slot" and lst[1] == "pattern" and lst[2] == "#":

            k += 1

            pusch = 0
            pusch_subslotProc = 0
            pdsch = []
            dlbfw = []
            ulbfw = []
            srs1 = 0
            srs2 = 0
            rach = 0
            pdcch = []
            pucch = 0
            ssb = 0
            csirs = []
            mac = []
            mac2 = []

            while k < len(lines) and "----" not in lines[k]:
                lst = lines[k].split()

                if len(lst) > 3: # Uplink channels: use end - start for time measurement

                    if lst[0] == "Average" and lst[1] == "PUSCH" and lst[2] == "run":
                        pusch = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUSCH_subslotProc" and lst[2] == "run":
                        pusch_subslotProc = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH" and lst[2] == "run":
                        pucch = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "ULBFW" and lst[2] == "run":
                        ulbfw = float(lst[4]) - float(lst[7])
                        k += 1
                        continue
                    
                    if lst[0] == "Average" and lst[1] == "SRS1" and lst[2] == "run":
                        srs1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "SRS2" and lst[2] == "run":
                        srs2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PRACH" and lst[2] == "run":
                        rach = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                if len(lst) > 4:  # Downlink channels: use end - start of slot boundaries for time measurement, also add timeslot index for checking start times
                     
                    if lst[0] == "Slot" and lst[4] == "PDSCH":
                        pdsch.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "PDCCH":
                        pdcch.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "CSIRS":
                        csirs.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SSB":
                        ssb.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "DLBFW":
                        dlbfw.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue
                    
                    if lst[0] == "Slot" and lst[4] == "MAC":
                        mac.append([float(lst[7]) - float(lst[10])]) # calculate MAC latency here, not in unpack
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "MAC2":
                        mac2.append(float(lst[7]) - float(lst[10])) # calculate MAC2 latency here, not in unpack
                        k += 1
                        continue

                k += 1

            is_correct = is_correct or check(
                args, pusch, pusch_subslotProc, pdsch, dlbfw, ulbfw, srs1, srs2, rach, pdcch, pucch, ssb, csirs
            )
            pusch, pusch_subslotProc, pdsch, dlbfw, ulbfw, srs1, srs2, rach, pdcch, pucch, ssb, csirs = unpack(
                args, pusch, pusch_subslotProc, pdsch, dlbfw, ulbfw, srs1, srs2, rach, pdcch, pucch, ssb, csirs
            )

            if pusch > 0:
                clat_ul.append(pusch)
            if pusch_subslotProc > 0:
                clat_ul_subslotProc.append(pusch_subslotProc)
            if pucch > 0:
                clat_cul.append(pucch)

            if ssb > 0:
                clat_ssb.append(ssb)

            clat_dl.extend(pdsch)
            clat_cdl.extend(pdcch)
            clat_cr.extend(csirs)
            if args.is_rec_bf:
                clat_dlbf.extend(dlbfw)
                clat_ulbf.extend(ulbfw)
                clat_sr1.append(srs1)
                if srs2 > 0:
                    clat_sr2.append(srs2)

            if args.is_prach:
                if rach > 0:
                    clat_ra.append(rach)

            if args.is_mac:
                clat_mac += mac # mac is per pattern latency, clat_mac is all slots latency

            if args.mac2 > 0:
                clat_mac2 += mac2 # mac2 is per pattern latency, clat_mac2 is all slots latency

        k += 1

    latencies["Structure"] = is_correct
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH"] = clat_ul
    latencies["PUSCH_SUBSLOT_PROC"] = clat_ul_subslotProc
    if args.is_pdcch:
        latencies["PDCCH"] = clat_cdl

    if args.is_pucch:
        latencies["PUCCH"] = clat_cul

    if args.is_rec_bf:
        latencies["DLBFW"] = clat_dlbf
        latencies["ULBFW"] = clat_ulbf
        latencies["SRS1"] = clat_sr1
        latencies["SRS2"] = clat_sr2

    if args.is_prach:
        latencies["PRACH"] = clat_ra

    if args.is_ssb:
        latencies["SSB"] = clat_ssb

    if args.is_csirs:
        latencies["CSI-RS"] = clat_cr

    if args.is_mac:
        latencies["MAC"] = clat_mac

    if args.mac2 > 0:
        latencies["MAC2"] = clat_mac2
        
    # calculate all memory usage
    memoryUseMB["totalUlNoBFW"] = memoryUseMB["PUSCH"] + memoryUseMB["PUCCH"] + memoryUseMB["SRS"] + memoryUseMB["PRACH"]
    memoryUseMB["totalDlNoBFW"] = memoryUseMB["PDSCH"] + memoryUseMB["PDCCH"] + memoryUseMB["SSB"] + memoryUseMB["CSIRS"]
    memoryUseMB["totalNoBFW"] = memoryUseMB["totalUlNoBFW"] + memoryUseMB["totalDlNoBFW"]
    memoryUseMB["total"] = memoryUseMB["totalNoBFW"] + memoryUseMB["BFW"]
    
    results = latencies
    results['memoryUseMB'] = memoryUseMB
    
    return results
