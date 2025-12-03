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

def parse(args, lines):

    clat_tot = []
    clat_ul = []
    clat_dl = []

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

    while k < len(lines):
        lst = lines[k].split()

        # currently only GPU memory are traced
        if(len(lst) == 17 and lst[5] == 'cuphyMemoryFootprint' and lst[7] == 'GPU'):
            memoryUseMB[lst[13]] += float(lst[9])
        
        elif len(lst) == 3 and lst[0] == "Slot" and lst[1] == "#":

            k += 1

            lst = lines[k].split()

            if len(lst) > 0 and lst[0] == "average" and lst[1] == "slot":
                clat_tot.append(float(lst[4]))

                k += 1

                while k < len(lines) and "----" not in lines[k]:
                    lst = lines[k].split()

                    if len(lst) > 3:

                        if lst[0] == "Ctx" and lst[4] == "PDSCH":
                            clat_dl.append(float(lst[7]))
                            k += 1
                            continue

                        if lst[0] == "Ctx" and lst[4] == "PUSCH":
                            clat_ul.append(float(lst[7]))
                            k += 1
                            continue

                    k += 1

        k += 1

    latencies = {}
    latencies["Total"] = clat_tot
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH"] = clat_ul

    # calculate all memory usage
    memoryUseMB["totalUlNoBFW"] = memoryUseMB["PUSCH"] + memoryUseMB["PUCCH"] + memoryUseMB["SRS"] + memoryUseMB["PRACH"]
    memoryUseMB["totalDlNoBFW"] = memoryUseMB["PDSCH"] + memoryUseMB["PDCCH"] + memoryUseMB["SSB"] + memoryUseMB["CSIRS"]
    memoryUseMB["totalNoBFW"] = memoryUseMB["totalUlNoBFW"] + memoryUseMB["totalDlNoBFW"]
    memoryUseMB["total"] = memoryUseMB["totalNoBFW"] + memoryUseMB["BFW"]
    
    results = latencies
    results['memoryUseMB'] = memoryUseMB
    
    return results
