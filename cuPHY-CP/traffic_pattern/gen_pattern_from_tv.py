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

import h5py, json, pickle
import sys, os
import numpy as np

def custom_serializer(o):
    if isinstance(o, list) and all(isinstance(i, int) for i in o):
        return "[" + ",".join(map(str, o)) + "]"
    raise TypeError


def read_from_file(filename):
    
    buffer = os.path.split(filename)

    ifile = open(filename, "rb")
    if buffer[-1].split('.')[-1] == "json":
        data = json.load(ifile)
    else: # dat by pickle
        data = pickle.load(ifile)
    ifile.close()
    return data

def dump_to_file(filename, data):

    buffer = os.path.split(filename)

    ofile = open(filename, "w")
    if buffer[-1].split('.')[-1] == "json":
        json_str = json.dumps(data, indent=2)
        # Adjust the format to put inner lists on a single line
        # Make sure the inner list is on a single line
        formatted_str = json_str.replace('[\n        ', '[').replace('\n      ]', ']')
        formatted_str = formatted_str.replace('\n        ]', ']')
        formatted_str = formatted_str.replace(',\n          ', ', ')
        formatted_str = formatted_str.replace('[[', '[\n        [')
        formatted_str = formatted_str.replace('[  ', '[')

        # Save to file
        with open(filename, "w") as file:
            file.write(formatted_str)
    else:
        pickle.dump(data, ofile, protocol=pickle.HIGHEST_PROTOCOL)
        ofile.close()
    return


nCells = 20
start_tv_ul = 3840

old_pattern = "60"
new_pattern = "60b"

ul_noPRACH_tv = start_tv_ul + nCells * 2
ul_4PRACH_tv = start_tv_ul + nCells * 1
ul_3PRACH_tv = start_tv_ul + nCells * 3

TVs = {"ul_noPRACH": ul_noPRACH_tv, 
       "ul_4PRACH": ul_4PRACH_tv,
       "ul_3PRACH": ul_3PRACH_tv}

traffic = read_from_file(f"POC2_{old_pattern}.json")
traffic["types"]["ul_4PRACH"]["pucch"] = []
traffic["types"]["ul_4PRACH"]["pusch"] = []
traffic["types"]["ul_4PRACH"]["prach"] = []
traffic["types"]["ul_3PRACH"]["pucch"] = []
traffic["types"]["ul_3PRACH"]["pusch"] = []
traffic["types"]["ul_3PRACH"]["prach"] = []
traffic["types"]["ul_noPRACH"]["pucch"] = []
traffic["types"]["ul_noPRACH"]["pusch"] = []

for slot_type, TV in TVs.items():
    
    filename = f"C:/Workspace/aerial_sdk/5GModel/nr_matlab/GPU_test_input/TVnr_ULMIX_{TV}_gNB_FAPI_s0.h5"

    print(f"------------------ TV {TV} ------------------- ")

    with h5py.File(filename, 'r') as f:
        PDU_idx = 1
        pucch_idx = 1
        pusch_idx = 1
        prach_idx = 1
        while(True):        
            if f'PDU{PDU_idx}' in f:
                pdu = f[f'PDU{PDU_idx}']
                # Check if the "type" field exists and print its value
                type = pdu['type'][()]
                if type == 7:
                    print(f"PUCCH {pucch_idx}, prbStart {pdu['prbStart']}, prbSize {pdu['prbSize']}")
                    traffic["types"][slot_type]["pucch"].append([int(pdu['StartSymbolIndex']), int(pdu['StartSymbolIndex']+pdu['NrOfSymbols']), int(pdu['prbStart']), int(pdu['prbStart'] + pdu['prbSize'])])
                    pucch_idx += 1
                elif type == 8:
                    print(f"PUSCH {pusch_idx}, prbStart {pdu['rbStart']}, prbSize {pdu['rbSize']}, lastPrb {pdu['rbStart']+pdu['rbSize']-1} isEarlyHarq{pdu['isEarlyHarq']}")
                    traffic["types"][slot_type]["pusch"].append([0, 14, int(pdu['rbStart']), int(pdu['rbStart'] + pdu['rbSize'])])
                    pusch_idx += 1
                elif type == 6:
                    prach = f[f"RO_Config_{prach_idx}"]
                    print(f"PRACH {prach_idx}, k1 {prach['k1']}")
                    traffic["types"][slot_type]["prach"].append([0, 12, int(prach['k1']), int(prach['k1'] + 12)])
                    prach_idx +=1
                else:                    
                    test = 1
                PDU_idx += 1
            else:
                break


dump_to_file(f"POC2_{new_pattern}.json", traffic)

