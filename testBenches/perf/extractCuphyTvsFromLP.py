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

# Description: this python file extracts all unique cuPHY TVs for each channel from a phase-4 launch pattern yaml file, it can also run phase-1 tests for the TVs and find the one with the worst running time. All results saving in extractCuphyTvsFromLP.json

# usage: extractCuphyTvsFromLP.py <phase-4 yaml file> <start slot> <end slot> <path to cuPHY phase-1 executables>
#       <phase-4 yaml file>: required, launch pattern file
#       <start slot> <end slot>: required, only extract channels for [start slot, end slot] (both inclusive)
#       <path to cuPHY phase-1 executables>: optional, can be aerial_sdk/build/cuPHY/examples or aerial_sdk/cuPHY/build/examples

# example: 
#       only check unique cuPHY TVs:     extractCuphyTvsFromLP.py $TV_DIR/launch_pattern_F08_1C_64.yaml 0 79
#       check and modify channel config:  extractCuphyTvsFromLP.py $TV_DIR/launch_pattern_F08_1C_64.yaml 0 79 ../../cuPHY/build/examples

# note: 
#       the array size depends on the start and end slots; For mMIMO pattern, please make sure include 15 slots (i.e., end - start + 1= 15). It's common to use start = 0, but not enforced
#       no need to rebuild phase-3 C testbench but the yaml file has to regenerated using python interface


import yaml
import argparse
import subprocess
import re, json
# Function to load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to get type fields for a range of slot numbers
def get_types_by_slot_range(data, start_slot, end_slot):
    channels = {}
    tvNames = {}
    for entry in data:
        slot = entry.get('slot')
        if start_slot <= slot <= end_slot:
            for config in entry.get('config', []):
                channels[slot] = config.get('type', [])
                tvNames[slot] = config.get('channels', [])
    return channels, tvNames

# Function to get unique channels from the data
def get_unique_channels(types):
    unique_channels = set()
    for type_list in types.values():
        unique_channels.update(type_list)
    return sorted(unique_channels)

def find_files(channels, fapiTvName, allTvFiles):
    cuPHYTvNames = {}
    for slotIdx in fapiTvName.keys():
        cuPHYTvNames[slotIdx] = []
        channelTypes = channels[slotIdx]
        
        # Replace "PDCCH_DL" and "PDCCH_UL" to "PDCCH"
        if "PDCCH_DL" in channels or "PDCCH_UL" in channelTypes:
            channelTypes.append("PDCCH")
        # Replace "CSI_RS" to "CSIRS"
        if "CSI_RS" in channelTypes:
            channelTypes.append("CSIRS")
        # Replace "PBCH" to "SSB"
        if "PBCH" in channelTypes:
            channelTypes.append("SSB")
        channelTypes = [ch for ch in channelTypes if ch not in ("PDCCH_DL", "PDCCH_UL", "CSI_RS", "PBCH")]
        # phase-4 launch pattern does not explicitly have BFW
        # BFW may exist if there is PDSCH, PUSCH or SR
        if "PDSCH" in channelTypes or "PUSCH" in channelTypes or "SRS" in channelTypes:
            channelTypes.append("BFW")

        # Extract the search pattern from the given string
        for fapiTvNameCurr in fapiTvName[slotIdx]:  
            search_pattern = fapiTvNameCurr.split('_gNB')[0]
            matching_files = [f for f in allTvFiles if search_pattern in f and 'FAPI' not in f]
            
            valid_files = []
            for cuphyTv in matching_files:
                pattern = re.compile(r'(\d{4})_([A-Z]+)')
                match = pattern.search(cuphyTv)
                if match:
                    channel_type = match.group(2)  # Extract the channel type
                    if channel_type in channelTypes:
                        valid_files.append(cuphyTv)
            cuPHYTvNames[slotIdx] += valid_files
    return cuPHYTvNames

def extract_unique_tvs(data):
    # Initialize a dictionary to hold unique TV names for each channel type
    unique_tvs = {}

    # Regular expression to match the four-digit number
    pattern = re.compile(r'(\d{4})_([A-Z]+)')

    # Iterate over each key-value pair in the input dictionary
    for key in data:
        for file in data[key]:
            # Use regex to find the channel type after the four-digit number
            match = pattern.search(file)
            if match:
                channel_type = match.group(2)  # Extract the channel type

                # Initialize the set for the channel type if it doesn't exist
                if channel_type not in unique_tvs:
                    unique_tvs[channel_type] = set()

                # Add the full TV name to the set
                unique_tvs[channel_type].add(file)
    # Convert sets to lists for easier readability
    for channel_type in unique_tvs:
        unique_tvs[channel_type] = list(sorted(unique_tvs[channel_type]))
    return unique_tvs

def extractTiming(channel_type, output):
    match channel_type:
        case "PUSCH":
            reMatch = re.search(r"GPU \(CUDA event\) (\d+\.\d+)", output)
        case "PUCCH":
            reMatch = re.search(r"PucchRx Pipeline\[00\]: Metric - GPU Time usec \(using CUDA events, over \d+ runs\): Run\s+([\d.]+)", output)
        case "PRACH":
            reMatch = re.search(r"\[CUPHY\.PRACH_RX\] Slot # \d+,  PRACH pipeline\(s\): ([\d.]+) us \(avg\. over \d+ iterations\)", output)
        case "SRS":
            reMatch = re.search(r"SrsRx Pipeline\[00\]: Metric - GPU Time usec \(using CUDA events, over \d+ runs\): Run\s+([\d.]+)", output)
        case "PDSCH":
            reMatch = re.search(r"\[CUPHY\.PDSCH_TX\] DL pipeline: ([\d.]+) us \(avg\. over \d+ iterations\)", output)
        case "PDCCH":
            reMatch = re.search(r"\[CUPHY\.PDCCH_TX\] PDCCH TX pipeline: ([\d.]+) us \(avg\. over \d+ iterations\)", output)
        case "CSIRS":
            reMatch = re.search(r"\[CUPHY\.CSIRS_TX\] CSI-RS TX Pipeline Only Run \(in (Stream|Graphs) mode\): ([\d.]+) us \(avg\. over \d+ iterations\)", output)
        case "SSB":
            reMatch = re.search(r"\[CUPHY\.SSB_TX\] SSB TX Pipeline in (Stream|Graphs) mode: ([\d.]+) us \(avg\. over \d+ iterations\)", output)
        case "BFW":
            reMatch = re.search(r"\[CUPHY\.BFW\] Slot\[0\]: Average \(\d+ runs\) elapsed time in usec \(CUDA event w/ \d+ ms delay kernel\) = ([\d.]+)", output)
        case _:
            return 0
    
    if(channel_type == "CSIRS" or channel_type == "SSB"):
        return float(reMatch.group(2)) # first is Stream/Graphs, second is latency
    else:
        return float(reMatch.group(1)) # first is latency

def run_phase_2_tests(path_to_tv, path_to_execut, cuPHY_tvs, procMode):
    nItr = 1000
    allLatency = {} # latency for each TV
    worstCaseTVs = {}
    executables = {
        "PUSCH": "pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe",
        "PUCCH": "pucch_rx_pipeline/cuphy_ex_pucch_rx_pipeline",
        "PRACH": "prach_receiver_multi_cell/prach_receiver_multi_cell",
        "SRS": "srs_rx_pipeline/cuphy_ex_srs_rx_pipeline",
        "PDSCH": "pdsch_tx/cuphy_ex_pdsch_tx",
        "PDCCH": "pdcch/embed_pdcch_tf_signal",
        "CSIRS": "csi_rs/nzp_csi_rs_test",
        "SSB": "ss/testSS",
        "BFW": "bfc/cuphy_ex_bfc"
    }
    # add additional options
    for channel_type in cuPHY_tvs.keys():
        executable = executables.get(channel_type)
        allLatency[channel_type] = []
        for tvName in cuPHY_tvs[channel_type]:
            executable = executables.get(channel_type)
            if(channel_type == "PDSCH"):
                fullCmd = f"{path_to_execut}/{executable} {path_to_tv}/{tvName} {nItr} 0 {procMode}"
            elif(channel_type == "PUCCH"):
                fullCmd = f"{path_to_execut}/{executable} -m {procMode} -i {path_to_tv}/{tvName} -n {nItr}"
            else:
                fullCmd = f"{path_to_execut}/{executable} -m {procMode} -i {path_to_tv}/{tvName} -r {nItr}"
            
            # Execute the command
            try:
                result = subprocess.run(fullCmd, shell=True, check=True, text=True, capture_output=True)
                
                # Access the output
                output = result.stdout
                # print("Command Output:", output)
                gpu_timing = extractTiming(channel_type, output)
                allLatency[channel_type].append(gpu_timing)
                
                # Access the error output if needed
                error_output = result.stderr
                if error_output:
                    print("Error Output:", error_output)

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")
                print(f"Error Output: {e.stderr}")
                allLatency[channel_type].append(-1)
        
        # find the max
        worstTvIndex = allLatency[channel_type].index(max(allLatency[channel_type]))
        worstCaseTVs[channel_type] = cuPHY_tvs[channel_type][worstTvIndex]
        
    return allLatency, worstCaseTVs
    
# Set up argument parser
parser = argparse.ArgumentParser(description='Extract type fields from YAML file based on slot number range.')
parser.add_argument('yaml_file', type=str, help='Path to the YAML file')
parser.add_argument('start_slot', type=float, help='Start slot number (inclusive)')
parser.add_argument('end_slot', type=float, help='End slot number (inclusive)')
parser.add_argument('cuPHYbuildExpFolder', type=str, nargs='?', default=False, help='folder for phase-1 tests, can be aerial_sdk/build/cuPHY/examples or aerial_sdk/cuPHY/build/examples')

# Parse arguments
args = parser.parse_args()

procMode = 1  # default graph mode; 0 is stream mode
path_to_tv = "/mnt/cicd_tvs/develop/GPU_test_input" # default TV path
    
# Load the YAML file
data = load_yaml(args.yaml_file)['SCHED']

# Get the type fields for the specified slot range
channels, fapiTvNames = get_types_by_slot_range(data, args.start_slot, args.end_slot)

# all TVs in develop
result = subprocess.run(
            ['ls', path_to_tv],
            capture_output=True,
            text=True,
            check=True
        )
# Filter the output to find matching files
allTvFiles = result.stdout.splitlines()

cuPHYTvNames = find_files(channels, fapiTvNames, allTvFiles)

# Extract unique TV names
unique_tvs = extract_unique_tvs(cuPHYTvNames)
combineRes = {}
combineRes['unique_tvs'] = unique_tvs

# run phase-1 tests and find the one with max running time
if args.cuPHYbuildExpFolder:
    allLatency, worstCaseTVs = run_phase_2_tests(path_to_tv, args.cuPHYbuildExpFolder, unique_tvs, procMode)
    combineRes['allLatency'] = allLatency
    combineRes['worstCaseTVs'] = worstCaseTVs

with open('extractCuphyTvsFromLP.json', 'w') as json_file:
    json.dump(combineRes, json_file, indent=4)