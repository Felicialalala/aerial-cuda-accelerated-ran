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

# Description: this python file extracts all channel configurations from a phase-4 launch pattern yaml file, it can also update the downlink channel configurations in phase-3 config file measure/TDD/slotConfig.json
#       extract all channels and print the channel configurations to terminal
#       extract 'PDSCH', 'PDCCH_DL', 'CSI_RS', 'PBCH' and automaticallt convert it to json codes by four arrays
#       replace the json codes with the current ones in cuphy_testWrkr.hpp

# usage: extractDlChanConfig.py <phase-4 yaml file> <start slot> <end slot> <path to cuphy_testWrkr.hpp>
#       <phase-4 yaml file>: required, launch pattern file
#       <start slot> <end slot>: required, only extract channels for [start slot, end slot] (both inclusive)
#       <path to slotConfig>: optional, an existing header file in phase-3 testbench. If given, the DL channel configs will be updated; Otherwise, no C++ code modification

# example: 
#       only check channel config:       extractDlChanConfig.py $TV_DIR/launch_pattern_F08_1C_64.yaml 0 14
#       check and modify chanel config:  extractDlChanConfig.py $TV_DIR/launch_pattern_F08_1C_64.yaml 0 14 measure/TDD/slotConfig.json

# note: 
#       the array size depends on the start and end slots; For mMIMO pattern, please make sure include 15 slots (i.e., end - start + 1= 15)
#       no need to rebulid phase-3 C testbench but the yaml file has to regenerated using python interface

import yaml
import argparse
import numpy as np
import json

# Function to load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to get type fields for a range of slot numbers
def get_types_by_slot_range(data, start_slot, end_slot):
    result = {}
    for entry in data:
        slot = entry.get('slot')
        if start_slot <= slot <= end_slot:
            for config in entry.get('config', []):
                result[slot] = config.get('type', [])
    return result

# Function to get unique channels from the data
def get_unique_channels(types):
    unique_channels = set()
    for type_list in types.values():
        unique_channels.update(type_list)
    return sorted(unique_channels)

# Function to create a binary array
def create_binary_array(types, unique_channels, start_slot, end_slot):
    num_slots = int(end_slot - start_slot + 1)
    num_channels = len(unique_channels)
    binary_array = np.zeros((num_channels, num_slots), dtype=int)

    for slot, type_list in types.items():
        slot_index = int(slot - start_slot)
        for channel in type_list:
            channel_index = unique_channels.index(channel)
            binary_array[channel_index, slot_index] = 1

    # Convert the NumPy array to a nested list
    binary_array_list = binary_array.tolist()

    return binary_array_list

# Function to generate json code for a subset of channels with custom names
def generate_json_code(binary_array, unique_channels, subset_channels, json_array_channelNames, numSlots):
    json_dict = {}
    for i, channel in enumerate(subset_channels):
        if channel in unique_channels:
            channel_index = unique_channels.index(channel)
            json_dict[json_array_channelNames[i]] = list(binary_array[channel_index])
    return json_dict

# Custom JSON dump function to control formatting
def custom_json_dump(data, file_path):
    with open(file_path, 'w') as file:
        file.write('{\n')
        for i, (key, value) in enumerate(data.items()):
            file.write(f'    "{key}":\n')
            file.write(f'    {{\n')
            for j, (sub_key, sub_value) in enumerate(value.items()):
                file.write(f'        "{sub_key}": {json.dumps(sub_value)}')
                if j < len(value) - 1:
                    file.write(',')
                file.write('\n')
            file.write('    }')
            if i < len(data) - 1:
                file.write(',')
            file.write('\n')
        file.write('}\n')
        
# Main function
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract type fields from YAML file based on slot number range.')
    parser.add_argument('yaml_file', type=str, help='Path to the YAML file')
    parser.add_argument('start_slot', type=float, help='Start slot number (inclusive)')
    parser.add_argument('end_slot', type=float, help='End slot number (inclusive)')
    parser.add_argument('header_file', type=str, nargs='?', default=None, help='Path to the existing header file (optional)')

    # Parse arguments
    args = parser.parse_args()

    # Load the YAML file
    data = load_yaml(args.yaml_file)['SCHED']

    # Get the type fields for the specified slot range
    types = get_types_by_slot_range(data, args.start_slot, args.end_slot)

    # Get unique channels
    unique_channels = get_unique_channels(types)

    # Create binary array
    binary_array = create_binary_array(types, unique_channels, args.start_slot, args.end_slot)

    # Print the binary array for all channels
    print(f"{'Slot':<15}{' '.join(f' {i}' for i in range(int(args.start_slot), int(args.end_slot) + 1))}")
    print("-" * (15 + (int(args.end_slot) - int(args.start_slot) + 1) * 6))
    for i, row in enumerate(binary_array):
        print(f"{unique_channels[i]:<15} {', '.join(map(str, row))}")
    
    # Replace the relevant lines in the header file if provided
    if args.header_file:
        print("-" * (15 + (int(args.end_slot) - int(args.start_slot) + 1) * 6))
        # Hardcoded subset of channels and corresponding C++ array names
        subset_channels = ['PDSCH', 'PDCCH_DL', 'CSI_RS', 'PBCH', 'MAC']
        json_array_channelNames = ['PDSCH', 'PDCCH', 'CSIRS', 'PBCH', 'MAC']

        # Generate C++ code for the specified subset of channels with custom names
        json_dict = generate_json_code(binary_array, unique_channels, subset_channels, json_array_channelNames, args.end_slot - args.start_slot + 1)

        # Replace the relevant lines in the header file
        with open(args.header_file, 'r') as file:
            data = json.load(file)
            data["dddsuudddd_mMIMO"] = json_dict
            
        custom_json_dump(data, args.header_file)
        
        print("slotConfig.json has been updated successfully.")
    
if __name__ == "__main__":
    main()