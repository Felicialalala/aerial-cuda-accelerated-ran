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

""" extract unique .h5 TV file names from a launch pattern yaml file and check their existence in a given TV directory

usage: python3 checkTvExistInLp.py <yaml_file_path> <tv_directory_to_check> [verbose]
       - yaml_file_path: path to the launch pattern yaml file
       - tv_directory_to_check: directory to check for .h5 TV files
       - verbose: optional flag to print additional information. 0: summary only; 1: summary and not found TVs; >1: summary and all TVs. (default: 1)
        
example: suppose in aerial_sdk/5GModel/nr_matlab foldler
        python3 test/checkLpTvExistInDir.py GPU_test_input/launch_pattern_nrSim_90634.yaml GPU_test_input
"""

import yaml
import os
import argparse

# Load yaml file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Extract unique .h5 file names from yaml data
def extract_unique_h5_filenames(yaml_data):
    h5_files = set()
    if isinstance(yaml_data, dict):
        for key, value in yaml_data.items():
            if isinstance(value, str) and value.endswith('.h5'):
                h5_files.add(value)
            elif isinstance(value, (list, dict)):
                h5_files.update(extract_unique_h5_filenames(value))
    elif isinstance(yaml_data, list):
        for item in yaml_data:
            if isinstance(item, str) and item.endswith('.h5'):
                h5_files.add(item)
            elif isinstance(item, (list, dict)):
                h5_files.update(extract_unique_h5_filenames(item))
    return list(h5_files)

# Check if .h5 files exist in the given directory
def check_files_in_directory(h5_files, directory):
    results = {}
    for file_name in h5_files:
        file_path = os.path.join(directory, file_name)
        results[file_name] = os.path.exists(file_path)
    return results

# Main function
def main(yaml_file_path, tv_directory_to_check, verbose):
    yaml_data = load_yaml(yaml_file_path)
    h5_files = extract_unique_h5_filenames(yaml_data)
    
    existence_results = check_files_in_directory(h5_files, tv_directory_to_check)

    found_count = 0
    not_found_count = 0
    
    if verbose > 1:
        print("Extracted unique .h5 files:")
        print(h5_files)
    
    for file_name, exists in existence_results.items():
        if exists:
            found_count += 1
            if verbose > 1:
                print(f"{file_name} in {tv_directory_to_check}: Found")
        else:
            not_found_count += 1
            if verbose >= 1:
                print(f"{file_name} in {tv_directory_to_check}: Not Found")

    # Print summary
    print("\nSummary:")
    print(f"Total unique .h5 files: {len(h5_files)}")
    print(f"Found: {found_count}")
    print(f"Not Found: {not_found_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract unique .h5 TV file names from a launch pattern yaml file and check their existence in a given TV directory.")
    parser.add_argument("yaml_file_path", type=str, help="path to the launch pattern yaml file.")
    parser.add_argument("tv_directory_to_check", type=str, help="directory to check for .h5 TV files.")
    parser.add_argument("verbose", type=int, nargs='?', default=1, help="optional flag to print additional information. 0: summary only; 1: summary and not found TVs; >1: summary and all TVs. (default: 1)")

    args = parser.parse_args()
    
    main(args.yaml_file_path, args.tv_directory_to_check, args.verbose)