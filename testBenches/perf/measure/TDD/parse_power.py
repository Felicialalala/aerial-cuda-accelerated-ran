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

def parse_power(lines):
    """
    Parse nvidia-smi CSV output from execute.py.
    
    Expected input format: CSV lines from nvidia-smi query with the following columns:
    clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], 
    memory.used [MiB], utilization.gpu [%], utilization.memory [%], temperature.gpu
    
    Example input lines:
    clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], memory.used [MiB], utilization.gpu [%], utilization.memory [%], temperature.gpu
    1980 MHz, 2619 MHz, 143.28 W, 8 MiB, 0 %, 0 %, 34
    1875 MHz, 2619 MHz, 145.12 W, 12 MiB, 5 %, 2 %, 35
    
    Returns:
        List of lists, where each inner list contains 7 string values:
        [sm_clock, mem_clock, power_draw, memory_used, gpu_util, memory_util, temperature]
        Units are stripped from the values (e.g., "143.28" instead of "143.28 W")
    """
    results = []
    # the input lines should be the result from nvidia-smi query by execute.py
    for line in lines:
        # Skip the header line
        if "clocks.current.sm [MHz]" in line:
            continue
            
        # Split by comma and strip whitespace
        lst = [item.strip() for item in line.split(',')]
        
        # Check if we have the expected 7 columns of data
        if len(lst) == 7:
            try:
                result = []
                # Extract numeric values from each field
                # SM Clock (MHz)
                sm_clock = lst[0].replace(' MHz', '').strip()
                result.append(sm_clock)
                
                # Memory Clock (MHz) 
                mem_clock = lst[1].replace(' MHz', '').strip()
                result.append(mem_clock)
                
                # Power Draw (W)
                power_draw = lst[2].replace(' W', '').strip()
                result.append(power_draw)
                
                # Memory Used (MiB)
                memory_used = lst[3].replace(' MiB', '').strip()
                result.append(memory_used)
                
                # GPU Utilization (%)
                gpu_util = lst[4].replace(' %', '').strip()
                result.append(gpu_util)
                
                # Memory Utilization (%)
                memory_util = lst[5].replace(' %', '').strip()
                result.append(memory_util)
                
                # Temperature (C)
                temperature = lst[6].strip()
                result.append(temperature)

                results.append(result)
            except (ValueError, IndexError):
                # Skip malformed lines
                import warnings
                warnings.warn(f"Malformed line from nvidia-smi query: {line}")
                continue

    return results
