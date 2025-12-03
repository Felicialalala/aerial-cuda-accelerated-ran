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

import os
import subprocess

if __name__ == "__main__":

    from measure.cli import arguments

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    base, args = arguments()

    # save GPU status (current clock frequency, power limit, persistence mode) restore after running test
    gpuStatSave = {}
    command = f"nvidia-smi -i {args.gpu} --query-gpu=clocks.current.graphics,power.limit,persistence_mode --format=csv,noheader"
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, encoding="utf-8")
    gpuStatSave['clockFreq']   = int(result.stdout.strip().split()[0])
    gpuStatSave['powerLimit']  = float(result.stdout.strip().split()[2])
    gpuStatSave['persistMode'] = result.stdout.strip().split()[4]
    
    # check whether need to change GPU configs
    if gpuStatSave['clockFreq'] != args.freq:
        if args.is_GH200:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {args.freq} --mode=1") # GH200 specific to set clock frequency
        else:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {args.freq}")

    if (args.power is not None) and (gpuStatSave['powerLimit'] != args.power):
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pl {args.power}")
            
    if(gpuStatSave['persistMode'] == 'Disabled'): # Enable persistence mode to run test if disabled
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 1 >/dev/null")
    
    # start testing
    if args.mig is not None:

        # Clean up any existing MIG instances first
        result = os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -dci > buffer.txt")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to destroy MIG compute instances. Make sure you have proper permissions.")
            
        result = os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -dgi > buffer.txt")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to destroy MIG GPU instances. Make sure you have proper permissions.")
        
        # Disable MIG mode first (to clean state)
        result = os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 0 > buffer.txt")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to disable MIG mode. Make sure you have proper permissions and the GPU supports MIG.")
        
        # Enable MIG mode
        result = os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 1 > buffer.txt")
        if result != 0:
            if os.path.exists("buffer.txt"):
                os.remove("buffer.txt")
            base.error("Failed to enable MIG mode. Make sure you have proper permissions and MIG-capable GPU.")

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if lines[0].split()[0] == "Enabled" and lines[0].split()[1] == "MIG":
            import measure.mig

            measure.mig.measure(base, args)
            
        else:

            base.error("encountered issues in enabling MIG on the selected GPU")

    else:

        import measure.nomig

        measure.nomig.measure(base, args)

    # restore GPU status
    if gpuStatSave['clockFreq'] != args.freq:
        if args.is_GH200:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {gpuStatSave['clockFreq']} --mode=1")
        else:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -lgc {gpuStatSave['clockFreq']}")
        
    if (args.power is not None) and (gpuStatSave['powerLimit'] != args.power):
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pl {gpuStatSave['powerLimit']}")
            
    if(gpuStatSave['persistMode'] == 'Disabled'):
        os.system(f"{sudo}nvidia-smi -i {args.gpu} -pm 0 >/dev/null")
