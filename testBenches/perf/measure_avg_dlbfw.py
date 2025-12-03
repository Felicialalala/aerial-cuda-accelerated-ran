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
import json
import yaml
import argparse
import numpy as np
from cuda.bindings import runtime
from progress.bar import IncrementalBar

base = argparse.ArgumentParser()
base.add_argument(
    "--cuphy",
    type=str,
    dest="cfld",
    help="Specifies the folder where cuPHY has been built",
    required=True,
)
base.add_argument(
    "--vectors",
    type=str,
    dest="vfld",
    help="Specifies the folder for the test vectors",
    required=True,
)
base.add_argument(
    "--config",
    type=str,
    dest="config",
    help="Specifies the file contaning the test cases list",
    required=True,
)
base.add_argument(
    "--uc",
    type=str,
    dest="uc",
    help="Specifies the file contaning the use case config",
    required=True,
)
base.add_argument(
    "--target",
    type=int,
    dest="target",
    help="Specified the resource percetage that the MPS should allocate to the run",
)
base.add_argument(
    "--gpu",
    type=int,
    dest="gpu",
    default=0,
    help="Specifies on which GPU to run the measurements",
)
base.add_argument(
    "--freq",
    type=int,
    dest="freq",
    help="Specifies the frequency at which the GPU will be set for the measurements",
    required=True,
)
base.add_argument(
    "--iterations",
    type=int,
    dest="iterations",
    default=1000,
    help="Specifies number of iterations to use in the averaging",
)
base.add_argument(
    "--slots",
    type=int,
    dest="slots",
    default=1,
    help="Specifies number of sweep iterations",
)
base.add_argument(
    "--power",
    type=int,
    dest="power",
    help="Specifies the maximum power draw for the GPU used for the measurements",
)
args = base.parse_args()

buffer = args.uc.split("_")
output = (
    "_".join(["sweep_streams_avg", buffer[2], "DLBFW", buffer[3], buffer[4]]) + ".json"
)

if args.target is not None:
    output = str(args.target).zfill(3) + "_" + output

# Get device count
err, device_count = runtime.cudaGetDeviceCount()
if err != runtime.cudaError_t.cudaSuccess:
    raise RuntimeError(f"Failed to get device count: {err}")

# Determine which device to use
if device_count == 1:  # with CUDA_VISIBLE_DEVICES set, only 1 device is visible
    device_id = 0
else:
    device_id = args.gpu

# Set the device
err = runtime.cudaSetDevice(device_id)
if err != (runtime.cudaError_t.cudaSuccess,):
    raise RuntimeError(f"Failed to set device {device_id}: {err}")

# Get multiprocessor count (SM count)
err, sms = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, device_id)
if err != runtime.cudaError_t.cudaSuccess:
    raise RuntimeError(f"Failed to get multiprocessor count: {err}")

target = int(np.floor(100 * args.target / sms))

os.system(f"sudo nvidia-smi -i {args.gpu} -pm 1 >/dev/null")
os.system(f"sudo nvidia-smi -i {args.gpu} -lgc {args.freq}")
if args.power is not None:
    os.system(f"sudo nvidia-smi -i {args.gpu} -pl {args.power}")

if args.target is not None:
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control -d"
    )

try:

    command = os.path.join(args.cfld, "examples/bfc/cuphy_ex_bfc")
    iterations = args.iterations

    sweeps = {}

    ifile = open(args.uc, "r")
    cases = json.load(ifile)
    ifile.close()

    ifile = open(args.config, "r")
    config = json.load(ifile)
    ifile.close()

    cases = cases["F14 - DLBFW"]
    config = config["F14 - DLBFW"]

    payload = {}
    payload["cells"] = len(cases)
    payload["slots"] = []

    channels = {}
    channels["BFC"] = []

    for case in cases:
        channels["BFC"].append(os.path.join(args.vfld, config[case]))

    payload["slots"].append(channels)

    ofile = open("bwc.yaml", "w")
    yaml.dump(payload, ofile)
    ofile.close()

    if args.target is None:
        system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_DEVICE_MAX_CONNECTIONS=32 {command} -i bwc.yaml -r {iterations} >buffer.txt"
    else:
        system = f"CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={target} CUDA_DEVICE_MAX_CONNECTIONS=32 {command} -i bwc.yaml -r {iterations} >buffer.txt"

    latencies = []

    bar = IncrementalBar(message=str(len(cases)) + " cell(s)", max=args.slots)

    for sweep_idx in range(args.slots):

        os.system(system)

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        latency = float(lines[-1].split()[-1])

        latencies.append(latency)

        bar.next()

    bar.finish()
    os.remove("bwc.yaml")

    ofile = open(output, "w")
    json.dump(latencies, ofile, indent=2)
    ofile.close()
finally:
    os.system(f"sudo nvidia-smi -i {args.gpu} -pm 0 >/dev/null")
    os.system(f"sudo nvidia-smi -i {args.gpu} -rgc")

    if args.target is not None:
        os.system(
            f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control"
        )

    if os.path.isfile("buffer.txt"):
        os.remove("buffer.txt")

    if os.path.isfile("bwc.yaml"):
        os.remove("bwc.yaml")
