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
import argparse
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import sys

base = argparse.ArgumentParser()
base.add_argument(
    "--filename",
    type=str,
    dest="filename",
    help="Specifies the name of the file containing the DLBFW results for the avg. cells",
    required=True,
)
base.add_argument(
    "--pdsch",
    type=str,
    dest="pdsch",
    help="Specifies the name of the file containing the PDSCH results",
    required=True,
)
args = base.parse_args()

folder, fn = os.path.split(args.filename)
target = int(fn[0:3])

ifile = open(args.filename, "r")
sweeps = json.load(ifile)
ifile.close()

folder, fn = os.path.split(args.pdsch)
pdsch_target = int(fn[0:3])

if target != pdsch_target:
    sys.exit("error: number of SMs for DLBFW and PDSCH pipeline need to be equal")

ifile = open(args.pdsch, "r")
pdsch = json.load(ifile)
ifile.close()

bwc_latencies = sweeps
pdsch_latencies = pdsch

final = list(product(bwc_latencies, pdsch_latencies))
latencies = [np.sum(x) for x in final]

y, x = np.histogram(latencies, bins=10000)
cy = np.cumsum(y) / len(latencies)

sms = int(os.path.split(args.filename)[-1].split("_")[0])

plt.plot(x[1:], cy)
plt.grid(True)
plt.ylabel("CDF")
plt.xlabel("Latency [us]")
plt.title("PDSCH + DLBFW: " + str(sms) + "SM")
plt.vlines(500, 0, 1, colors="r")
plt.legend(["Measurements", "Constraint"])

plt.savefig(args.filename.replace(".json", ".png"))
