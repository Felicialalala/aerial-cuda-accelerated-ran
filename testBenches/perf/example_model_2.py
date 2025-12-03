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

import numpy as np
import json
import argparse
import glob
import os
from copy import deepcopy

from architecture import available_resources, supported_architectures

from load_model_2 import schedule_2


def fit_oh(channel, fit_para, kernel_list, increment, min_oh, max_oh, homo_oh):

    for kernel in kernel_list:
        fit_para["launch_oh"][kernel] = min_oh

    mse = np.inf
    best_para = None

    while True:

        if homo_oh:
            print(
                "Analyzing "
                + channel
                + " overhead: "
                + str(fit_para["launch_oh"][kernel_list[0]])
            )

        error_sq = 0
        for filename in filenames:
            if channel == "DS":
                error_sq += get_error_DS(filename, fit_para)
            elif channel == "US":
                error_sq += get_error_US(filename, fit_para)

        if error_sq < mse:
            mse = error_sq
            best_para = deepcopy(fit_para["launch_oh"])

        # test next para
        overhead_buffer = fit_para["launch_oh"]

        if homo_oh is True:  # all kernels in this pipeline with the same overhead
            for idx in range(len(kernel_list)):
                overhead_buffer[kernel_list[idx]] += increment
        else:  # diff kernels may have diff overhead
            overhead_buffer[kernel_list[0]] += increment
            for idx in range(len(kernel_list)):
                if overhead_buffer[kernel_list[idx]] > max_oh:
                    if idx + 1 < len(kernel_list):
                        overhead_buffer[kernel_list[idx + 1]] += increment
                    else:
                        break
                    overhead_buffer[kernel_list[idx]] = min_oh
                else:
                    break

        for idx in range(len(kernel_list)):
            overhead_buffer[kernel_list[idx]] = np.round(
                overhead_buffer[kernel_list[idx]], 2
            )

        # print([overhead_buffer[i] for i in kernel_list])
        if overhead_buffer[kernel_list[-1]] > max_oh:
            break

        fit_para["launch_oh"] = overhead_buffer

    print("Best overhead: " + str(best_para[kernel_list[0]]))
    return best_para


def fit_shrinkRatio(channel, fit_para, increment, min_sr, max_sr):

    fit_para["shrinkRatio"] = min_sr

    mse = np.inf
    best_para = None

    while True:

        print("Analyzing " + channel + " shrinkRatio: " + str(fit_para["shrinkRatio"]))

        error_sq = 0
        for filename in filenames:
            if channel == "DS":
                error_sq += get_error_DS(filename, fit_para)
            elif channel == "US":
                error_sq += get_error_US(filename, fit_para)

        if error_sq < mse:
            mse = error_sq
            best_para = deepcopy(fit_para["shrinkRatio"])

        # test next para
        shrinkRatio_buffer = fit_para["shrinkRatio"]

        shrinkRatio_buffer += increment
        shrinkRatio_buffer = np.round(shrinkRatio_buffer, 2)

        if shrinkRatio_buffer > max_sr:
            break

        fit_para["shrinkRatio"] = shrinkRatio_buffer

    print("Best shrinkRatio: " + str(best_para))
    return best_para


def get_error_DS(filename, fit_para):
    error = 0.0

    folder, fn = os.path.split(filename)

    ifile = open(filename, "r")
    data = json.load(ifile)
    ifile.close()
    if fn[0] == "r":
        prefix = None
    else:
        prefix = fn[0:3]

    DS = {}

    for key in data.keys():
        if key in uc_key and "PDSCH" in key:
            for case in usecases[key]:
                DS[case] = data[key][case]

    for case in DS.keys():

        if prefix is not None:
            filename = os.path.join(
                args.folder, prefix + "_sweep_streams_" + case + ".json"
            )
        else:
            filename = os.path.joi777n(args.folder, "sweep_streams_" + case + ".json")

        if args.is_graph and "F01" in case:
            filename = filename.replace("streams", "graphs")

        ifile = open(filename, "r")
        ref_lats = json.load(ifile)
        ifile.close()

        case_num = 0
        for cell_num in range(1, 37):
            if str(cell_num).zfill(2) in ref_lats:
                case_num += 1
                latency_dl, _, _, _ = schedule_2(
                    case,
                    cell_num,
                    DS[case],
                    folder,
                    prefix,
                    available_resources(args.arch),
                    None,
                    fit_para,
                )
                error += (latency_dl - np.mean(ref_lats[str(cell_num).zfill(2)])) ** 2

    return error / min(case_num, 36)


def get_error_US(filename, fit_para):
    error = 0.0

    folder, fn = os.path.split(filename)

    ifile = open(filename, "r")
    data = json.load(ifile)
    ifile.close()
    if fn[0] == "r":
        prefix = None
    else:
        prefix = fn[0:3]

    US = {}

    for key in data.keys():
        if key in uc_key and "PUSCH" in key:
            for case in usecases[key]:
                US[case] = data[key][case]

    for case in US.keys():

        if prefix is not None:
            filename = os.path.join(
                args.folder, prefix + "_sweep_streams_" + case + ".json"
            )
        else:
            filename = os.path.join(args.folder, "sweep_streams_" + case + ".json")

        if args.is_graph and "F01" in case:
            filename = filename.replace("streams", "graphs")

        ifile = open(filename, "r")
        ref_lats = json.load(ifile)
        ifile.close()

        case_num = 0
        for cell_num in range(1, 37):
            if str(cell_num).zfill(2) in ref_lats:
                case_num += 1
                latency_dl, _, _, _ = schedule_2(
                    case,
                    cell_num,
                    US[case],
                    folder,
                    prefix,
                    available_resources(args.arch),
                    None,
                    fit_para,
                )
                error += (latency_dl - np.mean(ref_lats[str(cell_num).zfill(2)])) ** 2

    return error / min(case_num, 36)


base = argparse.ArgumentParser()
base.add_argument(
    "--filename",
    type=str,
    nargs="+",
    dest="filename",
    help="Specifies the parse results file name",
    required=True,
)
base.add_argument(
    "--folder",
    type=str,
    dest="folder",
    help="Specifies the folder containing reference results",
    required=True,
)
base.add_argument(
    "--uc",
    type=str,
    nargs="+",
    dest="uc",
    help="Specifies the file contaning the use case config",
    required=True,
)
base.add_argument(
    "--arch",
    type=str,
    dest="arch",
    choices=supported_architectures(),
    default="gv100",
    help="Specifies the architecture to use for the extrapolation",
)
base.add_argument(
    "--graph",
    action="store_true",
    dest="is_graph",
    default=False,
    help="Specifies whether it is graph",
)
args = base.parse_args()

ifile = open("pipelines.json", "r")
kernels = json.load(ifile)
ifile.close()

usecases = {}
uc_key = []

for item in args.uc:
    ifile = open(item, "r")
    uc = json.load(ifile)
    ifile.close()

    for key in uc.keys():
        uc_key.append(key)
        usecases[key] = uc[key]

filenames = []

for item in args.filename:
    filenames.extend(glob.glob(item))

# initialize
min_oh = 1.0
max_oh = 35.0
increment_oh = 1
fit_para = {"DS": {"launch_oh": {}}, "US": {"launch_oh": {}}}
for kernel in kernels["DS"].keys():
    fit_para["DS"]["launch_oh"][kernel] = min_oh
for kernel in kernels["US"].keys():
    fit_para["US"]["launch_oh"][kernel] = min_oh

min_sr = 0.45
max_sr = 1.0
increment_sr = 0.03
fit_para["DS"]["shrinkRatio"] = 0.73
fit_para["US"]["shrinkRatio"] = 0.73

best_para = deepcopy(fit_para)

# start fitting

for it in range(1):  # fitting iterations
    fit_para = deepcopy(best_para)
    best_para["DS"]["launch_oh"] = fit_oh(
        "DS",
        fit_para["DS"],
        list(kernels["DS"].keys()),
        increment=increment_oh,
        min_oh=min_oh,
        max_oh=max_oh,
        homo_oh=True,
    )
    best_para["US"]["launch_oh"] = fit_oh(
        "US",
        fit_para["US"],
        list(kernels["US"].keys()),
        increment=increment_oh,
        min_oh=min_oh,
        max_oh=max_oh,
        homo_oh=True,
    )

    fit_para = deepcopy(best_para)
    best_para["DS"]["shrinkRatio"] = fit_shrinkRatio(
        "DS", fit_para["DS"], increment=increment_sr, min_sr=min_sr, max_sr=max_sr
    )
    best_para["US"]["shrinkRatio"] = fit_shrinkRatio(
        "US", fit_para["US"], increment=increment_sr, min_sr=min_sr, max_sr=max_sr
    )

ofile = open("fit_para.json", "w")
json.dump(best_para, ofile, indent=2)
