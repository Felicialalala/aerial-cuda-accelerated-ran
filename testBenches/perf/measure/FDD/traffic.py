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
import os
import yaml

from ..analyze import extract


def traffic_het(args, vectors, k, testcases, filenames):

    testcases_dl, testcases_ul = testcases
    filenames_dl, filenames_ul = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = k

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}
        tidxs_dl = np.random.randint(0, len(testcases_dl), k)
        tidxs_ul = np.random.randint(0, len(testcases_ul), k)

        if not args.is_no_pdsch:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcases_dl[tidx_dl]])
                for tidx_dl in tidxs_dl
            ]
        if not args.is_no_pusch:
            channel["PUSCH"] = [
                os.path.join(args.vfld, filenames_ul[testcases_ul[tidx_ul]])
                for tidx_ul in tidxs_ul
            ]

        channels.append(channel)

    payload["slots"] = channels
    payload["parameters"] = extract(args, channels)

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile, sort_keys=False)
    ofile.close()


def traffic_avg(args, vectors, testcases, filenames):

    testcases_dl, testcases_ul = testcases
    filenames_dl, filenames_ul = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = len(testcases_dl)

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}

        if not args.is_no_pdsch:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcase])
                for testcase in testcases_dl
            ]

        if not args.is_no_pusch:
            channel["PUSCH"] = [
                os.path.join(args.vfld, filenames_ul[testcase])
                for testcase in testcases_ul
            ]

        channels.append(channel)

    payload["slots"] = channels

    payload["parameters"] = extract(args, channels)

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile, sort_keys=False)
    ofile.close()
