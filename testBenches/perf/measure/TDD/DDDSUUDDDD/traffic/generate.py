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
import yaml
import json
import re

from ....analyze import extract


def run(args, vectors, testcases, filenames, sLotConfig):

    (
        testcases_dl,
        testcases_ul,
        testcases_dlbf,
        testcases_ulbf,
        testcases_sr,
        testcases_ra,
        testcases_cdl,
        testcases_cul,
        testcases_ssb,
        testcases_cr,
        testcases_mac,
        testcases_mac2,
    ) = testcases
    (
        filenames_dl,
        filenames_ul,
        filenames_dlbf,
        filenames_ulbf,
        filenames_sr,
        filenames_ra,
        filenames_cdl,
        filenames_cul,
        filenames_ssb,
        filenames_cr,
        filenames_mac,
        filenames_mac2,
    ) = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = len(testcases_dl)

    ifile = open("measure/TDD/priorities.json", "r")
    priorities = json.load(ifile)
    ifile.close()

    buffer = {}

    for key in priorities.keys():
        label = "_".join([key, "PRIO"])
        buffer[label] = priorities[key]

    payload.update(buffer)

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}

        if not args.is_no_pdsch and sLotConfig["PDSCH"][sweep_idx % args.pattern_len]:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcase])
                for testcase in testcases_dl
            ]

        if testcases_cdl is not None and sLotConfig["PDCCH"][sweep_idx % args.pattern_len]:
            channel["PDCCH"] = [
                os.path.join(args.vfld, filenames_cdl[testcase])
                for testcase in testcases_cdl
            ]

        if testcases_cr is not None and sLotConfig["CSIRS"][sweep_idx % args.pattern_len]:
            channel["CSIRS"] = [
                os.path.join(args.vfld, filenames_cr[testcase])
                for testcase in testcases_cr
            ]

        if testcases_dlbf is not None and sLotConfig["PDSCH"][sweep_idx % args.pattern_len]:
            channel["DLBFW"] = [
                os.path.join(args.vfld, filenames_dlbf[testcase])
                for testcase in testcases_dlbf
            ]

        if testcases_ssb is not None and sLotConfig["PBCH"][sweep_idx % args.pattern_len]:
            channel["SSB"] = [
                os.path.join(args.vfld, filenames_ssb[testcase])
                for testcase in testcases_ssb
            ]

        if testcases_mac is not None and sLotConfig["MAC"][sweep_idx % args.pattern_len]:
            for testcase in testcases_mac:
                # Get the cumac TV name based on actual cell count
                # Example: in the input json config file, cumac TV defined as "TV_cumac_F08-MC-CC-8PC.h5". For test of 16 cells, we need to replace the "8" with 16, i.e., "TV_cumac_F08-MC-CC-16PC.h5"
                
                # Search for the default cell count in the TV name, using pattern '-(\d+)PC'
                macTvName = filenames_mac[testcase]
                matches = list(re.finditer(r'-(\d+)PC', macTvName))

                # If a cell count match is found, replace it with actual cell count
                if matches:
                    # Get the last match of '-(\d+)PC'
                    last_match = matches[-1]
                    number = int(last_match.group(1))
                    
                    # Replace the default cell count in TV name with actual cell count
                    macTvName = macTvName[:last_match.start(1)] + str(payload["cells"]) + macTvName[last_match.end(1):]
                    channel["MAC"] = [os.path.join(args.vfld, macTvName)]
                else:
                    raise ValueError("No match of cell count in cuMAC TV name")

        if testcases_mac2 is not None and args.mac2 > 0 and sLotConfig["MAC2"][sweep_idx % args.pattern_len]:
            for testcase in testcases_mac2:
                # Get the cumac TV name based on actual cell count
                # Example: in the input json config file, cumac TV defined as "TV_cumac_F08-MC-CC-8PC.h5". For test of 16 cells, we need to replace the "8" with 16, i.e., "TV_cumac_F08-MC-CC-16PC.h5"

                # Search for the default cell count in the TV name, using pattern '-(\d+)PC'
                mac2TvName = filenames_mac2[testcase]
                matches = list(re.finditer(r'-(\d+)PC', mac2TvName))

                # If a cell count match is found, replace it with actual cell count
                if matches:
                    # Get the last match of '-(\d+)PC'
                    last_match = matches[-1]
                    number = int(last_match.group(1))

                    # Replace the default cell count in TV name with actual cell count
                    mac2TvName = mac2TvName[:last_match.start(1)] + str(args.mac2) + mac2TvName[last_match.end(1):] # mac2 using fixed number of cells
                    channel["MAC2"] = [os.path.join(args.vfld, mac2TvName)]
                else:
                    raise ValueError("No match of cell count in cuMAC2 TV name")

        if sweep_idx % args.pattern_len == 0:

            if not args.is_no_pusch:

                channel["PUSCH"] = [
                    os.path.join(args.vfld, filenames_ul[testcase])
                    for testcase in testcases_ul
                ]

            if testcases_ulbf is not None:
                channel["ULBFW"] = [
                    os.path.join(args.vfld, filenames_ulbf[testcase])
                    for testcase in testcases_ulbf
            ]

            if testcases_cul is not None:
                channel["PUCCH"] = [
                    os.path.join(args.vfld, filenames_cul[testcase])
                    for testcase in testcases_cul
                ]

            if testcases_ra is not None:
                channel["PRACH"] = [
                    os.path.join(args.vfld, filenames_ra[testcase])
                    for testcase in testcases_ra
                ]

            if testcases_sr is not None:
                channel["SRS"] = [
                    os.path.join(args.vfld, filenames_sr[testcase])
                    for testcase in testcases_sr
                ]

        if sweep_idx % args.pattern_len == 1:

            if not args.is_no_pusch:

                channel["PUSCH"] = [
                    os.path.join(args.vfld, filenames_ul[testcase])
                    for testcase in testcases_ul
                ]

            if testcases_ulbf is not None:
                channel["ULBFW"] = [
                    os.path.join(args.vfld, filenames_ulbf[testcase])
                    for testcase in testcases_ulbf
            ]

            if testcases_cul is not None:
                channel["PUCCH"] = [
                    os.path.join(args.vfld, filenames_cul[testcase])
                    for testcase in testcases_cul
                ]
                
        channels.append(channel)

    payload["slots"] = channels
    payload["parameters"] = extract(args, channels)

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile, sort_keys=False)
    ofile.close()
