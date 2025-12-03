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

import argparse
import json

base = argparse.ArgumentParser()
base.add_argument(
    "--first",
    type=str,
    dest="priority",
    choices=["pdsch", "prach"],
    help="Specifies which channel has the highest priority",
    required=True,
)
args = base.parse_args()

standard = [
    ["pdsch", "pdcch", "csirs"],
    ["pucch", "pucch2"],
    ["pusch", "pusch2"],
    ["srs", "ssb"],
    ["prach"],
]
alternative = [
    ["prach"],
    ["pdsch", "pdcch"],
    ["pucch", "pucch2"],
    ["pusch", "pusch2"],
    ["srs", "ssb"],
]

if args.priority == "pdsch":
    choice = standard
elif args.priority == "prach":
    choice = alternative
else:
    raise NotImplementedError

buffer = {}

for idx, itm in enumerate(choice):

    for sub_itm in itm:

        buffer[sub_itm.upper()] = idx

ofile = open("measure/TDD/priorities.json", "w")
json.dump(buffer, ofile, indent=2)
ofile.close()
