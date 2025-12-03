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

def parse(args, lines):

    pdsch_crc_errors = 0
    pdsch_ldpc_errors = 0
    pdsch_fused_errors = 0
    pdsch_symbols_errors = 0

    pusch_bler_error = 0

    pdsch_processed_lines = 0
    pusch_processed_lines = 0

    for line in lines:

        lst = line.split()

        if len(lst) > 0:

            if lst[0] == "CRC" and lst[1] == "Error" and lst[2] == "Count:":
                pdsch_crc_errors += int(lst[3].replace(";", ""))
                pdsch_processed_lines += 1
                continue

            if lst[0] == "LDPC" and lst[1] == "Error" and lst[2] == "Count:":
                pdsch_ldpc_errors += int(lst[3].replace(";", ""))
                pdsch_processed_lines += 1
                continue

            if (
                lst[0] == "Fused"
                and lst[1] == "Rate"
                and lst[2] == "Matching"
                and lst[3] == "and"
                and lst[4] == "Modulation"
                and lst[5] == "Mapper:"
            ):
                pdsch_fused_errors += int(lst[7])
                pdsch_processed_lines += 1
                continue

            if (
                lst[0] == "PDSCH:"
                and lst[1] == "Found"
                and lst[3] == "mismatched"
                and lst[4] == "symbols"
            ):
                pdsch_symbols_errors += int(lst[2])
                pdsch_processed_lines += 1
                continue

            if (
                lst[0] == "Cell"
                and lst[1] == "#"
                and lst[3] == ":"
                and lst[6] == "Metric"
                and lst[8] == "Block"
                and lst[9] == "Error"
                and lst[10] == "Rate"
            ):
                pusch_bler_error += float(lst[12])
                pusch_processed_lines += 1
                continue

    results = {}
    results["PDSCH"] = {
        "CRC errors": pdsch_crc_errors,
        "LDPC encoding errors": pdsch_ldpc_errors,
        "Rate matching errors": pdsch_fused_errors,
        "Symbols errors": pdsch_symbols_errors,
        "Lines parsed": pdsch_processed_lines,
    }

    results["PUSCH"] = {"BLER": pusch_bler_error, "Line parsed": pusch_processed_lines}

    return results
