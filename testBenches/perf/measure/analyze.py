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

import h5py
import numpy as np


def extract(args, channels):

    pdsch = []
    pusch = []

    for slot in channels:

        buffer = slot.get("PDSCH", None)

        if buffer is not None:
            pdsch.append(buffer)

        buffer = slot.get("PUSCH", None)

        if buffer is not None:
            pusch.append(buffer)

    pdsch_flattened = []

    for slot in pdsch:
        pdsch_flattened.extend(slot)

    pdsch_flattened = list(set(pdsch_flattened))

    pusch_flattened = []

    for slot in pusch:
        pusch_flattened.extend(slot)

    pusch_flattened = list(set(pusch_flattened))

    pdsch_cell_params = {}

    pdsch_max_ntb_per_cell = 0
    pdsch_max_ncb_per_cell = 0
    pdsch_max_ncb_per_tb = 0
    pdsch_max_prb_per_cell = 0
    pdsch_max_ntx_per_cell = 0

    for vector in pdsch_flattened:

        ifile = h5py.File(vector)

        num_tb = int(ifile["cw_pars"].shape[0])

        num_cb = 0
        max_cb = np.zeros(num_tb, dtype=int)

        for k in range(num_tb):
            max_cb[k] = int(ifile[f"tb{k}_cbs"].shape[0])
            num_cb += int(ifile[f"tb{k}_cbs"].shape[0])

        max_cb_per_tb = int(np.max(max_cb))

        num_prb = int(ifile["cellStat_pars"]["nPrbDlBwp"][0])
        num_ntx = int(ifile["cellStat_pars"]["nTxAnt"][0])

        pdsch_cell_params[vector] = {}
        pdsch_cell_params[vector] = (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb)

        if pdsch_max_ntb_per_cell < num_tb:
            pdsch_max_ntb_per_cell = num_tb

        if pdsch_max_ncb_per_cell < num_cb:
            pdsch_max_ncb_per_cell = num_cb

        if pdsch_max_ncb_per_tb < max_cb_per_tb:
            pdsch_max_ncb_per_tb = max_cb_per_tb

        if pdsch_max_prb_per_cell < num_prb:
            pdsch_max_prb_per_cell = num_prb

        if pdsch_max_ntx_per_cell < num_ntx:
            pdsch_max_ntx_per_cell = num_ntx

    pusch_cell_params = {}

    pusch_max_ntb_per_cell = 0
    pusch_max_ncb_per_cell = 0
    pusch_max_ncb_per_tb = 0
    pusch_max_prb_per_cell = 0
    pusch_max_ntx_per_cell = 0

    for vector in pusch_flattened:

        ifile = h5py.File(vector)

        num_tb = int(ifile["tb_pars"].shape[0])

        num_cb = 0
        max_cb = np.zeros(num_tb, dtype=int)

        for k in range(num_tb):
            max_cb[k] = int(ifile["tb_pars"][k]["nCb"])
            num_cb += int(ifile["tb_pars"][k]["nCb"])

        max_cb_per_tb = int(np.max(max_cb))

        num_prb = int(ifile["gnb_pars"]["nPrb"][0])
        num_ntx = int(ifile["gnb_pars"]["nRx"][0])

        pusch_cell_params[vector] = {}
        pusch_cell_params[vector] = (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb)

        if pusch_max_ntb_per_cell < num_tb:
            pusch_max_ntb_per_cell = num_tb

        if pusch_max_ncb_per_cell < num_cb:
            pusch_max_ncb_per_cell = num_cb

        if pusch_max_ncb_per_tb < max_cb_per_tb:
            pusch_max_ncb_per_tb = max_cb_per_tb

        if pusch_max_prb_per_cell < num_prb:
            pusch_max_prb_per_cell = num_prb

        if pusch_max_ntx_per_cell < num_ntx:
            pusch_max_ntx_per_cell = num_ntx

    pdsch_max_ncb_per_slot = 0
    pdsch_max_ntb_per_slot = 0

    for slot in pdsch:

        cell_num_tb = 0
        cell_num_cb = 0

        for cell in slot:

            (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb) = pdsch_cell_params[cell]

            cell_num_tb += num_tb
            cell_num_cb += num_cb

        if pdsch_max_ncb_per_slot < cell_num_cb:
            pdsch_max_ncb_per_slot = cell_num_cb

        if pdsch_max_ntb_per_slot < cell_num_tb:
            pdsch_max_ntb_per_slot = cell_num_tb

    pusch_max_ncb_per_slot = 0
    pusch_max_ntb_per_slot = 0

    for slot in pusch:

        cell_num_tb = 0
        cell_num_cb = 0

        for cell in slot:

            (num_tb, num_cb, max_cb_per_tb, num_ntx, num_prb) = pusch_cell_params[cell]

            cell_num_tb += num_tb
            cell_num_cb += num_cb

        if pusch_max_ncb_per_slot < cell_num_cb:
            pusch_max_ncb_per_slot = cell_num_cb

        if pusch_max_ntb_per_slot < cell_num_tb:
            pusch_max_ntb_per_slot = cell_num_tb

    results = {}

    results["PDSCH"] = {}
    results["PDSCH"]["Max #TB per slot"] = pdsch_max_ntb_per_slot
    results["PDSCH"]["Max #CB per slot"] = pdsch_max_ncb_per_slot
    results["PDSCH"]["Max #TB per slot per cell"] = pdsch_max_ntb_per_cell
    results["PDSCH"]["Max #CB per slot per cell"] = pdsch_max_ncb_per_cell
    results["PDSCH"]["Max #CB per slot per cell per TB"] = pdsch_max_ncb_per_tb
    results["PDSCH"]["Max #TX per cell"] = pdsch_max_ntx_per_cell
    results["PDSCH"]["Max #PRB per cell"] = pdsch_max_prb_per_cell

    results["PUSCH"] = {}
    results["PUSCH"]["Max #TB per slot"] = pusch_max_ntb_per_slot
    results["PUSCH"]["Max #CB per slot"] = pusch_max_ncb_per_slot
    results["PUSCH"]["Max #TB per slot per cell"] = pusch_max_ntb_per_cell
    results["PUSCH"]["Max #CB per slot per cell"] = pusch_max_ncb_per_cell
    results["PUSCH"]["Max #CB per slot per cell per TB"] = pusch_max_ncb_per_tb
    results["PUSCH"]["Max #RX per cell"] = pusch_max_ntx_per_cell
    results["PUSCH"]["Max #PRB per cell"] = pusch_max_prb_per_cell
    results["PUSCH"]["PUSCH subslot proc flag"] = int(args.pusch_subslot_proc[0])

    if args.pattern != "dddsu":
        if len(args.pusch_subslot_proc) > 1:
            results["PUSCH"]["PUSCH2 subslot proc flag"] = int(args.pusch_subslot_proc[1])
        else:
            results["PUSCH"]["PUSCH2 subslot proc flag"] = 0 # disable subslot processing for PUSCH2 if not specified

    return results
