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


def check(args, pusch, pusch_subslotProc, pdsch, dlbfw, ulbfw, srs1, srs2, rach, pdcch, pucch, ssb, csirs):

    pdsch_slots = 4

    if len(pdsch) < pdsch_slots:
        return False

    for idx, itm in enumerate(pdsch):

        if np.abs(itm[0] - itm[2] * 500) > 50:
            return False
        
    for idx, itm in enumerate(pdcch):

        if np.abs(itm[0] - itm[2] * 500) > 50:
            return False

    return True


def unpack(args, pusch, pusch_subslotProc, pdsch, dlbfw, ulbfw, srs1, srs2, rach, pdcch, pucch, ssb, csirs):

    new_pdsch = []
    new_pdcch = []
    new_csirs = []
    new_dlbfw = []
    new_ssb   = []

    for idx, itm in enumerate(pdsch):
        new_pdsch.append(itm[1] - itm[0])

    for idx, itm in enumerate(pdcch):
        new_pdcch.append(itm - pdsch[idx][0])

    for idx, itm in enumerate(csirs):
        new_csirs.append(itm - pdcch[idx])

    for idx, itm in enumerate(ssb):
        new_ssb.append(itm[1] - itm[0])
                
    if args.is_rec_bf:
        for idx, itm in enumerate(dlbfw):
            new_dlbfw.append(itm[1] - itm[0])

    return (
        pusch,
        pusch_subslotProc,
        new_pdsch,
        new_dlbfw,
        ulbfw,
        srs1,
        srs2,
        rach,
        new_pdcch,
        pucch,
        new_ssb,
        new_csirs,
    )
