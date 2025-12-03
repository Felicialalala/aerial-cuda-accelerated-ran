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

"""Tests for csirs_tx.py."""
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.csirs import CsiRsTx
from aerial.phy5g.csirs import CsiRsConfig
from aerial.phy5g.csirs import CsiRsTxConfig

# Test vector numbers.
test_case_numbers = list(range(4001, 4066))
test_case_numbers += [4101, 4102, 4103]
test_case_numbers += list(range(4201, 4223))
test_case_numbers += list(range(4801, 4808))
test_case_numbers += list(range(4901, 4906))

# pylint: disable=too-many-locals
all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, cupy",
    all_cases,
    ids=[f"{test_case_number} - cuPy: {cupy}" for test_case_number, cupy in all_cases]
)
def test_csirs_tx_run(test_case_number, cupy):
    """Test running CSI-RS Tx against Aerial test vectors."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_CSIRS_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    csirs_prms_list = input_file["CsirsParamsList"]
    csirs_configs = []
    for csirs_prms in csirs_prms_list:
        csirs_configs.append(CsiRsConfig(
            start_prb=np.uint16(csirs_prms["StartRB"]),
            num_prb=np.uint16(csirs_prms["NrOfRBs"]),
            freq_alloc=list(map(int, format(csirs_prms["FreqDomain"], "016b"))),
            row=np.uint8(csirs_prms["Row"]),
            symb_L0=np.uint8(csirs_prms["SymbL0"]),
            symb_L1=np.uint8(csirs_prms["SymbL1"]),
            freq_density=np.uint8(csirs_prms["FreqDensity"]),
            scramb_id=np.uint16(csirs_prms["ScrambId"]),
            idx_slot_in_frame=np.uint8(csirs_prms["idxSlotInFrame"]),
            cdm_type=np.uint8(csirs_prms["CDMType"]),
            beta=csirs_prms["beta"],
            enable_precoding=np.uint8(csirs_prms["enablePrcdBf"]),
            precoding_matrix_index=np.uint16(0)
        ))

    csirs_pmw = np.array(input_file["Csirs_PM_W0"])
    csirs_pmw = csirs_pmw["re"] + 1j * csirs_pmw["im"]
    if csirs_pmw.size:
        precoding_matrices = [csirs_pmw]
    else:
        precoding_matrices = None

    csirs_tx_config = CsiRsTxConfig(csirs_configs=[csirs_configs],
                                    precoding_matrices=precoding_matrices)

    ref_tx_buffer = np.array(input_file["X_tf"])["re"] + 1j * np.array(input_file["X_tf"])["im"]
    ref_tx_buffer = np.ascontiguousarray(ref_tx_buffer.T)
    num_ant_dl = [1]
    if len(ref_tx_buffer.shape) == 3:
        num_ant_dl = [ref_tx_buffer.shape[2]]

    csirs_tx = CsiRsTx(num_prb_dl_bwp=[ref_tx_buffer.shape[0] // 12], num_ant_dl=num_ant_dl)

    if cupy:
        tx_buffer = cp.zeros(ref_tx_buffer.shape, dtype=cp.complex64)
    else:
        tx_buffer = np.zeros(ref_tx_buffer.shape, dtype=np.complex64)

    tx_buffer = csirs_tx(
        config=csirs_tx_config,
        tx_buffers=[tx_buffer]
    )[0]

    if cupy:
        tx_buffer = tx_buffer.get()

    assert np.allclose(tx_buffer, ref_tx_buffer, rtol=0.001)
