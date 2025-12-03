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

"""Test CrcChecker."""
# Ensure that all the test vectors are available in TEST_VECTOR_DIR.
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.ldpc.crc_check import CrcChecker

test_case_numbers = [7018, 7019, 7020, 7021, 7022, 7023, 7026, 7027, 7028, 7029, 7030, 7031, 7033,
                     7034, 7035, 7036, 7037, 7038, 7040, 7041, 7042, 7043, 7044, 7046, 7047, 7048,
                     7049, 7050, 7051, 7052, 7053, 7054, 7055, 7056, 7058, 7059, 7060, 7061, 7062,
                     7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7075,
                     7076, 7077, 7078, 7079, 7080, 7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088,
                     7089, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098, 7099, 7100, 7101,
                     7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114,
                     7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127,
                     7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7140,
                     7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7151, 7152, 7153,
                     7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213,
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7225, 7227, 7229,
                     7230, 7231, 7232, 7233, 7236, 7242, 7249, 7251, 7252, 7253, 7254, 7255, 7256,
                     7258, 7259, 7261, 7301, 7302, 7303, 7304, 7305, 7306, 7346, 7347]

all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, h2d",
    all_cases,
    ids=[f"{test_case_number} - cuPy: {h2d}" for test_case_number, h2d in all_cases]
)
def test_ldpc_crc_checker(pusch_config, cuda_stream, test_case_number, h2d):
    """Test CRC checking functionality."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    pusch_configs = pusch_config(input_file)

    tb_pars = np.array(input_file["tb_pars"])
    num_ues = len(tb_pars)
    input_bits = []
    for ue in range(num_ues):
        code_blocks = np.transpose(np.array(input_file[f"reference_TbCbs_est{ue}"]))
        input_bits.append(code_blocks)

    crc_checker = CrcChecker(cuda_stream=cuda_stream)

    if h2d:
        input_bits = [cp.array(elem, order='F') for elem in input_bits]

    tb_payloads, tb_crcs = crc_checker.check_crc(
        input_bits=input_bits,
        pusch_configs=pusch_configs
    )

    if h2d:
        tb_payloads = [elem.get(order='F') for elem in tb_payloads]
        tb_crcs = [elem.get(order='F') for elem in tb_crcs]

    ref_tb_payloads = np.array(input_file["tb_payload"])
    offset = 0
    for ue in range(num_ues):
        tbs = tb_pars["nTbByte"][ue]
        assert np.array_equal(tb_payloads[ue], ref_tb_payloads[offset:offset + tbs, 0])
        offset += tbs

        # 4-byte alignment in the reference payload.
        offset = (offset + 3) // 4 * 4

        assert tb_crcs[ue] == 0
