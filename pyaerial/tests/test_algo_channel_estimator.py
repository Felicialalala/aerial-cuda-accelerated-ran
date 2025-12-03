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

"""Tests for channel_estimator.py."""
import glob
from itertools import product
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import ChannelEstimator


# Full channel estimate test cases.
full_ch_est_tc_numbers = [7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212,
                          7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224,
                          7225, 7227, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238,
                          7239, 7240, 7242, 7243, 7244, 7246, 7247, 7248, 7249, 7250, 7251, 7252,
                          7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261]

# LS channel estimate test cases.
# DFT-S-OFDM not supported yet.
tc_to_skip = [7471]
ls_est_tc_numbers = set(list(range(7445, 7476))) - set(tc_to_skip)

all_tcs = list(product(full_ch_est_tc_numbers, [1], [True, False])) + \
    list(product(ls_est_tc_numbers, [3], [True, False]))


@pytest.mark.parametrize(
    "test_case_number, ch_est_algo, h2d",
    all_tcs,
    ids=[f"{test_case_number} - algo{algo} - cuPy: {h2d}"
         for test_case_number, algo, h2d in all_tcs]
)
def test_channel_estimator(pusch_config, test_case_number, ch_est_algo, h2d):
    """Test channel estimator against Aerial test vectors."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    slot = np.array(input_file["gnb_pars"]["slotNumber"])[0]

    pusch_configs = pusch_config(input_file)

    channel_estimator = ChannelEstimator(
        num_rx_ant=num_rx_ant,
        ch_est_algo=ch_est_algo,
        chest_filter_h5=filename
    )

    if h2d:
        rx_slot = cp.array(rx_slot, order='F')

    ch_est = channel_estimator.estimate(
        rx_slot=rx_slot,
        slot=slot,
        pusch_configs=pusch_configs
    )

    if h2d:
        ch_est = [ch.get(order='F') for ch in ch_est]

    # Check all UE groups.
    for ue_grp_idx in range(len(ch_est)):

        # Reference depends on the used algorithm.
        if ch_est_algo in [0, 1]:
            ref_ch_est = np.array(input_file[f"reference_H_est{ue_grp_idx}"])["re"] + 1j * \
                np.array(input_file[f"reference_H_est{ue_grp_idx}"])["im"]

        elif ch_est_algo == 3:
            ref_ch_est = \
                np.array(
                    input_file[f"reference_ChEst_w_delay_est_H_LS_est_save{ue_grp_idx}"]
                )["re"] + \
                1j * np.array(
                    input_file[f"reference_ChEst_w_delay_est_H_LS_est_save{ue_grp_idx}"]
                )["im"]

        else:
            assert False, "Invalid channel estimation algorithm!"

        if ref_ch_est.ndim == 3:
            ref_ch_est = ref_ch_est[..., None].transpose(2, 1, 0, 3)
        else:
            ref_ch_est = ref_ch_est.transpose(3, 2, 1, 0)

        assert np.allclose(ch_est[ue_grp_idx], ref_ch_est, atol=0.05)
