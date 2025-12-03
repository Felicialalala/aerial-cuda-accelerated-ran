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

"""Test ChannelEqualizer."""
# Ensure that all the test vectors are available in TEST_VECTOR_DIR.
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import ChannelEqualizer


test_case_numbers = [7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213,
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7227,
                     7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7242,
                     7243, 7244, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256,
                     7257, 7258, 7259, 7260, 7261]

tc_to_skip = [7236, 7237, 7238, 7239, 7240, 7246, 7247, 7257, 7261]

test_case_numbers = set(list(test_case_numbers)) - set(tc_to_skip)
all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, h2d",
    all_cases,
    ids=[f"{test_case_number} - cuPy: {h2d}" for test_case_number, h2d in all_cases]
)
def test_channel_equalizer(pusch_config, cuda_stream, test_case_number, h2d):
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)

    ch_est = []
    lw_inv = []
    for ue_grp_idx in range(len(input_file["ueGrp_pars"])):

        ch_est_ = np.array(input_file[f"reference_H_est{ue_grp_idx}"])["re"] + 1j * \
            np.array(input_file[f"reference_H_est{ue_grp_idx}"])["im"]
        if ch_est_.ndim == 3:
            ch_est_ = ch_est_[..., None].transpose(2, 1, 0, 3)
        else:
            ch_est_ = ch_est_.transpose(3, 2, 1, 0)
        ch_est.append(ch_est_)

        try:
            noise_cov = np.array(input_file[f"reference_nCov{ue_grp_idx}"]["re"]) + 1j * \
                np.array(input_file[f"reference_nCov{ue_grp_idx}"]["im"])
        except ValueError:
            noise_cov = np.array(input_file[f"reference_nCov{ue_grp_idx}"])

        if noise_cov.ndim == 4:
            noise_cov = np.mean(noise_cov, axis=0)
        lw_inv_ = np.linalg.inv(np.linalg.cholesky(noise_cov))
        if lw_inv_.ndim == 3:
            lw_inv_ = lw_inv_.transpose(1, 2, 0)
        lw_inv.append(lw_inv_)

    noise_var_pre_eq = np.array(input_file["reference_noiseVardBPerUe"])

    pusch_configs = pusch_config(input_file)

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]
    eq_coeff_algo = input_file["gnb_pars"]["eqCoeffAlgoIdx"][0]
    num_layers = input_file["tb_pars"]["numLayers"]
    mod_orders = input_file["tb_pars"]["qamModOrder"]

    if h2d:
        rx_slot = cp.array(rx_slot, dtype=cp.complex64, order='F')
        ch_est = [cp.array(elem, dtype=cp.complex64, order='F') for elem in ch_est]
        lw_inv = [cp.array(elem, dtype=cp.complex64, order='F') for elem in lw_inv]
        noise_var_pre_eq = cp.array(noise_var_pre_eq, dtype=cp.float32, order='F')

    channel_equalizer = ChannelEqualizer(
        num_rx_ant=num_rx_ant,
        eq_coeff_algo=eq_coeff_algo,
        enable_pusch_tdi=enable_pusch_tdi,
        cuda_stream=cuda_stream
    )

    llr, eq_sym = channel_equalizer.equalize(
        rx_slot=rx_slot,
        channel_est=ch_est,
        lw_inv=lw_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=pusch_configs
    )

    if h2d:
        llr = [elem.get(order='F') for elem in llr]
        eq_sym = [elem.get(order='F') for elem in eq_sym]

    # Check all UE groups.
    for ue_grp_idx in range(len(llr)):

        # Check equalized symbols.
        ref_eq_sym = np.array(input_file[f"reference_X_est{ue_grp_idx}"])
        ref_eq_sym = ref_eq_sym["re"] + 1j * ref_eq_sym["im"]
        if ref_eq_sym.ndim == 2:
            ref_eq_sym = ref_eq_sym.transpose()[None]
        else:
            ref_eq_sym = ref_eq_sym.transpose(0, 2, 1)

        # assert np.allclose(eq_sym[ue_grp_idx], ref_eq_sym, rtol=5e-2) #TODO

        # Check LLRs. Just check the sign as the magnitude can be quite different.
        ref_llr = np.array(input_file[f"reference_eqOutLLRs{ue_grp_idx}"])
        ref_llr = ref_llr.transpose(3, 2, 1, 0)

        assert np.all(
            np.sign(llr[ue_grp_idx][:mod_orders[ue_grp_idx], :num_layers[ue_grp_idx], ...]) ==
            np.sign(ref_llr[:mod_orders[ue_grp_idx], ...])
        )
