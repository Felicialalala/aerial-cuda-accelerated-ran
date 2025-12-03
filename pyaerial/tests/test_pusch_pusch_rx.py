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

"""Tests for pusch_rx.py."""
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.ldpc.util import random_tb
from aerial.phy5g.pusch import PuschRx
from aerial.util.fapi import dmrs_fapi_to_bit_array


def test_pusch_rx_init(pusch_rx):  # pylint: disable=unused-argument
    """Tests initialization and destruction of PuschRx."""
    # The fixture initializes and it gets destroyed when
    # it goes out of scope of this function.


# All cuPHY TVs. These are all that are used in cuPHY unit testing.
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
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7227,
                     7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7242,
                     7243, 7244, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256,
                     7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269,
                     7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282,
                     7283, 7284, 7285, 7286, 7287, 7288, 7289, 7290, 7291, 7294, 7295, 7296, 7297,
                     7298, 7299, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311,
                     7312, 7313, 7314, 7315, 7316, 7317, 7318, 7321, 7322, 7323, 7327, 7328, 7329,
                     7330, 7331, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7341, 7342, 7343, 7344,
                     7345, 7346, 7347, 7350, 7351, 7352, 7353, 7358, 7359, 7360, 7363, 7364, 7365,
                     7366, 7367, 7368, 7369, 7370, 7371, 7372, 7401, 7402, 7403, 7404, 7405, 7406,
                     7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419,
                     7420, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7432,
                     7444, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454, 7455, 7456,
                     7457, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469,
                     7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482,
                     7483, 7484, 7485, 7486, 7487, 7488, 7489, 7490, 7491, 7492, 7493, 7494, 7495,
                     7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7508,
                     7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7517, 7518, 7519, 7520, 7521,
                     7522, 7523, 7524, 7525, 7526, 7527, 7528, 7529, 7530, 7531, 7532, 7533, 7534,
                     7535, 7536, 7537, 7538, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547,
                     7548, 7549, 7550, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560,
                     7561, 7562, 7563, 7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573,
                     7574, 7575, 7576, 7577, 7578, 7579, 7580, 7581, 7582, 7583, 7584, 7585, 7586,
                     7601, 7602, 7603, 7604, 7605, 7606, 7607, 7608, 7609, 7610, 7611, 7612, 7613,
                     7622, 7702, 7703, 7704, 7705, 7706, 7707, 7708, 7709, 7901, 7902, 7905, 7911,
                     7912, 7913, 7914, 7915, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924,
                     7925, 7926, 7927, 7928, 7929, 7930, 7931, 7932, 7933]

# Multiple UE groups (FDM).
fdm_tc = [7233, 7234, 7235, 7239, 7240, 7246, 7247, 7248, 7249, 7250, 7259, 7260, 7264, 7265,
          7266, 7270, 7271, 7272, 7280, 7281, 7282, 7283, 7297, 7298, 7327, 7328, 7329, 7330,
          7338, 7340, 7344, 7349, 7470, 7481, 7483, 7490, 7500, 7933]

# Skipped TVs:

# DFT-S-OFDM not supported.
tc_to_skip = [7289, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7358, 7359, 7360, 7361, 7362, 7363,
              7364, 7471, 7485]
# UCI on PUSCH not supported.
tc_to_skip += list(np.arange(7501, 7587)) + list(np.arange(7915, 7934))
# Skip the invalid cfg cases and forceRxZero/low SNR cases.
tc_to_skip += list(np.arange(7702, 7712))
tc_to_skip += [7417, 7531, 7532]
# LBRM not supported yet.
tc_to_skip += [7321, 7322, 7323]

# Multiple UE groups (FDM) supported only by the PuschConfig-based API.
single_ue_grp_api_tc = list(set(test_case_numbers) - set(tc_to_skip) - set(fdm_tc))
single_ue_grp_api_tc = list(itertools.product(single_ue_grp_api_tc, [True, False]))
pusch_config_api_tc = list(set(test_case_numbers) - set(tc_to_skip))
pusch_config_api_tc = list(itertools.product(pusch_config_api_tc, [True, False]))


# pylint: disable=too-many-locals
@pytest.mark.parametrize(
    "test_case_number, h2d",
    single_ue_grp_api_tc,
    ids=[f"{test_case_number} - cuPy: {h2d}" for test_case_number, h2d in single_ue_grp_api_tc]
)
def test_pusch_rx(test_case_number, h2d):
    """Run test cases on PUSCH receiver."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    cell_id = input_file["gnb_pars"]["cellId"][0]
    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    num_tx_ant = 1
    num_prb_dl_bwp = 273
    num_prb_ul_bwp = 273
    mu = input_file["gnb_pars"]["mu"][0]
    enable_cfo_correction = input_file["gnb_pars"]["enableCfoCorrection"][0]
    enable_weighted_ave_cfo_est = input_file["gnb_pars"]["enableWeightedAverageCfo"][0]
    enable_to_estimation = input_file["gnb_pars"]["enableToEstimation"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]
    eq_coeff_algo_idx = input_file["gnb_pars"]["eqCoeffAlgoIdx"][0]

    pusch_rx = PuschRx(
        cell_id=cell_id,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        num_ul_bwp=num_prb_ul_bwp,
        num_dl_bwp=num_prb_dl_bwp,
        mu=mu,
        enable_cfo_correction=enable_cfo_correction,
        enable_weighted_ave_cfo_est=enable_weighted_ave_cfo_est,
        enable_to_estimation=enable_to_estimation,
        enable_pusch_tdi=enable_pusch_tdi,
        eq_coeff_algo=eq_coeff_algo_idx
    )

    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)
    num_ues = input_file["ueGrp_pars"]["nUes"][0]

    layers = input_file["tb_pars"]["numLayers"]
    rntis = input_file["tb_pars"]["nRnti"]
    slot = np.array(input_file["gnb_pars"]["slotNumber"])[0]

    start_prb = input_file["ueGrp_pars"]["startPrb"][0]
    num_prbs = input_file["ueGrp_pars"]["nPrb"][0]
    prg_size = input_file["ueGrp_pars"]["prgSize"][0]
    num_ul_streams = input_file["ueGrp_pars"]["nUplinkStreams"][0]
    start_sym = input_file["ueGrp_pars"]["StartSymbolIndex"][0]
    num_symbols = input_file["ueGrp_pars"]["NrOfSymbols"][0]
    dmrs_sym_loc_bmsk = input_file["ueGrp_pars"]["dmrsSymLocBmsk"][0]
    dmrs_syms = dmrs_fapi_to_bit_array(dmrs_sym_loc_bmsk)
    dmrs_scrm_id = input_file["tb_pars"]["dmrsScramId"][0]
    dmrs_max_len = input_file["tb_pars"]["dmrsMaxLength"][0]
    dmrs_add_ln_pos = input_file["tb_pars"]["dmrsAddlPosition"][0]
    num_dmrs_cdm_grps_no_data = input_file["tb_pars"]["numDmrsCdmGrpsNoData"][0]
    scids = input_file["tb_pars"]["nSCID"]
    dmrs_ports = input_file["tb_pars"]["dmrsPortBmsk"]
    data_scids = input_file["tb_pars"]["dataScramId"]
    mod_orders = input_file["tb_pars"]["qamModOrder"]
    code_rates = input_file["tb_pars"]["targetCodeRate"]
    tb_sizes = input_file["tb_pars"]["nTbByte"]
    rvs = input_file["tb_pars"]["rv"]
    ndis = input_file["tb_pars"]["ndi"]
    mcs_tables = input_file["tb_pars"]["mcsTableIndex"]
    mcs_indices = input_file["tb_pars"]["mcsIndex"]

    if h2d:
        rx_slot = cp.array(rx_slot, dtype=cp.complex64, order='F')

    tb_crcs = pusch_rx.run(
        rx_slot=rx_slot,
        num_ues=num_ues,
        slot=slot,
        num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
        dmrs_scrm_id=dmrs_scrm_id,
        start_prb=start_prb,
        num_prbs=num_prbs,
        prg_size=prg_size,
        num_ul_streams=num_ul_streams,
        dmrs_syms=dmrs_syms,
        dmrs_max_len=dmrs_max_len,
        dmrs_add_ln_pos=dmrs_add_ln_pos,
        start_sym=start_sym,
        num_symbols=num_symbols,
        scids=scids,
        layers=layers,
        dmrs_ports=dmrs_ports,
        rntis=rntis,
        data_scids=data_scids,
        mcs_tables=mcs_tables,
        mcs_indices=mcs_indices,
        code_rates=code_rates,
        mod_orders=mod_orders,
        tb_sizes=tb_sizes,
        rvs=rvs,
        ndis=ndis
    )[0]

    assert all(tb_crcs == 0)


@pytest.mark.parametrize(
    "test_case_number, h2d",
    pusch_config_api_tc,
    ids=[f"{test_case_number} - cuPy: {h2d}" for test_case_number, h2d in pusch_config_api_tc]
)
def test_pusch_rx_config_api(pusch_config, test_case_number, h2d):
    """Run test cases on PUSCH receiver."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    cell_id = input_file["gnb_pars"]["cellId"][0]
    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    num_tx_ant = 1
    num_prb_dl_bwp = 273
    num_prb_ul_bwp = 273
    mu = input_file["gnb_pars"]["mu"][0]
    enable_cfo_correction = input_file["gnb_pars"]["enableCfoCorrection"][0]
    enable_weighted_ave_cfo_est = input_file["gnb_pars"]["enableWeightedAverageCfo"][0]
    enable_to_estimation = input_file["gnb_pars"]["enableToEstimation"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]
    eq_coeff_algo_idx = input_file["gnb_pars"]["eqCoeffAlgoIdx"][0]
    slot = np.array(input_file["gnb_pars"]["slotNumber"])[0]

    pusch_rx = PuschRx(
        cell_id=cell_id,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        num_ul_bwp=num_prb_ul_bwp,
        num_dl_bwp=num_prb_dl_bwp,
        mu=mu,
        enable_cfo_correction=enable_cfo_correction,
        enable_weighted_ave_cfo_est=enable_weighted_ave_cfo_est,
        enable_to_estimation=enable_to_estimation,
        enable_pusch_tdi=enable_pusch_tdi,
        eq_coeff_algo=eq_coeff_algo_idx
    )

    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)

    pusch_configs = pusch_config(input_file)

    if h2d:
        rx_slot = cp.array(rx_slot, dtype=cp.complex64, order='F')

    tb_crcs = pusch_rx(
        slot=slot,
        rx_slot=rx_slot,
        config=pusch_configs
    )[0]

    assert all(tb_crcs == 0)


# PUSCH Tx to Rx test cases.
test_cases = [
    dict(
        rnti=[1000],
        num_ue=1,
        scid=[0],
        data_scid=[41],
        dmrs_position=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        layers=[1],
        qam=[6],
        coderate=[910],
        dmrs_ports=[1],
        slot=2,
        precoding_matrices=[np.array([[0.5 + 0.j, 0. + 0.5j, 0. + 0.5j, -0.5 + 0.j]])]
    ),
    dict(
        rnti=[1000, 1001],
        num_ue=2,
        scid=[0, 0],
        data_scid=[41, 42],
        dmrs_position=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        layers=[1, 1],
        qam=[6, 6],
        coderate=[910, 910],
        dmrs_ports=[1, 4],
        slot=7,
        precoding_matrices=[
            np.array([[0.5 + 0.j, 0. - 0.5j, 0. + 0.5j, 0.5 + 0.j]]),
            np.array([[0.5 + 0.j, 0. - 0.5j, 0. - 0.5j, -0.5 + 0.j]])
        ]
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_pusch_tx_to_rx(pdsch_tx, pusch_rx, test_case):
    """Test PUSCH Tx to Rx chain.

    In this test, PdschTx acts as the UE transmitter.
    """
    rnti = test_case["rnti"]
    num_ues = test_case["num_ue"]
    scid = test_case["scid"]
    data_scid = test_case["data_scid"]
    dmrs_position = test_case["dmrs_position"]
    layers = test_case["layers"]
    qam = test_case["qam"]
    coderate = test_case["coderate"]
    dmrs_ports = test_case["dmrs_ports"]
    slot = test_case["slot"]
    precoding_matrices = test_case["precoding_matrices"]

    tb_input = []
    for i in range(num_ues):
        tb_input_i = random_tb(
            mod_order=qam[i],
            code_rate=coderate[i],
            dmrs_syms=dmrs_position,
            num_prbs=273,
            start_sym=0,
            num_symbols=14,
            num_layers=layers[i]
        )
        tb_input.append(tb_input_i)

    # Transmit.
    xtf = pdsch_tx.run(
        tb_inputs=tb_input,
        num_ues=num_ues,
        slot=slot,
        start_prb=0,
        num_prbs=273,
        dmrs_syms=dmrs_position,
        start_sym=0,
        num_symbols=14,
        scids=scid,
        layers=layers,
        dmrs_ports=dmrs_ports,
        rntis=rnti,
        data_scids=data_scid,
        precoding_matrices=precoding_matrices,
        code_rates=[c * 10. for c in coderate],
        mod_orders=qam
    )

    # Receive.
    tb_crcs, tbs = pusch_rx.run(
        rx_slot=xtf,
        num_ues=num_ues,
        slot=slot,
        start_prb=0,
        num_prbs=273,
        prg_size=1,
        num_ul_streams=4,
        dmrs_syms=dmrs_position,
        dmrs_max_len=1,
        dmrs_add_ln_pos=1,
        start_sym=0,
        num_symbols=14,
        scids=scid,
        layers=layers,
        dmrs_ports=dmrs_ports,
        rntis=rnti,
        data_scids=data_scid,
        code_rates=[c * 10. for c in coderate],
        mod_orders=qam,
        tb_sizes=[len(tb) for tb in tb_input]
    )

    assert all(tb_crcs == 0)
    for ue_idx in range(num_ues):
        assert all(tbs[ue_idx] == tb_input[ue_idx])
