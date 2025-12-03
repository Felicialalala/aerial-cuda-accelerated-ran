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

"""Tests for srs_rx.py."""
import glob

import pytest
from pytest import TEST_VECTOR_DIR

import h5py
import numpy as np

from aerial.pycuphy import chest_filters
from aerial.phy5g.srs import SrsRx
from aerial.phy5g.srs import SrsRxConfig
from aerial.phy5g.srs import SrsConfig
from aerial.phy5g.srs import SrsRxCellConfig
from aerial.phy5g.srs import SrsRxUeConfig


test_case_numbers = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013,
                     8014, 8015, 8016, 8017, 8018, 8019, 8020, 8021, 8022, 8023, 8024, 8026, 8027,
                     8028, 8029, 8030, 8031, 8032, 8033, 8035, 8051, 8052, 8053, 8054, 8055, 8056,
                     8057, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112,
                     8113, 8114, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122, 8123, 8124, 8125,
                     8126, 8127, 8128, 8129, 8130, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138,
                     8139, 8140, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149, 8150, 8151,
                     8152, 8153, 8154, 8155, 8156, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8164,
                     8201, 8202, 8203, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8212, 8213,
                     8222, 8223, 8224, 8225, 8226, 8227, 8301, 8302, 8401, 8402, 8403, 8404, 8405,
                     8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414, 8415, 8420, 8421, 8801,
                     8802]

tc_to_skip = [8226]
test_case_numbers = list(set(list(test_case_numbers)) - set(tc_to_skip))
test_case_numbers.sort()


@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_srs_rx(cuda_stream, test_case_number):
    # pylint: disable=too-many-locals
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_SRS_gNB_CUPHY_s*.h5")[0]
    try:
        h5 = h5py.File(filename, "r")  # pylint: disable=invalid-name
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    chest_params = chest_filters.srs_chest_params_from_hdf5(filename, debias_prms_key='debiasPrms')

    rx_data = np.array(h5['DataRx']['re']) + 1j * np.array(h5['DataRx']['im'])
    rx_data = rx_data.transpose(2, 1, 0)

    chest_algo_idx = h5['srsCellParams']['chEstAlgoIdx'][0]
    num_rx_ant = h5['srsCellParams']['nRxAntSrs'][0]
    num_srs_ues = h5['srsCellParams']['nSrsUes'][0]

    srs_cell_config = SrsRxCellConfig(
        slot=h5['srsCellParams']['slotNum'][0],
        frame=h5['srsCellParams']['frameNum'][0],
        srs_start_sym=h5['srsCellParams']['srsStartSym'][0],
        num_srs_sym=h5['srsCellParams']['nSrsSym'][0]
    )
    srs_cell_configs = [srs_cell_config]

    srs_ue_configs = []
    for ue_idx in range(num_srs_ues):
        cell_idx = 0
        num_ant_ports = h5['srsUePrms']['nAntPorts'][ue_idx]
        num_syms = h5['srsUePrms']['nSyms'][ue_idx]
        num_repetitions = h5['srsUePrms']['nRepetitions'][ue_idx]
        comb_size = h5['srsUePrms']['combSize'][ue_idx]
        start_sym = h5['srsUePrms']['startSym'][ue_idx]
        sequence_id = h5['srsUePrms']['sequenceId'][ue_idx]
        config_idx = h5['srsUePrms']['configIdx'][ue_idx]
        bandwidth_idx = h5['srsUePrms']['bandwidthIdx'][ue_idx]
        comb_offset = h5['srsUePrms']['combOffset'][ue_idx]
        cyclic_shift = h5['srsUePrms']['cyclicShift'][ue_idx]
        frequency_position = h5['srsUePrms']['frequencyPosition'][ue_idx]
        frequency_shift = h5['srsUePrms']['frequencyShift'][ue_idx]
        frequency_hopping = h5['srsUePrms']['frequencyHopping'][ue_idx]
        resource_type = h5['srsUePrms']['resourceType'][ue_idx]
        periodicity = h5['srsUePrms']['Tsrs'][ue_idx]  # pylint: disable=invalid-name
        offset = h5['srsUePrms']['Toffset'][ue_idx]  # pylint: disable=invalid-name
        group_or_sequence_hopping = h5['srsUePrms']['groupOrSequenceHopping'][ue_idx]
        srs_ant_port_to_ue_ant_map = \
            np.frombuffer(h5['srsUePrms']['srsAntPortToUeAntMap'][ue_idx], dtype=np.uint8)
        prg_size = h5['srsUePrms']["prgSize"][ue_idx]
        start_prg = h5['srsChEstBufferInfo']['startPrbGrp'][ue_idx]
        num_prgs = h5['srsChEstBufferInfo']['nPrbGrps'][ue_idx]

        srs_config = SrsConfig(num_ant_ports=num_ant_ports,
                               num_syms=num_syms,
                               num_repetitions=num_repetitions,
                               comb_size=comb_size,
                               start_sym=start_sym,
                               sequence_id=sequence_id,
                               config_idx=config_idx,
                               bandwidth_idx=bandwidth_idx,
                               comb_offset=comb_offset,
                               cyclic_shift=cyclic_shift,
                               frequency_position=frequency_position,
                               frequency_shift=frequency_shift,
                               frequency_hopping=frequency_hopping,
                               resource_type=resource_type,
                               periodicity=periodicity,
                               offset=offset,
                               group_or_sequence_hopping=group_or_sequence_hopping)

        srs_rx_ue_config = SrsRxUeConfig(
            cell_idx=cell_idx,
            srs_config=srs_config,
            srs_ant_port_to_ue_ant_map=srs_ant_port_to_ue_ant_map,
            prg_size=prg_size,
            start_prg=start_prg,
            num_prgs=num_prgs
        )
        srs_ue_configs.append(srs_rx_ue_config)

    srs_rx_config = SrsRxConfig(
        srs_cell_configs=srs_cell_configs,
        srs_ue_configs=srs_ue_configs
    )

    srs_rx = SrsRx(num_rx_ant=[num_rx_ant],
                   chest_algo_idx=chest_algo_idx,
                   enable_delay_offset_correction=1,
                   chest_params=chest_params,
                   cuda_stream=cuda_stream)

    srs_reports = srs_rx(rx_data=[rx_data], config=srs_rx_config)

    for ue_idx, srs_report in enumerate(srs_reports):

        ch_est_ref = np.array(h5["HestUe" + str(ue_idx)])
        ch_est_ref = ch_est_ref["re"] + 1j * ch_est_ref["im"]

        if ch_est_ref.ndim == 2:
            ch_est_ref = ch_est_ref.transpose()
            ch_est_ref = ch_est_ref[..., None]
        else:
            ch_est_ref = ch_est_ref.transpose(2, 1, 0)

        rb_snr_ref = np.array(h5["rbSnrsUe" + str(ue_idx)])
        wideband_snr_ref = np.array(h5["widebandSrsStats"])["widebandSnr"][ue_idx]
        to_est_ms_ref = np.array(h5["widebandSrsStats"])["toEstMicroSec"][ue_idx]

        # Check RB SNR accuracy (the same test as in cuPHY side).
        ref = rb_snr_ref * rb_snr_ref
        tot_ref = np.sum(ref[np.where(ref > 0.1)[0]])
        error = srs_report.rb_snr - rb_snr_ref
        error = error * error
        tot_error = np.sum(error[np.where(ref > 0.1)[0]])
        eval_snr = 10 * np.log10(tot_ref / tot_error)
        assert np.isnan(eval_snr) or eval_snr >= 30

        # Check wideband SRS statistics.
        assert np.isclose(srs_report.to_est_ms, to_est_ms_ref, atol=0.01)
        assert np.isclose(srs_report.wideband_snr, wideband_snr_ref, atol=0.5)

        # Estimation might not be accurate above 50dB, capped at 50 dB for further comparison.
        wideband_cs_corr_ratio_db = min(srs_report.wideband_cs_corr_ratio_db, 50)
        wideband_cs_corr_ratio_db_ref = \
            min(np.array(h5["widebandSrsStats"])["widebandCsCorrRatioDb"][ue_idx], 50)
        assert np.isclose(wideband_cs_corr_ratio_db, wideband_cs_corr_ratio_db_ref, atol=0.5)

        # Check channel estimation accuracy. Compare SNR to tolerance.
        error = ch_est_ref - srs_report.ch_est
        error_energy = np.mean(np.abs(error) * np.abs(error))
        signal_energy = np.mean(np.abs(ch_est_ref) * np.abs(ch_est_ref))
        eval_snr = 10 * np.log10(signal_energy / error_energy)
        assert eval_snr >= 30
