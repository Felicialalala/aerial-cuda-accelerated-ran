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

"""Tests for srs_tx.py."""
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py
import itertools
import numpy as np

from aerial.phy5g.srs import SrsTx
from aerial.phy5g.srs import SrsConfig
from aerial.phy5g.srs import SrsTxConfig

test_case_numbers = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013,
                     8014, 8015, 8016, 8017, 8018, 8019, 8020, 8021, 8022, 8023, 8024, 8026, 8027,
                     8028, 8029, 8030, 8031, 8032, 8033, 8035, 8051, 8052, 8053, 8054, 8055, 8056,
                     8057, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112,
                     8113, 8114, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122, 8123, 8124, 8125,
                     8126, 8127, 8128, 8129, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138, 8139,
                     8140, 8141, 8142, 8143, 8144, 8145, 8146, 8147, 8148, 8149, 8151, 8152, 8153,
                     8154, 8155, 8156, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8164, 8201, 8202,
                     8203, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8212, 8213, 8222, 8223,
                     8224, 8225, 8226, 8227, 8301, 8302, 8401, 8402, 8403, 8404, 8405, 8406, 8407,
                     8408, 8409, 8410, 8411, 8412, 8413, 8414, 8415, 8420, 8421, 8801, 8802]

all_cases = list(itertools.product(test_case_numbers, [True, False]))


# pylint: disable=too-many-locals
@pytest.mark.parametrize(
    "test_case_number, copy_to_cpu",
    all_cases,
    ids=[f"{test_case_number}, copy_to_cpu: {copy_to_cpu}"
         for test_case_number, copy_to_cpu in all_cases]
)
def test_srs_tx(cuda_stream, test_case_number, copy_to_cpu):
    """Test SrsTx."""
    # pylint: disable=too-many-locals
    filenames = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_SRS_UE*.h5")
    num_ues = len(filenames)

    srs_configs = []
    ref_tx_buffers = []
    num_tx_ant = []
    for filename in filenames:
        try:
            h5 = h5py.File(filename, "r")  # pylint: disable=invalid-name
        except FileNotFoundError:
            pytest.skip("Test vector file not available, skipping...")
            return

        srs_config = SrsConfig(
            num_ant_ports=h5["SrsParams"]["N_ap_SRS"][0],
            num_syms=h5["SrsParams"]["N_symb_SRS"][0],
            num_repetitions=h5["SrsParams"]["R"][0],
            comb_size=h5["SrsParams"]["K_TC"][0],
            start_sym=h5["SrsParams"]["l0"][0],
            sequence_id=h5["SrsParams"]["n_ID_SRS"][0],
            config_idx=h5["SrsParams"]["C_SRS"][0],
            bandwidth_idx=h5["SrsParams"]["B_SRS"][0],
            comb_offset=h5["SrsParams"]["k_TC_bar"][0],
            cyclic_shift=h5["SrsParams"]["n_SRS_cs"][0],
            frequency_position=h5["SrsParams"]["n_RRC"][0],
            frequency_shift=h5["SrsParams"]["n_shift"][0],
            frequency_hopping=h5["SrsParams"]["b_hop"][0],
            resource_type=h5["SrsParams"]["resourceType"][0],
            periodicity=h5["SrsParams"]["Tsrs"][0],
            offset=h5["SrsParams"]["Toffset"][0],
            group_or_sequence_hopping=h5["SrsParams"]["groupOrSequenceHopping"][0]
        )
        num_tx_ant.append(h5["SrsParams"]["N_ap_SRS"][0])
        srs_configs.append(srs_config)

        ref_tx_buffer = np.array(h5["X_tf"])["re"] + 1j * np.array(h5["X_tf"])["im"]
        ref_tx_buffers.append(ref_tx_buffer)

    num_slot_per_frame = h5["SrsParams"]["N_slot_frame"][0]
    num_symb_per_slot = h5["SrsParams"]["N_symb_slot"][0]
    srs_tx = SrsTx(num_max_srs_ues=num_ues,
                   num_slot_per_frame=num_slot_per_frame,
                   num_symb_per_slot=num_symb_per_slot,
                   cuda_stream=cuda_stream)

    slot = h5["SrsParams"]["idxSlotInFrame"][0]
    frame = h5["SrsParams"]["idxFrame"][0]

    config = SrsTxConfig(slot=slot, frame=frame, srs_configs=srs_configs)

    tx_buffers = srs_tx(config=config, copy_to_cpu=copy_to_cpu)

    for ue_idx in range(num_ues):
        num_sc = ref_tx_buffers[ue_idx].shape[2]
        tx_buffer = tx_buffers[ue_idx][:num_sc, ...]
        if not copy_to_cpu:
            tx_buffer = tx_buffer.get()

        np.allclose(tx_buffer,
                    ref_tx_buffers[ue_idx].transpose(2, 1, 0)[..., :num_tx_ant[ue_idx]],
                    rtol=0.001)
