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

"""Test the whole LPDC coding chain end to end."""
import itertools
import pytest

import cupy as cp
import numpy as np

from aerial.phy5g.ldpc import CrcEncoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.phy5g.ldpc import LdpcEncoder
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import LdpcRateMatch
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import random_tb

test_cases = [
    # The tuples denote:
    # enable_scrambling, mcs, num_prb, rv
    # Note that RV1 and RV2 need a lower code rate to be self-decodable.
    (True, 3, 100, 0),
    (True, 1, 100, 1),
    (True, 1, 100, 2),
    (True, 3, 100, 3),
    (False, 3, 100, 0),
    (False, 1, 100, 1),
    (False, 1, 100, 2),
    (False, 3, 100, 3),
    (True, 10, 6, 0),
    (True, 1, 6, 1),
    (True, 1, 6, 2),
    (True, 10, 6, 3),
    (False, 10, 6, 0),
    (False, 1, 6, 1),
    (False, 1, 6, 2),
    (False, 10, 6, 3),
    (True, 10, 100, 0),
    (True, 1, 100, 1),
    (True, 1, 100, 2),
    (True, 10, 100, 3),
    (False, 10, 100, 0),
    (False, 1, 100, 1),
    (False, 1, 100, 2),
    (False, 10, 100, 3),
    (True, 10, 272, 0),
    (True, 1, 272, 1),
    (True, 1, 272, 2),
    (True, 10, 272, 3),
    (False, 10, 272, 0),
    (False, 1, 272, 1),
    (False, 1, 272, 2),
    (False, 10, 272, 3),
]

test_cases = list(itertools.product(test_cases, [True, False]))
test_cases = [(enable_scrambling, mcs, num_prb, rv, h2d)
              for (enable_scrambling, mcs, num_prb, rv), h2d in test_cases]


@pytest.mark.parametrize(
    "enable_scrambling, mcs, num_prb, rv, h2d",
    test_cases,
    ids=[f"(Scrambling: {enable_scrambling}, MCS: {mcs}, #PRB: {num_prb}, RV: {rv}, cuPy: {h2d})"
         for enable_scrambling, mcs, num_prb, rv, h2d in test_cases]
)
def test_ldpc_endtoend(cuda_stream, enable_scrambling, mcs, num_prb, rv, h2d):

    start_sym = 0
    num_ofdm_symbols = 14
    num_layers = 1

    rnti = 20000               # UE RNTI
    data_scid = 41             # Data scrambling ID
    cinit = (rnti << 15) + data_scid
    dmrs_sym = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    crc_encoder = CrcEncoder(cuda_stream=cuda_stream)
    crc_check = CrcChecker(cuda_stream=cuda_stream)
    ldpc_encoder = LdpcEncoder(cuda_stream=cuda_stream)
    ldpc_decoder = LdpcDecoder(cuda_stream=cuda_stream)
    ldpc_rate_match = LdpcRateMatch(enable_scrambling=enable_scrambling, cuda_stream=cuda_stream)
    ldpc_derate_match = LdpcDeRateMatch(
        enable_scrambling=enable_scrambling,
        cuda_stream=cuda_stream
    )

    mod_order, code_rate = get_mcs(mcs)

    # Generate a random transport block (in bits).
    transport_block = random_tb(
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs_syms=dmrs_sym,
        num_prbs=num_prb,
        start_sym=start_sym,
        num_symbols=num_ofdm_symbols,
        num_layers=num_layers,
        return_bits=False
    )

    # LDPC components take TB size and code rate in different units.
    tb_size = transport_block.shape[0] * 8
    code_rate /= 1024.

    if h2d:
        transport_block = cp.array(transport_block, order='F')

    code_blocks = crc_encoder.encode(
        tb_inputs=[transport_block],
        tb_sizes=[tb_size],
        code_rates=[code_rate]
    )

    coded_bits = ldpc_encoder.encode(
        code_blocks=code_blocks,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        redundancy_versions=[rv]
    )

    num_data_sym = (num_ofdm_symbols - np.array(dmrs_sym).sum())
    rate_match_len = num_data_sym * num_prb * 12 * num_layers * mod_order
    rate_matched_bits = ldpc_rate_match.rate_match(
        coded_blocks=coded_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lens=[rate_match_len],
        mod_orders=[mod_order],
        num_layers=[num_layers],
        redundancy_versions=[rv],
        cinits=[cinit]
    )

    assert len(rate_matched_bits) == 1
    rate_matched_bits = rate_matched_bits[0]

    rx_bits = 1 - 2. * rate_matched_bits

    derate_matched_bits = ldpc_derate_match.derate_match(
        input_llrs=[rx_bits],
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lengths=[rate_match_len],
        mod_orders=[mod_order],
        num_layers=[num_layers],
        redundancy_versions=[rv],
        ndis=[1],
        cinits=[cinit]
    )

    decoded_bits = ldpc_decoder.decode(
        input_llrs=derate_matched_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        redundancy_versions=[rv],
        rate_match_lengths=[rate_match_len]
    )

    decoded_tb = crc_check.check_crc(
        input_bits=decoded_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate]
    )[0][0]

    if h2d:
        decoded_tb = decoded_tb.get(order='F')
        transport_block = transport_block.get(order='F')

    assert np.array_equal(transport_block, decoded_tb)
