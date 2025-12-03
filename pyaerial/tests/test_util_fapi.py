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

"""Unit tests for aerial/util/fapi.py."""
from aerial.util.fapi import (
    dmrs_fapi_to_bit_array,
    dmrs_bit_array_to_fapi,
    dmrs_fapi_to_sym,
    bit_array_to_mac_pdu,
    mac_pdu_to_bit_array,
)


def test_dmrs_fapi_to_bit_array():
    """Test dmrs_fapi_to_bit_array()."""
    ul_dmrs_symb_pos = 4
    array = dmrs_fapi_to_bit_array(ul_dmrs_symb_pos)
    assert array == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_dmrs_bit_array_to_fapi():
    """Test dmrs_bit_array_to_fapi()."""
    ul_dmrs_symb_pos = 4
    array = dmrs_fapi_to_bit_array(ul_dmrs_symb_pos)
    symb_pos = dmrs_bit_array_to_fapi(array)
    assert ul_dmrs_symb_pos == symb_pos


def test_dmrs_fapi_to_sym():
    """Test dmrs_fapi_to_sym()."""
    ul_dmrs_symb_pos = 4
    symb_idx = dmrs_fapi_to_sym(ul_dmrs_symb_pos)
    assert symb_idx == [2]


def test_mac_pdu_to_bit_array():
    """Test mac_pdu_to_bit_array()."""
    mac_pdu = [32, 4, 16]
    bit_array = mac_pdu_to_bit_array(mac_pdu)
    assert bit_array == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]


def test_bit_array_to_mac_pdu():
    """Test bit_array_to_mac_pdu()."""
    bit_array = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    mac_pdu = bit_array_to_mac_pdu(bit_array)
    assert mac_pdu == [32, 192, 7]
