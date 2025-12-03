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

"""Utility functions for the SCF FAPI interface.

The FAPI module contains various utilities for handling the interface between the PUSCH database
schema (SCF FAPI) and cuPHY.
"""
from typing import List
from typing import Union

import numpy as np


def dmrs_fapi_to_bit_array(dmrs_symb_pos: np.uint16) -> list:
    """Convert the DMRS symbol position decimal value to a bit array.

    Args:
       dmrs_symb_pos (np.uint16): DMRS symbol position decimal value as defined in SCF FAPI.

    Returns:
        list: A bit array to be used for cuPHY interface, indicating the positions
        of DMRS symbols. The first bit corresponds to OFDM symbol 0.
    """
    return [int(k) for k in format(dmrs_symb_pos, "015b")[14:0:-1]]


def dmrs_bit_array_to_fapi(x: List[int]) -> np.uint16:
    """Convert a bit array to DMRS symbol position decimal value.

    Args:
        x (list): A bit array to be used for cuPHY interface, indicating the positions of
            DMRS symbols. The first bit corresponds to OFDM symbol 0.

    Returns:
        np.uint16: DMRS symbol position decimal value as defined in SCF FAPI.
    """
    k = 0
    pow_two = 1
    for bit in x:
        k = k + int(bit) * pow_two
        pow_two = pow_two * 2
    return np.uint16(k)


def dmrs_fapi_to_sym(dmrs_symb_pos: np.uint16) -> list:
    """Convert the DMRS symbol position decimal value to a list of DMRS symbol indices.

    Args:
       dmrs_symb_pos (np.uint16): DMRS symbol position decimal value as defined in SCF FAPI.

    Returns:
        list: A list of DMRS symbol indices.
    """
    return list(np.nonzero(dmrs_fapi_to_bit_array(dmrs_symb_pos))[0])


def mac_pdu_to_bit_array(mac_pdu: Union[list, np.ndarray]) -> list:
    """Convert MAC PDU bytes to a bit array.

    Args:
        mac_pdu (list): A list of bytes, the content of the MAC PDU.

    Returns:
        list: The same MAC PDU as a bit array, i.e. the bytes are converted to a list of bits.
    """
    bits = []
    for byte in mac_pdu:
        bits += [int(d) for d in format(byte, "08b")]
    return bits


def bit_array_to_mac_pdu(bits: list) -> list:
    """Convert a bit array to MAC PDU bytes.

    Args:
        bits (list): A MAC PDU as a bit array.

    Returns:
        list: A list of bytes corresponding to the above MAC PDU.
    """
    mac_pdu = [
        int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8)
    ]
    return mac_pdu
