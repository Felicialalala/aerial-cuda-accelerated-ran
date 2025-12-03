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

"""pyAerial library - Soft demapper."""
from typing import List

import numpy as np


class Demapper:
    """This class provides demapping of symbols to log-likelihood ratios.

    The algorithm used is the exact log-MAP mapping, which is computationally
    intensive. Note also that this is currently implemented purely in Python
    so it may be slow.
    """
    def __init__(self, mod_order: int) -> None:
        """Initialize demapper.

        Args:
            mod_order (int): Modulation order. Supported values: 2, 4, 6, 8.
        """
        self.mod_order = mod_order
        self.bit_to_symbol_map = self._bit_to_symbol_map()

    def _generate_bit_strings(self) -> List[str]:
        """Generate all possible bit strings for the given modulation order."""
        bit_strings = []

        def _gen_bits(n: int, bit_string: str = "") -> None:
            """A recursive function to generate a bit string."""
            if len(bit_string) == n:
                bit_strings.append(bit_string)
            else:
                _gen_bits(n, bit_string + "0")
                _gen_bits(n, bit_string + "1")

        _gen_bits(self.mod_order)
        return bit_strings

    def _bit_to_symbol_map(self) -> dict:
        """Generate a mapping from bit strings to symbols for the given modulation order."""
        bit_to_symbol_map = {}
        bit_strings = self._generate_bit_strings()
        for bit_string in bit_strings:
            bits = [int(b) for b in bit_string]
            if self.mod_order == 2:
                bit_to_symbol_map[bit_string] = self._map_qpsk(bits)
            elif self.mod_order == 4:
                bit_to_symbol_map[bit_string] = self._map_16qam(bits)
            elif self.mod_order == 6:
                bit_to_symbol_map[bit_string] = self._map_64qam(bits)
            elif self.mod_order == 8:
                bit_to_symbol_map[bit_string] = self._map_256qam(bits)
        return bit_to_symbol_map

    def _map_qpsk(self, bits: List[int]) -> complex:
        """Map bits to QPSK symbols."""
        symbol = (1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])
        symbol /= np.sqrt(2)
        return symbol

    def _map_16qam(self, bits: List[int]) -> complex:
        """Map bits to 16QAM symbols."""
        symbol = (1 - 2 * bits[0]) * (2 - (1 - 2 * bits[2])) + \
            1j * (1 - 2 * bits[1]) * (2 - (1 - 2 * bits[3]))
        symbol /= np.sqrt(10)
        return symbol

    def _map_64qam(self, bits: List[int]) -> complex:
        """Map bits to 64QAM symbols."""
        symbol = (1 - 2 * bits[0]) * (4 - (1 - 2 * bits[2]) * (2 - (1 - 2 * bits[4]))) + \
            1j * (1 - 2 * bits[1]) * (4 - (1 - 2 * bits[3]) * (2 - (1 - 2 * bits[5])))
        symbol /= np.sqrt(42)
        return symbol

    def _map_256qam(self, bits: List[int]) -> complex:
        """Map bits to 256QAM symbols."""
        symbol = (1 - 2 * bits[0]) * (8 - (1 - 2 * bits[2])) * \
            (4 - (1 - 2 * bits[4]) * (2 - (1 - 2 * bits[6]))) + \
            1j * (1 - 2 * bits[1]) * (8 - (1 - 2 * bits[3])) * \
            (4 - (1 - 2 * bits[5]) * (2 - (1 - 2 * bits[7])))
        symbol /= np.sqrt(170)
        return symbol

    def demap(self, syms: np.ndarray, noise_var_inv: np.ndarray) -> np.ndarray:
        """Run demapping.

        Args:
            syms (np.ndarray): An array of modulation symbols.
            noise_var_inv (np.ndarray): Inverse of noise variance per subcarrier. The size of this
                array must broadcast with `syms`.

        Returns:
            np.ndarray: Log-likelihood ratios. The first dimension is modulation order, otherwise
            the dimensions are the same as those of `syms`.
        """
        llr = np.zeros((self.mod_order, *syms.shape))

        zero_sum = np.zeros_like(llr)
        one_sum = np.zeros_like(llr)

        for bit_string, ref_sym in self.bit_to_symbol_map.items():
            exponent = np.square(np.abs(syms - ref_sym))
            exponent = -2 * self.mod_order * exponent * noise_var_inv
            sum_term = np.exp(exponent)
            for i in range(self.mod_order):
                if bit_string[i] == '0':
                    zero_sum[i, ...] += sum_term
                else:
                    one_sum[i, ...] += sum_term
        one_sum[np.where(one_sum == 0.)] = 1e-15
        zero_sum[np.where(zero_sum == 0.)] = 1e-15
        llr = np.log2(zero_sum) - np.log2(one_sum)
        return llr
