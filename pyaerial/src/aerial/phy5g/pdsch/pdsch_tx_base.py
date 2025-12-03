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

"""pyAerial - PDSCH transmitter pipeline base class definition."""
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import List

from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import _SlotConfigT

# Constant definitions.
NUM_RE_PER_PRB = 12
NUM_PRB_MAX = 273
NUM_SYMBOLS = 14
MAX_DL_LAYERS = 16


class PdschTxPipeline(Pipeline, Generic[_SlotConfigT, Array]):
    """A base class for PDSCH transmitter pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 slot: int,
                 tb_inputs: List[Array],
                 config: List[_SlotConfigT],
                 **kwargs: Any) -> Array:
        """Abstract method that runs the transmitter pipeline.

        This method gives the signature that the transmitter pipelines should
        implement.

        Args:
            slot (int): Slot number.
            tb_inputs (List[Array]): List of transport blocks, one per UE.
            config (List[_SlotConfigT]): Dynamic slot configuration in this slot. Note that the
                type of this configuration should be derived from `SlotConfig`.

        Returns:
            Array: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        raise NotImplementedError
