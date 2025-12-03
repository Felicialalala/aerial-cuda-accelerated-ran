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

"""pyAerial - PUSCH receiver pipeline base class definition."""
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import List
from typing import Tuple

from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import _SlotConfigT


class PuschRxPipeline(Pipeline, Generic[_SlotConfigT, Array]):
    """A base class for PUSCH receiver pipeline implementations."""

    @abstractmethod
    def __call__(self,
                 slot: int,
                 rx_slot: Array,
                 config: List[_SlotConfigT],
                 **kwargs: Any) -> Tuple[Array, List[Array]]:
        """Abstract method that runs the receiver pipeline.

        This method gives the signature that the receiver pipelines should
        implement.

        Args:
            slot (int): Slot number.
            rx_slot (Array): Received slot as an Array.
            config (List[_SlotConfigT]): Dynamic slot configuration in this slot. Note that the
                type of this configuration should be derived from `SlotConfig`.

        Returns:
            Array, List[Array]: A tuple containing:

            - *Array*: Transport block CRCs.

            - *List[Array]*: Transport blocks, one per UE, without CRC.
        """
        raise NotImplementedError
