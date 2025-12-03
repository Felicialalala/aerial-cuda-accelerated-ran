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

"""
This module defines the fundamental base classes for the PyAerial Algorithm
Framework. It includes `AlgorithmBase`, an abstract class defining the common
interface for all algorithms, and `TensorSpec`, a class for specifying the
shape, type, and other metadata of input/output tensors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import numpy as np


class TensorSpec:
    """Specification for a tensor, including shape and metadata."""

    def __init__(self,
                 *,
                 name: str,
                 shape: List[Union[str, int]],
                 dtype: str = "float32",
                 is_dynamic: bool = False,
                 dynamic_axes: Dict[str, int] = None):
        """
        Args:
            name: Tensor name
            shape: Symbolic or fixed shape (e.g., ["batch", "subcarriers",
                   "layers", 2])
            dtype: Data type (float32, float16, complex64, complex128, etc.)
                   Complex types will automatically be detected.
            is_dynamic: Whether tensor has dynamic dimensions
            dynamic_axes: Map of dimension name to dimension index for
                          dynamic dims
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.is_dynamic = is_dynamic
        self.dynamic_axes = dynamic_axes or {}

    @property
    def is_complex(self) -> bool:
        """Infer whether this tensor holds complex values based on dtype."""
        return 'complex' in str(self.dtype).lower()

    def __repr__(self) -> str:
        """Return string representation of the TensorSpec object."""
        return (f"TensorSpec(name='{self.name}', shape={self.shape}, "
                f"dtype='{self.dtype}', is_complex={self.is_complex})")


class AlgorithmBase(ABC):
    """Base class for all algorithm implementations."""

    @abstractmethod
    def get_input_specs(self) -> List[TensorSpec]:
        """Return specifications for algorithm inputs."""
        pass

    @abstractmethod
    def get_output_specs(self) -> List[TensorSpec]:
        """Return specifications for algorithm outputs."""
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Process inputs and return outputs."""
        pass

    def get_example_inputs(
        self, **kwargs: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """Generate example inputs based on specs."""
        inputs = {}
        for spec in self.get_input_specs():
            # Replace symbolic dimensions with default values
            shape = []
            for dim in spec.shape:
                if isinstance(dim, str):
                    # Use provided value or default to 1
                    shape.append(kwargs.get(dim, 1))
                else:
                    shape.append(dim)

            # Create random tensor with appropriate shape and data type
            if spec.is_complex:
                # Complex input needs real and imaginary parts
                real = np.random.randn(*shape).astype(np.float32)
                imag = np.random.randn(*shape).astype(np.float32)
                inputs[spec.name] = real + 1j * imag
                # Convert to the correct complex dtype if specified
                if spec.dtype == 'complex64':
                    inputs[spec.name] = inputs[spec.name].astype(np.complex64)
                elif spec.dtype == 'complex128':
                    inputs[spec.name] = inputs[spec.name].astype(np.complex128)
            else:
                inputs[spec.name] = np.random.randn(*shape).astype(spec.dtype)

        return inputs

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the algorithm."""
        return {
            "algorithm_type": "base",
            "version": "1.0.0"
        }
