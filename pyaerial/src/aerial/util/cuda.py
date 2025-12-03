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

"""CUDA-related utilities."""
from typing import Any

import cuda.bindings.runtime as cudart  # type: ignore


def get_cuda_stream() -> cudart.cudaStream_t:
    """Return a CUDA stream.

    Returns:
        cudart.cudaStream_t: A new CUDA stream.
    """
    cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
    return cuda_stream


def check_cuda_errors(result: cudart.cudaError_t) -> Any:
    """Check CUDA errors.

    Args:
        result (cudart.cudaError_t): CUDA error value.
    """
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]
