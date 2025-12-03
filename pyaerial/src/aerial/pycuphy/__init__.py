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

"""pyAerial - The cuPHY backend."""
import ctypes
import os

# Load shared libraries.
libcuphy_path = os.path.dirname(os.path.realpath(__file__))

dynamic_libs = ["libfmtlog-shared.so", "libnvlog.so", "libcuphy.so", "libchanModels.so"]
for lib in dynamic_libs:
    so = os.path.join(libcuphy_path, lib)
    if os.path.isfile(so):
        ctypes.cdll.LoadLibrary(so)

# Disable lints due to import being in the wrong place.
from ._pycuphy import *  # type: ignore  # noqa: F403  # pylint: disable=C0413
