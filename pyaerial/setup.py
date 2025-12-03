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

"""Setup file for PyAerial package."""
import os
import sys

import platform
import setuptools

sys.path.insert(0, os.path.abspath("./src/aerial"))
import version_aerial  # pylint: disable=E0401,C0413

machine = platform.machine()
version = version_aerial.RELEASE

setuptools.setup(
    name="pyaerial",
    version=version,
    author="NVIDIA",
    description="NVIDIA pyAerial library",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={"": [
        "version_aerial.py",
        f"_pycuphy.cpython-310-{machine}-linux-gnu.so",
        "libcuphy.so",
        "libnvlog.so",
        "libfmtlog-shared.so",
        "libchanModels.so"
    ]},
    python_requires=">=3.7",
    zip_safe=False,
)
