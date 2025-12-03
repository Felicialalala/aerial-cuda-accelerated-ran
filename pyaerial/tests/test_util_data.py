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

"""Tests for pusch.py."""
import pytest
import pandas as pd

from aerial.util.data import PuschRecord


def test_pusch_record_simple():

    try:
        df = pd.read_parquet("test_data/example.parquet")
    except FileNotFoundError:
        pytest.skip("Test data not available, skipping...")
        return

    # Grab first row
    row = df.iloc[0]
    dict_row = dict(row)

    # Verify record conversion works
    PuschRecord(**dict_row)

    # Test also the other way of doing the conversion.
    PuschRecord.from_series(row)

    # Test also that columns returns something.
    fields = PuschRecord.columns()
    assert isinstance(fields, tuple)
