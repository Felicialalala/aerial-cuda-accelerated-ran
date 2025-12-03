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

"""Python bindings for cuPHY - Filters for PUSCH channel estimation."""
# pylint: disable=too-many-lines
import pathlib
from typing import Optional

import h5py  # type: ignore
import numpy as np

# Channel estimation filter coefficient file.
CUPHY_CHEST_COEFF_FILE = str(pathlib.Path(__file__).parent.parent.parent.parent.parent /
                             'testVectors' / 'cuPhyChEstCoeffs.h5')


def chest_params_from_hdf5(filename: str) -> dict:
    """Load channel estimation parameters from a HDF5 file.

    Args:
        filename (str): Absolute path and filename of the HDF5 file.

    Returns:
        dict: Channel estimation parameters in a dictionary.
    """
    chest_params = dict()
    h5_file = h5py.File(filename, "r")
    keys = list(h5_file.keys())
    for key in keys:
        chest_params[key] = np.array(h5_file[key])

    return chest_params


def _to_numpy(h5_param):  # type: ignore
    """Convert to numpy array."""
    np_array = np.array(h5_param)
    np_array = np_array.T
    if np_array.dtype == np.uint8:
        return np_array
    if np_array.dtype == np.float16:
        np_array = np_array.astype(np.float32)
        return np_array
    if not np_array.dtype == np.float32:
        np_array = np_array["re"] + 1j * np_array["im"]
        np_array = np_array.astype(np.complex64)
    return np_array


def pusch_chest_params_from_hdf5(filename: str) -> dict:
    """Load PUSCH channel estimation parameters from a HDF5 file.

    Args:
        filename (str): Absolute path and filename of the HDF5 file.

    Returns:
        dict: Channel estimation parameters in a dictionary.
    """
    chest_params_from_h5 = chest_params_from_hdf5(filename)

    param_keys = [
        "ShiftSeq",
        "ShiftSeq4",
        "UnShiftSeq",
        "UnShiftSeq4",
        "WFreq",
        "WFreq4",
        "WFreqSmall",
    ]
    chest_params = {key: _to_numpy(chest_params_from_h5[key]) for key in param_keys}
    return chest_params


def srs_chest_params_from_hdf5(
    filename: str, debias_prms_key: Optional[str] = None
) -> dict:
    """Load SRS channel estimation parameters from a HDF5 file.

    This function loads parameters from the given HDF5 file and forms
    the dictionary that can be given directly to the cuPHY Python API.

    Args:
        filename (str): Absolute path of the HDF5 file containing the
            channel estimation filters and parameters.
        debias_prms_key (str): The string key used to find the debias parameters
            from the HDF5 file. Default: "srsNoiseEstDebiasPrms".

    Returns:
        dict: The channel esimation parameter dictionary.
    """
    chest_params_from_h5 = chest_params_from_hdf5(filename)
    debias_prms_key = debias_prms_key or "srsNoiseEstDebiasPrms"

    # These are the keys that need to be found from the given File.
    srs_param_keys = [
        "W_comb2_nPorts1_narrow",
        "W_comb2_nPorts1_wide",
        "W_comb2_nPorts2_narrow",
        "W_comb2_nPorts2_wide",
        "W_comb2_nPorts4_narrow",
        "W_comb2_nPorts4_wide",
        "W_comb2_nPorts8_narrow",
        "W_comb2_nPorts8_wide",
        "W_comb4_nPorts1_narrow",
        "W_comb4_nPorts1_wide",
        "W_comb4_nPorts2_narrow",
        "W_comb4_nPorts2_wide",
        "W_comb4_nPorts4_narrow",
        "W_comb4_nPorts4_wide",
        "W_comb4_nPorts6_narrow",
        "W_comb4_nPorts6_wide",
        "W_comb4_nPorts12_narrow",
        "W_comb4_nPorts12_wide",
        "focc_table",
        "focc_table_comb2",
        "focc_table_comb4",
        "srsRkhs_eigenCorr_grid0",
        "srsRkhs_eigenCorr_grid1",
        "srsRkhs_eigenCorr_grid2",
        "srsRkhs_secondStageTwiddleFactors_grid2",
        "srsRkhs_secondStageFourierPerm_grid2",
        "srsRkhs_eigenVecs_grid0",
        "srsRkhs_eigenVecs_grid1",
        "srsRkhs_eigenVecs_grid2",
        "srsRkhs_eigValues_grid0",
        "srsRkhs_eigValues_grid1",
        "srsRkhs_eigValues_grid2",
    ]

    chest_params = {key: _to_numpy(chest_params_from_h5[key]) for key in srs_param_keys}

    # Add debias parameters which needs to be separately opened from the h5.
    keys = list(np.array(chest_params_from_h5[debias_prms_key]).dtype.fields.keys())  # type: ignore

    chest_params.update(
        {key: chest_params_from_h5[debias_prms_key][key][0] for key in keys}
    )
    return chest_params
