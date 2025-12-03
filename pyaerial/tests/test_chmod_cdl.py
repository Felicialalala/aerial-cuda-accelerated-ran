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

from aerial import pycuphy
import numpy as np
import pytest
from cuda.bindings import runtime
import os


def cdl_chan_numpy(cdl_cfg, n_tti, cuda_stream):
    """Test CDL channel using numpy for Tx input signal, save data into H5 files."""
    n_bs_ant = np.prod(cdl_cfg.bs_ant_size)  # number of BS antennas
    n_ue_ant = np.prod(cdl_cfg.ue_ant_size)  # number of UE antennas
    # Step 1: create random input signals using numpy
    tx_signal_size = cdl_cfg.n_cell * cdl_cfg.n_ue * \
        cdl_cfg.signal_length_per_ant * n_bs_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
    else:
        tx_signal_in = None

    # Step 2: create CDL channel object
    cdl = pycuphy.CdlChan(
        cdl_cfg=cdl_cfg,
        tx_signal_in_cpu=tx_signal_in,
        rand_seed=0,
        stream_handle=cuda_stream
    )

    # Step 3: run test and save H5 file(s)
    rx_signal_size = cdl_cfg.n_cell * cdl_cfg.n_ue * \
        cdl_cfg.signal_length_per_ant * n_ue_ant
    if rx_signal_size > 0:
        rx_signal_out = np.empty(rx_signal_size, dtype=np.complex64)
    else:
        rx_signal_out = None
    for tti_idx in range(0, n_tti):
        # generate input signal
        if tx_signal_size > 0:
            tx_signal_in.real = np.random.randn(tx_signal_size)
            tx_signal_in.imag = np.random.randn(tx_signal_size)

        # run CDL test
        cdl.run(
            tx_freq_signal_in_cpu=tx_signal_in,
            rx_freq_signal_out_cpu=rx_signal_out,
            ref_time0=tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_numpy"
            cdl.save_cdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"cdlChan_{cdl_cfg.n_cell}cell{cdl_cfg.n_ue}Ue_"
                f"{n_bs_ant}x{n_ue_ant}_"
                f"{cdl_cfg.delay_profile}{int(cdl_cfg.delay_spread)}_"
                f"dopp{int(cdl_cfg.max_doppler_shift)}_"
                f"cfo{int(cdl_cfg.cfo_hz)}_"
                f"runMode{cdl_cfg.run_mode}_"
                f"freqConvert{cdl_cfg.freq_convert_type}_"
                f"scSampling{cdl_cfg.sc_sampling}_"
                f"FP32_swap0{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            # os.remove(out_filename)


def cdl_chan_gpu_only(cdl_cfg, n_tti, cuda_stream):
    """Test CDL channel using GPU memory for tx input signal, save data into H5 files."""
    n_bs_ant = np.prod(cdl_cfg.bs_ant_size)  # number of BS antennas
    n_ue_ant = np.prod(cdl_cfg.ue_ant_size)  # number of UE antennas
    # Step 1: create buffer for random input signals and obtain GPU memory address
    tx_signal_size = cdl_cfg.n_cell * cdl_cfg.n_ue * \
        cdl_cfg.signal_length_per_ant * n_bs_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
        # Allocate GPU memory
        err, tx_signal_in_gpu = runtime.cudaMalloc(tx_signal_in.nbytes)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate GPU memory: {err}")
    else:
        tx_signal_in = None
        tx_signal_in_gpu = 0

    # Step 2: create CDL channel object
    cdl = pycuphy.CdlChan(
        cdl_cfg=cdl_cfg,
        tx_signal_in_gpu=tx_signal_in_gpu,
        rand_seed=0,
        stream_handle=cuda_stream
    )

    # Step 3: run test and save H5 file(s)
    for tti_idx in range(0, n_tti):
        # generate input signal
        if tx_signal_size > 0:
            tx_signal_in.real = np.random.randn(tx_signal_size)
            tx_signal_in.imag = np.random.randn(tx_signal_size)
            # Copy data from host to device
            err = runtime.cudaMemcpy(
                tx_signal_in_gpu, tx_signal_in.ctypes.data,
                tx_signal_in.nbytes, runtime.cudaMemcpyKind.cudaMemcpyHostToDevice
            )
            if err != (runtime.cudaError_t.cudaSuccess,):
                raise RuntimeError(f"Failed to copy data to GPU: {err}")

        # run CDL test
        cdl.run(tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_gpuOnly"
            cdl.save_cdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"cdlChan_{cdl_cfg.n_cell}cell{cdl_cfg.n_ue}Ue_"
                f"{n_bs_ant}x{n_ue_ant}_"
                f"{cdl_cfg.delay_profile}{int(cdl_cfg.delay_spread)}_"
                f"dopp{int(cdl_cfg.max_doppler_shift)}_"
                f"cfo{int(cdl_cfg.cfo_hz)}_"
                f"runMode{cdl_cfg.run_mode}_"
                f"freqConvert{cdl_cfg.freq_convert_type}_"
                f"scSampling{cdl_cfg.sc_sampling}_"
                f"FP32_swap0{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            # os.remove(out_filename)

    # Cleanup GPU memory
    if tx_signal_size > 0:
        err = runtime.cudaFree(tx_signal_in_gpu)
        if err != (runtime.cudaError_t.cudaSuccess,):
            print(f"Warning: Failed to free GPU memory: {err}")


@pytest.mark.parametrize(
    "cdl_type, n_tti, run_mode, numpy_indicator", [
        ('A', 100, 0, 0),
        ('A', 100, 0, 1),
        ('A', 100, 1, 0),
        ('A', 100, 1, 1),
        ('C', 100, 0, 0),
        ('C', 100, 0, 1),
        ('C', 100, 1, 0),
        ('C', 100, 1, 1)
    ])
def test_cdl_chan(cdl_type, n_tti, run_mode, numpy_indicator, cuda_stream):
    """Main test function of CDL channel

    Paremeters:
    - cdl_type: The type of CDL channel to use for the test
        - 'A' - CDLA30-10, 'B' - CDLB100-400, 'C' - CDLC300-100
    - n_tti: number of TTIs in test
        - time stamp is tti_idx * 5e-4 assuming 500 us per time slot
        - tti_idx is (0, n_tti)
    - run_mode: The mode in which the test should be executed
        - 0: only time channel
        - 1: time + freq channel on prbg
        - 2: time + freq channel on prbg and sc
    - numpy_indicator: 1 - run test with numpy; 0 - run test with GPU momery directly
    - cuda_stream: cuda_stream to run test
    """
    try:
        # CDL configuration
        cdl_cfg = pycuphy.CdlConfig()
        cdl_cfg.delay_profile = cdl_type
        match cdl_type:
            case 'A':
                cdl_cfg.delay_spread = 30
                cdl_cfg.max_doppler_shift = 10
            case 'B':
                cdl_cfg.delay_spread = 100
                cdl_cfg.max_doppler_shift = 400
            case 'C':
                cdl_cfg.delay_spread = 300
                cdl_cfg.max_doppler_shift = 100
            case _:
                raise NotImplementedError("Unsupported CDL channel type")
        cdl_cfg.run_mode = run_mode
        if numpy_indicator:
            cdl_chan_numpy(cdl_cfg, n_tti, cuda_stream)
        else:
            cdl_chan_gpu_only(cdl_cfg, n_tti, cuda_stream)

    except Exception as e:
        assert False, f"Error running CDL channel test: {e}"
