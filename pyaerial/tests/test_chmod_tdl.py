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


def tdl_chan_numpy(tdl_cfg, n_tti, cuda_stream):
    """Test TDL channel using numpy for Tx input signal, save data into H5 files."""
    # Step 1: create random input signals using numpy
    tx_signal_size = tdl_cfg.n_cell * tdl_cfg.n_ue * \
        tdl_cfg.signal_length_per_ant * tdl_cfg.n_bs_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
    else:
        tx_signal_in = None

    # Step 2: create TDL channel object
    tdl = pycuphy.TdlChan(
        tdl_cfg=tdl_cfg,
        tx_signal_in_cpu=tx_signal_in,
        rand_seed=0,
        stream_handle=cuda_stream
    )

    # Step 3: run test and save H5 file(s)
    rx_signal_size = tdl_cfg.n_cell * tdl_cfg.n_ue * \
        tdl_cfg.signal_length_per_ant * tdl_cfg.n_ue_ant
    if rx_signal_size > 0:
        rx_signal_out = np.empty(rx_signal_size, dtype=np.complex64)
    else:
        rx_signal_out = None
    for tti_idx in range(0, n_tti):
        # generate input signal
        if tx_signal_size > 0:
            tx_signal_in.real = np.random.randn(tx_signal_size)
            tx_signal_in.imag = np.random.randn(tx_signal_size)

        # run TDL test
        tdl.run(
            tx_freq_signal_in_cpu=tx_signal_in,
            rx_freq_signal_out_cpu=rx_signal_out,
            ref_time0=tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_numpy"
            tdl.save_tdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"tdlChan_{tdl_cfg.n_cell}cell{tdl_cfg.n_ue}Ue_"
                f"{tdl_cfg.n_bs_ant}x{tdl_cfg.n_ue_ant}_"
                f"{tdl_cfg.delay_profile}{int(tdl_cfg.delay_spread)}_"
                f"dopp{int(tdl_cfg.max_doppler_shift)}_"
                f"cfo{int(tdl_cfg.cfo_hz)}_"
                f"runMode{tdl_cfg.run_mode}_"
                f"freqConvert{tdl_cfg.freq_convert_type}_"
                f"scSampling{tdl_cfg.sc_sampling}_"
                f"FP32_swap0{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            os.remove(out_filename)


def tdl_chan_gpu_only(tdl_cfg, n_tti, cuda_stream):
    """Test TDL channel using GPU memory for tx input signal, save data into H5 files."""

    # Step 1: create buffer for random input signals and obtain GPU memory address
    tx_signal_size = tdl_cfg.n_cell * tdl_cfg.n_ue * \
        tdl_cfg.signal_length_per_ant * tdl_cfg.n_bs_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
        # Allocate GPU memory
        err, tx_signal_in_gpu = runtime.cudaMalloc(tx_signal_in.nbytes)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate GPU memory: {err}")
    else:
        tx_signal_in = None
        tx_signal_in_gpu = 0

    # Step 2: create TDL channel object
    tdl = pycuphy.TdlChan(
        tdl_cfg=tdl_cfg,
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

        # run TDL test
        tdl.run(tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_gpuOnly"
            tdl.save_tdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"tdlChan_{tdl_cfg.n_cell}cell{tdl_cfg.n_ue}Ue_"
                f"{tdl_cfg.n_bs_ant}x{tdl_cfg.n_ue_ant}_"
                f"{tdl_cfg.delay_profile}{int(tdl_cfg.delay_spread)}_"
                f"dopp{int(tdl_cfg.max_doppler_shift)}_"
                f"cfo{int(tdl_cfg.cfo_hz)}_"
                f"runMode{tdl_cfg.run_mode}_"
                f"freqConvert{tdl_cfg.freq_convert_type}_"
                f"scSampling{tdl_cfg.sc_sampling}_"
                f"FP32_swap0{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            os.remove(out_filename)

    # Cleanup GPU memory
    if tx_signal_size > 0:
        err = runtime.cudaFree(tx_signal_in_gpu)
        if err != (runtime.cudaError_t.cudaSuccess,):
            print(f"Warning: Failed to free GPU memory: {err}")


@pytest.mark.parametrize(
    "tdl_type, n_tti, run_mode, numpy_indicator", [
        ('A', 100, 0, 0),
        ('A', 100, 0, 1),
        ('A', 100, 1, 0),
        ('A', 100, 1, 1),
        ('C', 100, 0, 0),
        ('C', 100, 0, 1),
        ('C', 100, 1, 0),
        ('C', 100, 1, 1)
    ])
def test_tdl_chan(tdl_type, n_tti, run_mode, numpy_indicator, cuda_stream):
    """Main test function of TDL channel

    Paremeters:
    - tdl_type: The type of TDL channel to use for the test
        - 'A' - TDLA30-10, 'B' - TDLB100-400, 'C' - TDLC300-100
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
        # TDL configuration
        tdl_cfg = pycuphy.TdlConfig()
        tdl_cfg.delay_profile = tdl_type
        match tdl_type:
            case 'A':
                tdl_cfg.delay_spread = 30
                tdl_cfg.max_doppler_shift = 10
            case 'B':
                tdl_cfg.delay_spread = 100
                tdl_cfg.max_doppler_shift = 400
            case 'C':
                tdl_cfg.delay_spread = 300
                tdl_cfg.max_doppler_shift = 100
            case _:
                raise NotImplementedError("Unsupported TDL channel type")
        tdl_cfg.run_mode = run_mode

        if numpy_indicator:
            tdl_chan_numpy(tdl_cfg, n_tti, cuda_stream)
        else:
            tdl_chan_gpu_only(tdl_cfg, n_tti, cuda_stream)

    except Exception as e:
        assert False, f"Error running TDL channel test: {e}"
