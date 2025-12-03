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

"""pyAerial library - fading channel."""

# pylint: disable=no-member
from itertools import product
import numpy as np
from cuda.bindings import runtime  # type: ignore
from aerial import pycuphy


class FadingChan:
    """Fading channel class.

    This class implements the fading channel that processes the frequency
    Tx samples and outputs frequency Rx samples. It includes OFDM modulation,
    tapped delay line (TDL) channel, OFDM demodulation, and adds noise based on
    input SNR.
    """
    def __init__(self,
                 *,
                 cuphy_carrier_prms: pycuphy.CuphyCarrierPrms,  # type: ignore
                 tdl_cfg: pycuphy.TdlConfig = None,  # type: ignore
                 cdl_cfg: pycuphy.CdlConfig = None,  # type: ignore
                 fading_type: int = 1,  # 0: AWGN (TODO), 1：TDL, 2：CDL
                 freq_in: np.ndarray = None,
                 proc_sig_freq: bool = False,
                 disable_noise: bool = False,
                 rand_seed: int = 0) -> None:
        """
        Initialize the FadingChan class.

        - cuphy_carrier_prms: carrier parameters for the channel
        - tdl_config: configuration of TDL channel
        - cdl_config: configuration of CDL channel
        - fading_type: 0: AWGN, 1: TDL, 2: CDL
        - freq_in: input frequency tx
        - proc_sig_freq: processing signal in freq domain
          will use the CFR from TDL class to process data on frequency domain.
          This mode may be inaccurate if CFO presents.
        - disable_noise: disable additive Gaussian noise
        - rand_seed: random seed for TDL/CDL channel generation
        """
        # Step 1: carrier and TDL configurations.
        # Create CUDA stream
        err, self.stream = runtime.cudaStreamCreate()
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create CUDA stream: {err}")
        self.prach = 0  # TODO: Add support for PRACH
        # save network setting
        self.n_bs_layer = cuphy_carrier_prms.n_bs_layer
        self.n_ue_layer = cuphy_carrier_prms.n_ue_layer
        self.n_symbol_slot = cuphy_carrier_prms.n_symbol_slot
        # TDL configurations, default is TDLA30-5-Low
        if fading_type == 1:
            self.fast_fading_cfg = tdl_cfg
            self.n_bs_ant = tdl_cfg.n_bs_ant
            self.n_ue_ant = tdl_cfg.n_ue_ant
            assert self.n_bs_layer <= self.n_bs_ant, "n_bs_layer should be <= n_bs_ant"
            assert self.n_ue_layer <= self.n_ue_ant, "n_ue_layer should be <= n_ue_ant"
            self.fast_fading_cfg.n_bs_ant = self.n_bs_layer
            self.fast_fading_cfg.n_ue_ant = self.n_ue_layer

        elif fading_type == 2:
            self.fast_fading_cfg = cdl_cfg
            self.n_bs_ant = np.prod(cdl_cfg.bs_ant_size)
            self.n_ue_ant = np.prod(cdl_cfg.ue_ant_size)
            # TODO: for now n_bs_layer and n_ue_layer should be equal to n_bs_ant and n_ue_ant
            assert self.n_bs_layer == self.n_bs_ant, "n_bs_layer should be equal to n_bs_ant"
            assert self.n_ue_layer == self.n_ue_ant, "n_ue_layer should be equal to n_ue_ant"

        else:
            raise ValueError("Invalid fading type")

        self.fading_type = fading_type
        self.disable_noise = disable_noise
        self.proc_sig_freq = proc_sig_freq
        # check numerology match
        self.mu = cuphy_carrier_prms.mu
        self.tti_len = 1e-3 / pow(2, self.mu)
        assert self.fast_fading_cfg.sc_spacing_hz == 15e3 * pow(2, self.mu)
        assert cuphy_carrier_prms.f_samp == 4096 * self.fast_fading_cfg.sc_spacing_hz
        assert self.fast_fading_cfg.f_samp == 4096 * self.fast_fading_cfg.sc_spacing_hz
        self.freq_in = freq_in  # Save the numpy buffer location.

        # Allocate DL output buffer data size match with config.
        self.freq_data_out_size_dl = [
            self.fast_fading_cfg.n_cell,
            self.fast_fading_cfg.n_ue,
            cuphy_carrier_prms.n_ue_layer,
            cuphy_carrier_prms.n_symbol_slot,
            cuphy_carrier_prms.n_sc
        ]
        self.freq_out_noise_free_dl = np.empty(
            self.freq_data_out_size_dl, dtype=np.complex64)
        self.freq_out_noisy_dl = np.empty(
            self.freq_data_out_size_dl, dtype=np.complex64)

        # Allocate UL output buffer data size match with config.
        self.freq_data_out_size_ul = [
            self.fast_fading_cfg.n_cell,
            self.fast_fading_cfg.n_ue,
            cuphy_carrier_prms.n_bs_layer,
            cuphy_carrier_prms.n_symbol_slot,
            cuphy_carrier_prms.n_sc
        ]
        self.freq_out_noise_free_ul = np.empty(
            self.freq_data_out_size_ul, dtype=np.complex64)
        self.freq_out_noisy_ul = np.empty(
            self.freq_data_out_size_ul, dtype=np.complex64)

        # Step 2: Create OFDM modulation, TDL channel, OFDM demodulation
        if self.proc_sig_freq:
            if self.fast_fading_cfg.cfo_hz > 0:
                print("Warning: channel may be inaccurate due to CFO")

            # Create OFDM modulation. To get batch info
            self.ofdm_mod = pycuphy.OfdmModulate(
                cuphy_carrier_prms=cuphy_carrier_prms,
                freq_data_in_cpu=self.freq_in,
                stream_handle=self.stream
            )

            self.fast_fading_cfg.signal_length_per_ant = (
                cuphy_carrier_prms.n_symbol_slot * cuphy_carrier_prms.n_sc
            )
            self.fast_fading_cfg.proc_sig_freq = 1  # proc tx signal in freq domain
            # Only create TDL or CDL channel
            self.fast_fading_cfg.batch_len = self.ofdm_mod.get_each_symbol_len_with_cp()

            if self.fading_type == 1:
                self.fast_fading = pycuphy.TdlChan(
                    tdl_cfg=self.fast_fading_cfg,
                    tx_signal_in_cpu=self.freq_in,
                    rand_seed=rand_seed,
                    stream_handle=self.stream
                )
            elif self.fading_type == 2:
                self.fast_fading = pycuphy.CdlChan(
                    cdl_cfg=self.fast_fading_cfg,
                    tx_signal_in_cpu=self.freq_in,
                    rand_seed=rand_seed,
                    stream_handle=self.stream
                )

        else:
            # Create OFDM modulation.
            self.ofdm_mod = pycuphy.OfdmModulate(
                cuphy_carrier_prms=cuphy_carrier_prms,
                freq_data_in_cpu=self.freq_in,
                stream_handle=self.stream
            )

            self.fast_fading_cfg.signal_length_per_ant = int(
                self.ofdm_mod.get_time_data_length() / cuphy_carrier_prms.n_bs_layer
            )  # Input data length per antenna.
            self.tx_signal_in_gpu = self.ofdm_mod.get_time_data_out()

            # Create TDL or CDL channel.
            self.fast_fading_cfg.batch_len = self.ofdm_mod.get_each_symbol_len_with_cp()
            if fading_type == 1:
                self.fast_fading = pycuphy.TdlChan(
                    tdl_cfg=self.fast_fading_cfg,
                    tx_signal_in_gpu=self.tx_signal_in_gpu,
                    rand_seed=rand_seed,
                    stream_handle=self.stream
                )
            elif fading_type == 2:
                self.fast_fading = pycuphy.CdlChan(
                    cdl_cfg=self.fast_fading_cfg,
                    tx_signal_in_gpu=self.tx_signal_in_gpu,
                    rand_seed=rand_seed,
                    stream_handle=self.stream
                )

            # Create OFDM demodulation.
            rx_signal_out_gpu = self.fast_fading.get_rx_signal_out()
            self.ofdm_demod = pycuphy.OfdmDeModulate(
                cuphy_carrier_prms=cuphy_carrier_prms,
                time_data_in_gpu=rx_signal_out_gpu,
                freq_data_out_cpu=self.freq_out_noise_free_dl,
                prach=self.prach,
                per_ant_samp=False,
                stream_handle=self.stream
            )

            # Create OFDM demodulation for dumping genie channel
            # assign per rx-tx pair sample location
            self.rx_time_ant_pair_signal_out_gpu = (
                self.fast_fading.get_rx_time_ant_pair_signal_out()
            )
            self.freq_data_out_ant_pair_size = [
                cuphy_carrier_prms.n_ue_layer,
                cuphy_carrier_prms.n_bs_layer,
                cuphy_carrier_prms.n_symbol_slot,
                cuphy_carrier_prms.n_sc
            ]
            self.freq_out_ant_pair_noise_free = np.empty(
                self.freq_data_out_ant_pair_size, dtype=np.complex64)
            self.per_ant_samp_direction = 0  # 0: downlink, 1: uplink

            if self.fast_fading_cfg.save_ant_pair_sample != 0:
                self.ofdm_demod_ant_pair = pycuphy.OfdmDeModulate(
                    cuphy_carrier_prms=cuphy_carrier_prms,
                    time_data_in_gpu=self.rx_time_ant_pair_signal_out_gpu,
                    freq_data_out_cpu=self.freq_out_ant_pair_noise_free,
                    prach=self.prach,
                    per_ant_samp=True,
                    stream_handle=self.stream
                )
            else:
                self.ofdm_demod_ant_pair = None

        if not self.disable_noise:
            self.gau_noise_adder = pycuphy.GauNoiseAdder(
                num_threads=1024,  # 4 thread blocks, each with 256 threads
                rand_seed=rand_seed,
                stream_handle=self.stream
            )

        # create numpy arraries for dumping CFR on PRBG or SC
        # numpy array for CFR on PRBG
        n_prbg = int(np.ceil(self.fast_fading_cfg.n_sc / self.fast_fading_cfg.n_sc_prbg))
        self.cfr_prbg = np.empty(
            (self.fast_fading_cfg.n_cell, self.fast_fading_cfg.n_ue, self.n_symbol_slot,
             self.n_ue_ant, self.n_bs_ant, n_prbg),
            dtype=np.complex64
        )

        # numpy array for CFR on Sc
        self.cfr_sc = np.empty(
            (self.fast_fading_cfg.n_cell, self.fast_fading_cfg.n_ue, self.n_symbol_slot,
             self.n_ue_ant, self.n_bs_ant, self.fast_fading_cfg.n_sc),
            dtype=np.complex64
        )

    def __del__(self) -> None:
        """Destructor to clean up CUDA stream."""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                err = runtime.cudaStreamDestroy(self.stream)
                if err != (runtime.cudaError_t.cudaSuccess,):
                    # Don't raise in destructor, just print warning
                    print(f"Warning: Failed to destroy CUDA stream: {err}")
            except RuntimeError as runtime_error:
                # Don't raise in destructor, just print warning
                print(f"Warning: Exception during CUDA stream cleanup: {runtime_error}")
            except ImportError as import_error:
                # Handle case where runtime module might not be available during cleanup
                print(f"Warning: Runtime module unavailable during cleanup: {import_error}")

    def add_noise_with_snr(
        self,
        snr_db: float,
        enable_swap_tx_rx: bool = False
    ) -> np.ndarray:
        """Add Gaussian noise to a complex signal with a specified SNR.

        Args:
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.
            enable_swap_tx_rx (bool): Swap tx and rx to simulate UL channel using DL class.

        Returns:
            np.ndarray: The frequency-domain signal with noise added.
        """
        if self.disable_noise:
            return self.freq_out_noise_free_ul if enable_swap_tx_rx else self.freq_out_noise_free_dl

        if enable_swap_tx_rx:
            noisy_signal = self.freq_out_noisy_ul
        else:
            noisy_signal = self.freq_out_noisy_dl

        if self.proc_sig_freq:
            d_signal = self.fast_fading.get_rx_signal_out()
        else:
            d_signal = self.ofdm_demod.get_freq_data_out()

        self.gau_noise_adder.add_noise(
            noisy_signal=noisy_signal,
            d_signal=d_signal,
            signal_size=noisy_signal.size,
            snr_db=snr_db
        )

        return noisy_signal

    def add_noise_with_snr_numpy(
        self,
        snr_db: float,
        enable_swap_tx_rx: bool = False
    ) -> np.ndarray:
        """Add Gaussian noise to a complex signal with a specified SNR.

        Args:
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.
            enable_swap_tx_rx (bool): Swap tx and rx to simulate UL channel using DL class.
            This function is GPU-accelerated by add_noise_with_snr() and pycuphy.GauNoiseAdder

        Returns:
            np.ndarray: The frequency-domain signal with noise added.
        """
        if self.disable_noise:
            return self.freq_out_noise_free_ul if enable_swap_tx_rx else self.freq_out_noise_free_dl

        # convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        # Generate Gaussian noise for both real and imaginary parts.
        if enable_swap_tx_rx:  # UL
            noise_real = np.sqrt(0.5 / snr_linear) * \
                np.random.randn(*self.freq_out_noise_free_ul.shape)
            noise_imag = np.sqrt(0.5 / snr_linear) * \
                np.random.randn(*self.freq_out_noise_free_ul.shape)
            noise = noise_real + 1j * noise_imag
            # Add noise to the signal.
            self.freq_out_noisy_ul = self.freq_out_noise_free_ul + noise
            return self.freq_out_noisy_ul

        # DL
        noise_real = np.sqrt(0.5 / snr_linear) * \
            np.random.randn(*self.freq_out_noise_free_dl.shape)
        noise_imag = np.sqrt(0.5 / snr_linear) * \
            np.random.randn(*self.freq_out_noise_free_dl.shape)
        noise = noise_real + 1j * noise_imag
        # Add noise to the signal.
        self.freq_out_noisy_dl = self.freq_out_noise_free_dl + noise
        return self.freq_out_noisy_dl

    def dump_channel(self,
                     freq_in: np.ndarray = None,
                     enable_swap_tx_rx: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Dump TDL channel to numpy arrays.

        Returns:
            tuple: A tuple containing two numpy arrays (cfr_sc, cfr_prbg).
        """
        # Only dump CFR for now
        #     self.fast_fading.dump_cir(self.cir)

        self.fast_fading.dump_cfr_prbg(self.cfr_prbg)

        # if skip OFDM, cfr on sc is already dumped in dump_process_tx
        if not self.proc_sig_freq:  # CFR on SC not saved
            if self.fast_fading_cfg.cfo_hz == 0 or self.ofdm_demod_ant_pair is None:
                self.fast_fading.dump_cfr_sc(self.cfr_sc)
            else:
                self.get_genie_channel(freq_in, self.cfr_sc, enable_swap_tx_rx)

        # self.fast_fading.save_tdl_chan_to_h5_file() # save TDL channel, for debugging purpose
        # self.fast_fading.save_cdl_chan_to_h5_file() # save CDL channel, for debugging purpose
        runtime.cudaStreamSynchronize(self.stream)
        return self.cfr_sc, self.cfr_prbg

    def get_genie_channel(self,
                          freq_in: np.ndarray = None,
                          cfr_sc: np.ndarray = None,
                          enable_swap_tx_rx: bool = False
                          ) -> None:
        """
        need to do ofdm demodulation of the rx-tx ant pair sample to get genie channel
        """
        self.ofdm_demod_ant_pair.run()  # stream synchronize inside self.ofdm_demod_ant_pair.run()
        # If CFO present, we will use output samples to calculate genie channel
        # we will iterate the tx samples per antenna and get the channel per tx-rx antenna pair

        for cell_idx, ue_idx, symbol_idx, ue_ant_idx, bs_ant_idx in product(
            range(self.fast_fading_cfg.n_cell),
            range(self.fast_fading_cfg.n_ue),
            range(self.n_symbol_slot),
            range(self.n_ue_ant),
            range(self.n_bs_ant)
        ):
            if enable_swap_tx_rx is not self.per_ant_samp_direction:
                self.freq_out_ant_pair_noise_free = np.transpose(
                    self.freq_out_ant_pair_noise_free,
                    (1, 0, 2, 3)
                )
                self.per_ant_samp_direction = not self.per_ant_samp_direction

            if enable_swap_tx_rx:
                if freq_in is not None:
                    freq_in_value = freq_in[cell_idx][ue_idx][ue_ant_idx][symbol_idx]
                else:
                    freq_in_value = self.freq_in[cell_idx][ue_idx][ue_ant_idx][symbol_idx]

                freq_out = (
                    self.freq_out_ant_pair_noise_free[bs_ant_idx][ue_ant_idx][symbol_idx]
                )
            else:
                if freq_in is not None:
                    freq_in_value = freq_in[cell_idx][ue_idx][bs_ant_idx][symbol_idx]
                else:
                    freq_in_value = self.freq_in[cell_idx][ue_idx][bs_ant_idx][symbol_idx]

                freq_out = (
                    self.freq_out_ant_pair_noise_free[ue_ant_idx][bs_ant_idx][symbol_idx]
                )
            # Assign the channel to cfr_sc
            cfr_sc[cell_idx][ue_idx][symbol_idx][ue_ant_idx][bs_ant_idx] = freq_out / freq_in_value

    def reset(self,
              ) -> None:
        """
        Reset the fading channel.
        """
        self.fast_fading.reset()

    def run(self,
            *,
            tti_idx: int,
            snr_db: float,
            enable_swap_tx_rx: bool = False,
            tx_column_major_ind: bool = False,
            freq_in: np.ndarray = None
            ) -> np.ndarray:
        """
        Run the fading channel.

        Args:
            tti_idx (int): TTI index.
            snr_db (float): Signal-to-Noise Ratio in dB.
            enable_swap_tx_rx (bool): Swap tx and rx to simulate UL channel using DL class.
            freq_in (np.ndarray): Frequency domain input samples.

        Returns:
            np.ndarray: Frequency domain samples after channel processing.
        """
        if self.proc_sig_freq:
            # time stamp, assuming 500 us per slot
            if enable_swap_tx_rx:
                rx_freq_signal_out_cpu = self.freq_out_noise_free_ul
            else:
                rx_freq_signal_out_cpu = self.freq_out_noise_free_dl
            self.fast_fading.run(
                tx_freq_signal_in_cpu=self.freq_in if freq_in is None else freq_in,
                rx_freq_signal_out_cpu=rx_freq_signal_out_cpu,
                ref_time0=tti_idx * self.tti_len,
                enable_swap_tx_rx=enable_swap_tx_rx,
                tx_column_major_ind=tx_column_major_ind
            )
            # optional debug output
            # self.fast_fading.save_tdl_chan_to_h5_file()
            # self.fast_fading.save_cdl_chan_to_h5_file()
            # self.fast_fading.print_time_chan()

            # Dump CFR on subcarrier level
            self.fast_fading.dump_cfr_sc(self.cfr_sc)
            runtime.cudaStreamSynchronize(self.stream)
        else:
            if freq_in is not None:  # using new array as input
                self.ofdm_mod.run(
                    freq_in=freq_in,
                    enable_swap_tx_rx=enable_swap_tx_rx
                )
            else:
                self.ofdm_mod.run(
                    enable_swap_tx_rx=enable_swap_tx_rx
                )
            # time stamp, self.tti_len per slot
            self.fast_fading.run(
                ref_time0=tti_idx * self.tti_len,
                tx_column_major_ind=tx_column_major_ind,
                enable_swap_tx_rx=enable_swap_tx_rx
            )
            self.ofdm_demod.run(
                freq_data_out_cpu=(
                    self.freq_out_noise_free_ul
                    if enable_swap_tx_rx
                    else self.freq_out_noise_free_dl
                ),
                enable_swap_tx_rx=enable_swap_tx_rx
            )  # stream synchronize inside ofdm_demod.run()
        # add noise in freq domain
        return self.add_noise_with_snr(snr_db, enable_swap_tx_rx)
