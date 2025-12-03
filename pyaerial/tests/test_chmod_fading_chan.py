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
import matplotlib.pyplot as plt
from aerial.phy5g.chan_models import FadingChan


def freq_out_ref_check(
    cuphy_carrier_prms: pycuphy.CuphyCarrierPrms,
    sim_params: list,
    freq_in: np.ndarray,
    freq_out: np.ndarray,
    cfr_sc: np.ndarray,
    enable_swap_tx_rx: bool = False
) -> float:
    """
    Calculate and return the SNR in decibels.

    Parameters:
    cuphy_carrier_prms: The carrier parameters.
    sim_params: The simulation parameters.
    freq_in: The input frequency.
    freq_out: The output frequency.
    cfr_sc: The channel frequency response.
    enable_swap_tx_rx: A flag to enable swapping TX and RX.

    Returns:
    The SNR in decibels.
    """
    freq_out_ref = np.zeros_like(freq_out)  # same dim with freq_out
    [n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc] = sim_params
    for cell_idx in range(n_cell):
        for ue_idx in range(n_ue):
            for ofdm_sym_idx in range(cuphy_carrier_prms.n_symbol_slot):
                for ue_ant_idx in range(n_ue_ant):
                    for bs_ant_idx in range(n_bs_ant):
                        tmp_chan = cfr_sc[cell_idx][ue_idx][ofdm_sym_idx][ue_ant_idx][bs_ant_idx]
                        if enable_swap_tx_rx:  # UL
                            freq_out_ref[cell_idx][ue_idx][bs_ant_idx][ofdm_sym_idx] += (
                                freq_in[cell_idx][ue_idx][ue_ant_idx][ofdm_sym_idx] * tmp_chan
                            )
                        else:  # DL
                            freq_out_ref[cell_idx][ue_idx][ue_ant_idx][ofdm_sym_idx] += (
                                freq_in[cell_idx][ue_idx][bs_ant_idx][ofdm_sym_idx] * tmp_chan
                            )

    # Calculate noise (difference between noisy and reference signals)
    noise = freq_out - freq_out_ref
    # Calculate signal power (mean squared magnitude of reference signal)
    signal_power = np.mean(np.abs(freq_out_ref)**2)
    # Calculate noise power (mean squared magnitude of noise)
    noise_power = np.mean(np.abs(noise)**2)
    # Calculate SNR in decibels
    snr_db = 1000 if noise_power == 0 else 10 * \
        np.log10(signal_power / noise_power)

    return snr_db


def plot_snr_hist(snrs: np.ndarray) -> None:
    """
    Plot CDF of input SNRs, save the plot into a PNG file snr_cdf_plot.png
    """
    # Sort data
    snr_sorted = np.sort(snrs)

    # Compute CDF
    cdf = np.arange(1, len(snr_sorted) + 1) / len(snr_sorted)

    # Plot CDF
    plt.figure(figsize=(8, 6))
    plt.plot(snr_sorted, cdf, marker='.', linestyle='none')
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.savefig("snr_cdf_plot.png")
    plt.show()


@pytest.mark.parametrize(
    "n_sc, tdl_cdl_type, n_tti, fading_type, disable_noise", [
        (1632, 'A', 100, 1, True),
        (1632, 'C', 100, 1, True),
        (3276, 'A', 100, 1, True),
        (3276, 'C', 100, 1, True),
        (1632, 'A', 100, 1, False),
        (1632, 'C', 100, 1, False),
        (3276, 'A', 100, 1, False),
        (3276, 'C', 100, 1, False),
        (1632, 'A', 100, 2, True),
        (1632, 'C', 100, 2, True),
        (3276, 'A', 100, 2, True),
        (3276, 'C', 100, 2, True),
        (1632, 'A', 100, 2, False),
        (1632, 'C', 100, 2, False),
        (3276, 'A', 100, 2, False),
        (3276, 'C', 100, 2, False)
    ]
)
def test_fading_chan(n_sc, tdl_cdl_type, n_tti, disable_noise, fading_type, snr_db=10):
    """
    Test the fading channel model with specified parameters.

    - n_sc: number of subcarriers
    - tdl_cdl_type: TDL/CDL channel model type (e.g., 'A', 'B', 'C')
    - n_tti: number of TTIs in test
    - disable_noise: disable noise addition for test purpose (default: False)
    - fading_type: 0 - AWGN (not supported yet); 1 - TDL; 2 - CDL
    - snr_db: SNR in dB
    """
    try:
        # carrier parameters and tdl configurations
        cuphy_carrier_prms = pycuphy.CuphyCarrierPrms()

        if fading_type == 1:  # TDL
            tdl_cfg = pycuphy.TdlConfig()
            tdl_cfg.delay_profile = tdl_cdl_type
            match tdl_cdl_type:
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
            # channel configurations
            tdl_cfg.cfo_hz = 0
            tdl_cfg.delay = 1e-6

            # antennas, N_sc
            cuphy_carrier_prms.n_bs_layer = 4
            cuphy_carrier_prms.n_ue_layer = 4
            cuphy_carrier_prms.n_sc = n_sc
            tdl_cfg.n_bs_ant = cuphy_carrier_prms.n_bs_layer
            tdl_cfg.n_ue_ant = cuphy_carrier_prms.n_ue_layer
            tdl_cfg.n_sc = cuphy_carrier_prms.n_sc

            # run mode of TDL
            tdl_cfg.run_mode = 2  # TDL time channel, filter tx signal, generate gennie channel
            # with CFO, gennie channel generation has to go through per-antenna pair samples
            tdl_cfg.save_ant_pair_sample = tdl_cfg.cfo_hz > 0

            # number of cells, number of UEs, number of BS antennas,
            # number of UEs antennas, number of subcarriers
            n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc = (
                tdl_cfg.n_cell,
                tdl_cfg.n_ue,
                tdl_cfg.n_bs_ant,
                tdl_cfg.n_ue_ant,
                tdl_cfg.n_sc
            )
            cdl_cfg = None

        else:  # CDL
            cdl_cfg = pycuphy.CdlConfig()
            cdl_cfg.delay_profile = tdl_cdl_type
            match tdl_cdl_type:
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
            # channel configurations
            cdl_cfg.cfo_hz = 0
            cdl_cfg.delay = 1e-6

            # antennas, N_sc
            cuphy_carrier_prms.n_sc = n_sc
            # Modify ue_ant_size
            cdl_cfg.ue_ant_size = [1, 1, 1, 1, 1]
            cdl_cfg.ue_ant_pattern = 0
            cdl_cfg.ue_ant_polar_angles = [0, 90]

            # need to match ant_size and layer
            n_bs_ant = np.prod(cdl_cfg.bs_ant_size)  # number of BS antennas
            n_ue_ant = np.prod(cdl_cfg.ue_ant_size)  # number of UE antennas
            cuphy_carrier_prms.n_bs_layer = n_bs_ant
            cuphy_carrier_prms.n_ue_layer = n_ue_ant
            cdl_cfg.n_sc = cuphy_carrier_prms.n_sc
            # run mode of cdl
            cdl_cfg.run_mode = 2  # cdl time channel, filter tx signal, generate gennie channel
            # with CFO, gennie channel generation has to go through per-antenna pair samples
            cdl_cfg.save_ant_pair_sample = cdl_cfg.cfo_hz > 0

            [n_cell, n_ue, n_sc] = [cdl_cfg.n_cell, cdl_cfg.n_ue, cdl_cfg.n_sc]
            tdl_cfg = None

        # testing config
        proc_sig_freq = True  # process signal in frequency
        enable_swap_tx_rx = True  # enable swapping tx and rx

        # allocate numpy buffers for freq in and freq out samples
        freq_data_in_size = [n_cell,
                             n_ue,
                             cuphy_carrier_prms.n_ue_layer
                             if enable_swap_tx_rx
                             else cuphy_carrier_prms.n_bs_layer,
                             cuphy_carrier_prms.n_symbol_slot,
                             cuphy_carrier_prms.n_sc
                             ]
        freq_in = np.empty(freq_data_in_size, dtype=np.complex64)

        # create FadingChan object
        fading_chan = FadingChan(
            cuphy_carrier_prms=cuphy_carrier_prms,
            tdl_cfg=tdl_cfg,
            cdl_cfg=cdl_cfg,
            fading_type=fading_type,
            freq_in=freq_in,
            proc_sig_freq=proc_sig_freq,
            disable_noise=disable_noise,
            rand_seed=0
        )

        # run FadingChan object with specific TTI index and SNR (dB)
        snr_empirical = np.zeros(n_tti)
        for tti_idx in range(0, n_tti):
            # generate freq in data using numpy
            normalize_factor = 1 / \
                np.sqrt(
                    2 * (cuphy_carrier_prms.n_ue_layer
                         if enable_swap_tx_rx else cuphy_carrier_prms.n_bs_layer)
                )
            freq_in.real = np.random.randn(
                *freq_data_in_size) * normalize_factor
            freq_in.imag = np.random.randn(
                *freq_data_in_size) * normalize_factor
            # run fading channel
            # fading_chan.reset()  # optional reset fading channel
            freq_out = fading_chan.run(
                tti_idx=tti_idx,
                snr_db=snr_db,
                enable_swap_tx_rx=enable_swap_tx_rx,
                tx_column_major_ind=False
                # or freq_in=freq_in_new with a numpy array (to be created)
            )
            assert freq_out.size > 0, "freq_out is empty"

            cfr_sc, cfr_prbg = fading_chan.dump_channel(
                # or freq_in=freq_in_new with a numpy array (to be created)
            )

            snr_empirical[tti_idx] = freq_out_ref_check(
                cuphy_carrier_prms=cuphy_carrier_prms,
                sim_params=[n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc],
                # or freq_in_new with a numpy array (to be created)
                freq_in=freq_in,
                freq_out=freq_out,
                cfr_sc=cfr_sc,
                enable_swap_tx_rx=enable_swap_tx_rx
            )

        # Print avg, min, max in one line
        print(
            f"Fading channel test with average SNR: {np.mean(snr_empirical):.2f} dB, "
            f"Min SNR: {np.min(snr_empirical):.2f} dB, "
            f"Max SNR: {np.max(snr_empirical):.2f} dB"
        )
        # plot the CDF of SNRs
        # plot_snr_hist(snr_empirical)

    except Exception as e:
        assert False, f"Error running fading channel test: {e}"
