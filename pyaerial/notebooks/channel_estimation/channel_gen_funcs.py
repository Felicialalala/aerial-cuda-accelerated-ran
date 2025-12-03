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

import sionna
import numpy as np
import tensorflow as tf

from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.ldpc import get_mcs, random_tb
from aerial.util.cuda import get_cuda_stream

# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus)


class SionnaChannelGenerator(tf.keras.Model):
    """
    Generator class for Sionna channels.
    """
    def __init__(self, num_prbs: int, channel_name: str = 'UMa', batch_size: int = 1):
        """
        Initializor for a Sionna Channel Generator. It defines an anntena array and
        a resource grid in order to generate channels conveniently.

        For simplicity, we currently hardcode for single user, single antenna, single layer,
        and several other parameters like frequency, delay spread, link direction, etc.
        """
        super().__init__()

        self.num_prbs = num_prbs
        self.batch_size = batch_size

        # parameters for channel modeling
        self.channel_model = channel_name
        self.fc = 3.5e9                 # Frequency [Hz]
        self.link_direction = 'uplink'  # Link direction (direction of the signal)
        self.delay_spread = 100e-9      # Nominal delay spread [s]
        self.ue_speed = 1               # User speed [m/s]

        single_ant = sionna.phy.channel.tr38901.PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=1,
            polarization='single',
            polarization_type='V',
            antenna_pattern='omni',
            carrier_frequency=self.fc)

        self.ue_array, self.gnb_array = single_ant, single_ant

        self.channel = self.make_channel()

        self.rg = sionna.phy.ofdm.ResourceGrid(
            num_ofdm_symbols=1,
            fft_size=self.num_prbs * 12,  # Num. subcarriers
            subcarrier_spacing=30e3,      # [kHz]
            num_tx=1,
            num_streams_per_tx=1)

        self.channel_generator = sionna.phy.channel.GenerateOFDMChannel(
            self.channel,
            self.rg,
            normalize_channel=True)

        self.awgn = sionna.phy.channel.AWGN()

        sionna.phy.config.xla_compat = True  # Necessary for proper model compilation

    def make_channel(self):
        """
        Create the channel object, using one of the 3GPP defined channel models.
        Available channel models in Sionna: https://nvlabs.github.io/sionna/api/channel.html
        This function further requires the batch size for the number of channels to sample,
        and transmit and receive arrays for the dimensions of the channel, and some
        information to determine the delay/doppler spread to include in the channel,
        like user speeds and carrier frequency.
        """
        rx_array = self.gnb_array if self.link_direction == 'uplink' else self.ue_array
        tx_array = self.ue_array if self.link_direction == 'uplink' else self.gnb_array

        num_rx_ant, num_tx_ant = rx_array.num_ant, tx_array.num_ant

        # Setup network topology (required in UMi, UMa, and RMa)
        if self.channel_model in ['UMi', 'UMa']:
            topology = sionna.phy.channel.gen_single_sector_topology(
                batch_size=self.batch_size,
                num_ut=1,
                scenario=self.channel_model.lower(),
                min_ut_velocity=self.ue_speed,
                max_ut_velocity=self.ue_speed,
            )

        # Configure a channel impulse reponse (CIR) generator for the channel models
        if self.channel_model == "Rayleigh":
            ch_model = sionna.phy.channel.RayleighBlockFading(
                num_rx=1,
                num_rx_ant=num_rx_ant,
                num_tx=1,
                num_tx_ant=num_tx_ant
            )
        elif "CDL" in self.channel_model:
            ch_model = sionna.phy.channel.tr38901.CDL(
                model=self.channel_model[-1],
                delay_spread=self.delay_spread,
                carrier_frequency=self.fc,
                ut_array=self.ue_array,
                bs_array=self.gnb_array,
                direction=self.link_direction,
                min_speed=self.ue_speed
            )
        elif 'UMi' in self.channel_model or 'UMa' in self.channel_model:
            if 'UMa' in self.channel_model:
                model = sionna.phy.channel.tr38901.UMa
            else:
                model = sionna.phy.channel.tr38901.UMi

            ch_model = model(carrier_frequency=self.fc,
                             o2i_model='low',
                             bs_array=self.gnb_array,
                             ut_array=self.ue_array,
                             direction=self.link_direction,
                             enable_pathloss=False)
            ch_model.set_topology(*topology, los=None)

        elif 'TDL' in self.channel_model:
            ch_model = sionna.phy.channel.tr38901.TDL(
                model=self.channel_model[-1],
                delay_spread={'A': 30e-9, 'B': 100e-9, 'C': 300e-9}[self.channel_model[-1]],
                carrier_frequency=self.fc,
                num_rx_ant=num_rx_ant,
                num_tx_ant=num_tx_ant
            )
        else:
            raise ValueError(f"Invalid channel model {self.channel_model}!")

        return ch_model

    @tf.function(jit_compile=True)
    def gen_channel_jit(self, snr_db):
        """ Sample channel and add noise based on an SNR value in dB. """
        h = self.channel_generator(self.batch_size)

        # Add noise
        No = tf.math.pow(10., tf.cast(-snr_db, tf.float32) / 10.)
        h_n = self.awgn(h, No)

        # Squeeze: tx/rx id, tx/rx ant id, ofdm symb dimensions
        return tf.squeeze(h, (1, 2, 3, 4, 5)), tf.squeeze(h_n, (1, 2, 3, 4, 5))


class PyAerialChannelEstimateGenerator():
    """
    A Generator Class for PyAerial Channel Estimates.

    This classes uses SionnaChannelGenerator
    Implements PyAerial channels estimators:
        - 'LS': Least Squares
        - 'MMSE': Minimum Mean Squared Error
        - 'MS MMSE': Multi-stage MMSE
    """
    def __init__(self, sionna_channel, batch=True):
        self.num_prbs = sionna_channel.num_prbs
        self.num_subcarriers = self.num_prbs * 12
        self.batch_size = 32 if batch else 1  # only works for these 2 values

        # DMRS parameters
        self.dmrs_params = dict(
            num_ues=1,                    # We simulate only one UE
            slot=0,                       # Slot number
            num_dmrs_cdm_grps_no_data=2,  # Number of DMRS CDM groups without data
            dmrs_scrm_id=41,              # DMRS scrambling ID
            start_prb=0,                  # Start PRB index
            num_prbs=self.num_prbs,       # Number of allocated PRBs
            dmrs_syms=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Binary list indicating
                                                                   # which symbols are DMRS
            start_sym=2,                  # Start symbol index
            num_symbols=12,               # Number of symbols in the UE group allocation.
            scids=[0],                    # DMRS scrambling ID
            layers=[1],                   # Number of layers per user
            dmrs_ports=[1],               # DMRS port
        )

        # DMRS parameters only used for channel estimation
        self.ch_est_dmrs_params = dict(
            prg_size=1,                   # PRG size
            num_ul_streams=1,             # Number of UL streams
            dmrs_max_len=1,               # 1: single-symbol DMRS. 2: double-symbol
            dmrs_add_ln_pos=0,            # Number of additional DMRS positions.
        )

        self.tx_tensor = self.make_tx_tensor()[:self.num_subcarriers]

        # Obtain the symbols with DMRS
        self.dmrs_idxs = self.dmrs_params['dmrs_syms'].index(1)

        self.rg = sionna.phy.ofdm.ResourceGrid(
            num_ofdm_symbols=14,  # Note: generating a single symbol does not speed up
            fft_size=self.num_prbs * 12,
            subcarrier_spacing=30e3,
            num_tx=1,
            num_streams_per_tx=1)

        self.mapper = sionna.phy.ofdm.ResourceGridMapper(self.rg)

        self.channel = sionna.phy.channel.OFDMChannel(
            channel_model=sionna_channel.channel,
            resource_grid=self.rg,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True
        )

        self.ls_estimator = self.create_channel_estimator('LS')
        self.mmse_estimator = self.create_channel_estimator('MS MMSE')

    def create_channel_estimator(self, estimator_type):
        """
        # Create the PyAerial (cuPHY) channel estimator with each algorithm:
            #- 0 - MMSE (legacy)
            #- 1 - Multi-stage MMSE with delay estimation (default)
            #- 2 - RKHS (not available yet)
            #- 3 - LS channel estimation only
        """
        assert estimator_type in ['MMSE', 'MS MMSE', 'LS']
        ch_est_algo = {'MMSE': 0, 'MS MMSE': 1, 'LS': 3}[estimator_type]

        return ChannelEstimator(num_rx_ant=1, ch_est_algo=ch_est_algo,
                                cuda_stream=get_cuda_stream())

    def make_tx_tensor(self, mcs=1, seed=42):
        """
        Creates a tensor containing the data to be transmitted.
        This tensor needs to be passed through the channel to obtain the received
        symbols and extract the DM-RS from those.
        """
        pusch_tx = PdschTx(cell_id=41, num_rx_ant=1, num_tx_ant=1)

        # Get modulation order and coderate.
        mod_order, coderate = get_mcs(mcs)  # MCS index (according to TS 38.214 tables)

        # Generate random bits for the transport block reused in each transmission
        np.random.seed(seed)
        tb_input = random_tb(mod_order=mod_order,
                             code_rate=coderate,
                             dmrs_syms=self.dmrs_params['dmrs_syms'],
                             num_prbs=self.num_prbs,
                             start_sym=self.dmrs_params['start_sym'],
                             num_symbols=self.dmrs_params['num_symbols'],
                             num_layers=self.dmrs_params['layers'][0])

        # Transmit PUSCH: Take transport blocks and ...
        # Input parameters are lists as the interface supports multiple UEs
        tx_tensor = pusch_tx.run(**self.dmrs_params,
                                 rntis=[1234],           # UE RNTI
                                 data_scids=[0],         # Data scrambling ID
                                 tb_inputs=[tb_input],   # Input transport block in bytes
                                 code_rates=[coderate],  # Code rate x 1024
                                 mod_orders=[mod_order]  # Modulation order
                                 )  # OUTPUT: subcarriers x symbols(time) x tx_antennas

        del pusch_tx  # Otherwise throws an error - with this throws only a warning

        return tx_tensor

    @tf.function(jit_compile=True)
    def apply_channel(self, snr):
        """Transmit the Tx tensor through the radio channel."""
        # Add batch and num_tx dimensions that Sionna expects and reshape.

        # (subcarriers, symbols, tx_ant)
        tx_tensor = tf.transpose(self.tx_tensor, (2, 1, 0))
        # (tx_ant, symbols, subcarriers)
        tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]
        # (1, num_tx=1, num_streams_per_tx=1, num_data_symbols = symbols * subcarriers)
        tx_tensor = tf.repeat(tx_tensor, self.batch_size, axis=0)  # Repeat Tensor across batches
        # (batch_size, num_tx=1, num_streams_per_tx=1, symbols, subcarriers)
        tx_tensor = self.mapper(tx_tensor)
        # (batch_size, num_tx=1, num_streams_per_tx=1, symbols, subcarriers)
        No = tf.math.pow(10., -tf.convert_to_tensor(snr) / 10.)
        rx_tensor, h = self.channel(tx_tensor, No)
        # rx : (batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size)
        # h  : (batch_size, num_tx, num_tx_ant, num_rx, num_rx_ant, num_ofdm_symbols, fft_size)
        h = tf.transpose(h[:, 0, 0, 0, ...], (0, 3, 2, 1))
        rx_tensor = tf.transpose(rx_tensor[:, 0, ...], (0, 3, 2, 1))
        # both: (batch_size, subcarriers, symb, rx_ant)
        return rx_tensor, h

    def __call__(self, snr):
        """ Object call function. When the PyAerialChannelEstimator is called, it
        creates a transmit tensor, passes it over a channel, extracts the received
        DM-RS from the received signal and performs channel estimation on it."""
        # Transmits tx_tensor over channel
        y, h = self.apply_channel(np.array([snr], dtype=np.float32))
        y, h = y.numpy(), h.numpy()

        # Get GT channel
        gt = h[:, :self.num_subcarriers, self.dmrs_idxs, 0]

        ls = np.zeros((self.batch_size, int(self.num_subcarriers / 2)), dtype=np.complex64)
        mmse = np.zeros((self.batch_size, self.num_subcarriers), dtype=np.complex64)
        for b in range(self.batch_size):
            # Get LS and MMSE channel estimates
            est_param = {'rx_slot': y[b], **self.dmrs_params, **self.ch_est_dmrs_params}
            ls[b] = self.ls_estimator.estimate(**est_param)[0].swapaxes(0, 2).squeeze() / np.sqrt(2)
            mmse[b] = self.mmse_estimator.estimate(**est_param)[0].squeeze()

        return ls, mmse, gt


def sionna_to_pyaerial_shape(ch, n_sub: int, interp: int = 2, n_symb: int = 2,
                             n_ant: int = 4, n_layers: int = 4, est_type: str = 'ls'):
    """
    Transforms a [batch, sub] array generated by Sionna into
    - LS:      [batch,  subcarrier, layers,    rx antennas,      symbols]
    - MMSE/GT: [batch, rx antennas, layers, interp * subcarrier, symbols]
    The choice of which shape is selected depends on <est_type>

    interp = 2   # interpolation factor from LS to MMSE (= comb size)
    n_symb = 2   # num of dmrs symbols in 1 slot
    n_ant = 4    # number of rx antennas
    n_layers = 4 # number of layers
    n_sub = 48*6 # number of subcarriers (dmrs has 1 every 2)
    """
    assert n_symb * n_ant * n_layers <= ch.shape[0]

    if est_type == 'ls':
        ch_r = ch.reshape(n_ant, n_layers, n_symb, n_sub).transpose(3, 1, 0, 2)
    else:
        ch_r = ch.reshape(n_ant, n_layers, n_symb, interp * n_sub).transpose(0, 1, 3, 2)

    return ch_r[None, ...]  # add batch dimension
