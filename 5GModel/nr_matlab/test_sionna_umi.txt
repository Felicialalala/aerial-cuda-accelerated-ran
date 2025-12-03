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

import matplotlib
import mitsuba
import tensorflow as tf
import sionna
from sionna.nr import PUSCHConfig, PUSCHTransmitter
from sionna.channel import TimeChannel, OFDMChannel
from sionna.channel.tr38901 import AntennaArray, UMi, TDL
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.ofdm import ResourceGrid
import time

sionna.config.xla_compat=True

_num_tx_ant = 2
_num_rx_ant = 4
_carrier_frequency = 3.5e9
_ut_array = AntennaArray(num_rows=1,num_cols=int(_num_tx_ant/2),polarization="dual",polarization_type="cross",antenna_pattern="omni",carrier_frequency=_carrier_frequency)
_bs_array = AntennaArray(num_rows=1,num_cols=int(_num_rx_ant/2),polarization="dual",polarization_type="cross",antenna_pattern="38.901",carrier_frequency=_carrier_frequency)

batch_size = 1
_num_tx = 1
topology = gen_topology(batch_size,_num_tx,"umi",0,5)
_channel_model = UMi(carrier_frequency=_carrier_frequency,o2i_model="low",ut_array=_ut_array,bs_array=_bs_array,direction="uplink",enable_pathloss=False,enable_shadow_fading=False)
_channel_model.set_topology(*topology)

FFTsize = int(3276);
num_active_REs = int(3276);
num_left_subc = int((FFTsize-num_active_REs)/2)
num_right_subc = FFTsize - num_active_REs - num_left_subc
cpLen = 288
resource_grid = ResourceGrid(num_ofdm_symbols=14,fft_size=FFTsize,subcarrier_spacing=30000,num_tx=_num_tx,num_streams_per_tx =_num_tx_ant,cyclic_prefix_length = cpLen,num_guard_carriers=[num_left_subc,num_right_subc],dc_null=False)
_channel_FD = OFDMChannel(channel_model=_channel_model,resource_grid=resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)

@tf.function(jit_compile=True)
def call_channel_FD(_channel_FD, x_FD):
    y_FD,h_FD = _channel_FD([x_FD, 0.01])
	
x_FD_shape = [int(batch_size), int(_num_tx), int(_num_tx_ant), 14, int(FFTsize)]
x_FD = tf.complex(tf.random.normal((x_FD_shape)), tf.random.normal((x_FD_shape)))
start_time = time.time()
#y_FD,h_FD = _channel_FD([x_FD, 0.01])
call_channel_FD(_channel_FD, x_FD)
print("--- %s seconds ---" % (time.time() - start_time))

# TDL fading
_channel_model_tdl = TDL(model='A', delay_spread=3e-8, carrier_frequency=3.5e9, min_speed=0, max_speed=0.9, num_rx_ant=_num_rx_ant,num_tx_ant=_num_tx_ant)
_channel_FD_TDL = OFDMChannel(channel_model=_channel_model_tdl,resource_grid=resource_grid, add_awgn=True, normalize_channel=True, return_channel=True)
y_FD,h_FD = _channel_FD_TDL([x_FD, 0.01])

bandwidth = 122.88e6
num_time_samples = int(0.5e-3*122.88e6)
lmin = int(0)
lmax = int(2.33e-6*bandwidth)
_channel_TD = TimeChannel(_channel_model,bandwidth,num_time_samples,lmin,lmax,normalize_channel=True,return_channel=True)




_carrier_frequency = 3.5e9
_subcarrier_spacing = 30e3
_num_layers = 2
_mcs_index = 14
_mcs_table = 1
_num_prb = 16
# PUSCHConfig for the first transmitter
pusch_config = PUSCHConfig()
pusch_config.carrier.subcarrier_spacing = _subcarrier_spacing/1000
pusch_config.carrier.n_size_grid = 273 #_num_prb
pusch_config.num_antenna_ports = _num_tx_ant
pusch_config.num_layers = _num_layers
pusch_config.precoding = "codebook"
pusch_config.tpmi = 1
pusch_config.dmrs.dmrs_port_set = list(range(_num_layers))
pusch_config.dmrs.config_type = 2
pusch_config.dmrs.length = 2
pusch_config.dmrs.additional_position = 1
pusch_config.dmrs.num_cdm_groups_without_data = 3
pusch_config.tb.mcs_index = _mcs_index
pusch_config.tb.mcs_table = _mcs_table

pusch_configs = [pusch_config]
_pusch_transmitter_FD = PUSCHTransmitter(pusch_configs, output_domain="freq")

x, b = _pusch_transmitter_FD(batch_size) # Complex Tensor x has shape [batch_size, num_tx, num_tx_ant, num_time_samples]

x_shape = [int(batch_size), int(_num_tx), int(_num_tx_ant), int(num_time_samples)]
x = tf.complex(tf.random.normal((x_shape)), tf.random.normal((x_shape)))
y, h = _channel([x, 0.01])


x_FD_shape = [int(batch_size), int(_num_tx), int(_num_tx_ant), 14, int(FFTsize)]
x_FD = tf.complex(tf.random.normal((x_FD_shape)), tf.random.normal((x_FD_shape)))
start_time = time.time()
y_FD,h_FD = _channel_FD([x_FD, 0.01])
print("--- %s seconds ---" % (time.time() - start_time))
