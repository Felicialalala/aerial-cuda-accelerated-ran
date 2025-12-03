% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

pusch_rx = hdf5_load_nv('pusch_rx_MIMO8x16_PRB272_DataSyms6.h5');

config = struct('NumTransportBlocks',          uint32(8),      ...
                'NumLayers',                   uint32(1),      ...
                'InputLayerSize',              uint32(117504), ...
                'NumFillerBits',               uint32(72),     ...
                'TransportBlockSize',          uint32(108552), ...
                'CodeBlocksPerTransportBlock', uint32(13),     ...
                'ScramblingEnabled',           uint32(1),      ...
                'DmrsConfig',                  uint32(3));

h5File = H5F.create('pusch_rx_MIMO8x16_PRB272_SYM6_SCR1.h5');
hdf5_write_nv(h5File, 'DataRx', pusch_rx.DataRx);
hdf5_write_nv(h5File, 'Data_sym_loc', pusch_rx.Data_sym_loc);
hdf5_write_nv(h5File, 'DescrShiftSeq', pusch_rx.DescrShiftSeq);
hdf5_write_nv(h5File, 'Noise_pwr', pusch_rx.Noise_pwr);
hdf5_write_nv(h5File, 'RxxInv', pusch_rx.RxxInv);
hdf5_write_nv(h5File, 'UnShiftSeq', pusch_rx.UnShiftSeq);
hdf5_write_nv(h5File, 'WFreq', pusch_rx.WFreq);
hdf5_write_nv(h5File, 'config', config);
H5F.close(h5File);
