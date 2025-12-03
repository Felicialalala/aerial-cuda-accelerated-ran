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

%%
%BUILD CHANNEL

H = generate_channel_wrapper;

%%
%BUILD TV

[Y,H_true,TV,PAR] = TV_generation_main(H);

%% GENERATE HDF5 OUTPUT FILE
fname = sprintf('channel_est_%dPRB_%0.fkHz_%dAntenna.h5', TV.mod.Nf / 12, TV.mod.df/1000, TV.mod.L_BS);
h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
% Transpose the DMRS index matrices for real-time implementation.
% Also, subtract 1 so that indices will be zero-based.
hdf5_write_nv(h5File, 'DMRS_index_freq', int16(TV.pilot.DMRS_index_freq' - 1));
hdf5_write_nv(h5File, 'DMRS_index_time', int16(TV.pilot.DMRS_index_time' - 1));
hdf5_write_nv(h5File, 'W_freq',          single(TV.filter.W_freq));
hdf5_write_nv(h5File, 'W_time',          single(TV.filter.W_time));
hdf5_write_nv(h5File, 'Y',               single(Y));
H5F.close(h5File);


%%
%PROCESS TV

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%GPU%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SNR = interpolate_TV_main(Y,H_true,TV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%GPU%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
