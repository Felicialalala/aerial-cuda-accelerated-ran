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

% \brief This file generated the ofdm TV for verify OFDM modulation and demodulation
% It will generate a ofdm_test_tv.h5 file, used for ofdm_mod_demod_ex
% Resutls will be output from ofdm_mod_demod_ex
clear all
% load phy and carrier from any file or degbugging process,  DO NOT run standalone
% phy and carrier can be generated inside gNBtransmitter.m, under gNbDlGenTimeDomainSig
load('pdsch_phy.mat'); % inlcude both Phy and carrier
genOfdmTv(Phy, carrier)

function genOfdmTv(Phy, carrier)
Xtf_frame = Phy.tx.Xtf_frame;
Xtf = Phy.tx.Xtf;
Xt = Phy.tx.Xt;
Xt_frame = Phy.tx.Xt_frame;

h5File  = H5F.create("ofdm_test_tv.h5", 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'X_tf', single(Xtf));
hdf5_write_nv(h5File, 'Xtf_frame', single(Xtf_frame));
hdf5_write_nv(h5File, 'X_t', single(Xt));
hdf5_write_nv(h5File, 'Xt_frame', single(Xt_frame));
carrier_pars.N_sc = uint16(carrier.N_sc);
carrier_pars.N_FFT = uint16(carrier.Nfft);
carrier_pars.N_txLayer = uint16(carrier.numTxAnt);
carrier_pars.N_rxLayer = uint16(carrier.numRxAnt);
carrier_pars.id_slot = uint16(carrier.idxSlot);
carrier_pars.id_subFrame = uint16(carrier.idxSubframe);
carrier_pars.mu = uint8(carrier.mu);
carrier_pars.cpType = uint8(carrier.CpType);
carrier_pars.f_c = uint32(1/carrier.T_c);
carrier_pars.f_samp = uint32(carrier.f_samp);
carrier_pars.N_symbol_slot = uint16(carrier.N_symb_slot);
carrier_pars.kappa_bits = uint16(log2(carrier.k_const));
carrier_pars.ofdmWindowLen = uint16(0); % no windowing involved
carrier_pars.rolloffFactor = single(0.5); % no windowing involved

hdf5_write_nv(h5File, 'carrier_pars', carrier_pars);

h5File.close();

end