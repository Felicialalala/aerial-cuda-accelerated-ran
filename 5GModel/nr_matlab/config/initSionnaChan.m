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

function Chan = initSionnaChan(SysPar, Chan)
    if strcmp(Chan.type, 'TDL')
        args_chan_cfg = pyargs("model", Chan.DelayProfile, "delay_spread", Chan.DelaySpread, ...
                                 "carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                 "min_speed", 0, "max_speed", Chan.MaximumDopplerShift/(SysPar.carrier.carrierFreq*1e9)*3e8, ...
                                 "num_rx_ant", py.int(Chan.Nout), "num_tx_ant", py.int(Chan.Nin));
        sionna_chan = py.sionna.channel.tr38901.TDL(args_chan_cfg);
    elseif strcmp(Chan.type, 'CDL') || strcmp(Chan.type, 'UMi') || strcmp(Chan.type, 'UMa') || strcmp(Chan.type, 'RMa')
        UE_num_pol = length(Chan.UE_AntPolarizationAngles);
        gNB_num_pol = length(Chan.gNB_AntPolarizationAngles);
        list_polarization = {'single','dual'};
        UE_polarization = list_polarization{UE_num_pol};
        gNB_polarization = list_polarization{gNB_num_pol};
        list_polarAngles = {[0],[90], [0, 90], [45, -45]};
        list_polarType = {'V', 'H', 'VH', 'cross'};
        idx1 = find(cellfun(@(x) all(x==Chan.UE_AntPolarizationAngles),list_polarAngles, 'UniformOutput', 1));
        if isempty(idx1)
            error('Undefined UE polarization angles for Sionna channel model!')
        end
        UE_polarization_type = list_polarType{idx1};
        idx2 = find(cellfun(@(x) all(x==Chan.gNB_AntPolarizationAngles),list_polarAngles, 'UniformOutput', 1));
        if isempty(idx2)
            error('Undefined gNB polarization angles for Sionna channel model!')
        end
        gNB_polarization_type = list_polarType{idx2};        
        UE_num_rows_per_panel = Chan.UE_AntArraySize(1);
        UE_num_cols_per_panel = Chan.UE_AntArraySize(2);
        gNB_num_rows_per_panel = Chan.gNB_AntArraySize(1);
        gNB_num_cols_per_panel = Chan.gNB_AntArraySize(2);
        UE_antenna_pattern = strrep(Chan.UE_AntPattern,'isotropic','omni');
        gNB_antenna_pattern = strrep(Chan.gNB_AntPattern,'isotropic','omni');
        args_ue_panel_array_cfg = pyargs("num_rows_per_panel", py.int(UE_num_rows_per_panel), "num_cols_per_panel", py.int(UE_num_cols_per_panel), ...
                                        "polarization", UE_polarization, "polarization_type", UE_polarization_type, ... % note that the total num of ant is UE_num_rows_per_panel*UE_num_cols_per_panel*num_pol
                                        "antenna_pattern", UE_antenna_pattern, "carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                        "num_rows", py.int(1), "num_cols", py.int(1), ... % just assume single panel
                                        "element_vertical_spacing", 0.5, "element_horizontal_spacing", 0.5); % assume half-wavelength by default
        args_gnb_panel_array_cfg = pyargs("num_rows_per_panel", py.int(gNB_num_rows_per_panel), "num_cols_per_panel", py.int(gNB_num_cols_per_panel), ...
                                        "polarization", gNB_polarization, "polarization_type", gNB_polarization_type, ... % note that the total num of ant is UE_num_rows_per_panel*UE_num_cols_per_panel*num_pol
                                        "antenna_pattern", gNB_antenna_pattern, "carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                        "num_rows", py.int(1), "num_cols", py.int(1), ... % just assume single panel
                                        "element_vertical_spacing", 0.5, "element_horizontal_spacing", 0.5); % assume half-wavelength by default
        py_UE_panel_array = py.sionna.channel.tr38901.PanelArray(args_ue_panel_array_cfg);
        py_gNB_panel_array = py.sionna.channel.tr38901.PanelArray(args_gnb_panel_array_cfg);
        
        if strcmp(Chan.type, 'CDL')
            args_chan_cfg = pyargs("model", Chan.DelayProfile, "delay_spread", Chan.DelaySpread, ...
                                 "carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                 "ut_array", py_UE_panel_array, "bs_array", py_gNB_panel_array,...
                                 "direction", lower(Chan.link_direction),...
                                 "min_speed", 0.0, "max_speed", Chan.MaximumDopplerShift/(SysPar.carrier.carrierFreq*1e9)*3e8);
            sionna_chan = py.sionna.channel.tr38901.CDL(args_chan_cfg);
        elseif strcmp(Chan.type, 'UMi') || strcmp(Chan.type, 'UMa')  || strcmp(Chan.type, 'RMa')                                    
            if strcmp(Chan.type, 'UMi') 
                args_chan_cfg = pyargs("carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                 "o2i_model", 'low',... % 'low'/'high' loss outdoor to indoor model. See section 7.4.3 of [TR38901]
                                 "ut_array", py_UE_panel_array, "bs_array", py_gNB_panel_array,...
                                 "direction", lower(Chan.link_direction), ...
                                 "enable_pathloss", py.False,"enable_shadow_fading", py.True);
                sionna_chan = py.sionna.channel.tr38901.UMi(args_chan_cfg);
            elseif strcmp(Chan.type, 'UMa') 
                args_chan_cfg = pyargs("carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                 "o2i_model", 'low',... % 'low'/'high' loss outdoor to indoor model. See section 7.4.3 of [TR38901]
                                 "ut_array", py_UE_panel_array, "bs_array", py_gNB_panel_array,...
                                 "direction", lower(Chan.link_direction), ...
                                 "enable_pathloss", py.False,"enable_shadow_fading", py.True);
                sionna_chan = py.sionna.channel.tr38901.UMa(args_chan_cfg);
            elseif strcmp(Chan.type, 'RMa')
                args_chan_cfg = pyargs("carrier_frequency", SysPar.carrier.carrierFreq*1e9, ...
                                 "ut_array", py_UE_panel_array, "bs_array", py_gNB_panel_array,...
                                 "direction", lower(Chan.link_direction), ...
                                 "enable_pathloss", py.False,"enable_shadow_fading", py.True);
                sionna_chan = py.sionna.channel.tr38901.RMa(args_chan_cfg);
            end
            args_gen_topology = pyargs("batch_size", py.int(1), "num_ut", py.int(1), ...
                                        "scenario", lower(Chan.type),"min_bs_ut_dist", py.None, "isd", py.None, ... % all in meters
                                        "bs_height", py.None, "min_ut_height", py.None, "max_ut_height", py.None, ...
                                        "min_ut_velocity", 0.0, "max_ut_velocity", Chan.MaximumDopplerShift/(SysPar.carrier.carrierFreq*1e9)*3e8);
            sionna_topology = py.sionna.channel.gen_single_sector_topology(args_gen_topology);
            sionna_chan.set_topology(sionna_topology{1},sionna_topology{2},sionna_topology{3},sionna_topology{4},sionna_topology{5},sionna_topology{6});
        end
    else
        error('The channel type is not defined or integrated from  Sionna!')
    end
    args_rg_cfg = pyargs("num_ofdm_symbols", py.int(SysPar.carrier.N_symb_slot), "fft_size", py.int(SysPar.carrier.N_grid_size_mu*12), ... % generate CFR on freq (-fft_size/2:fft_size/2-1)*scs, not necessary on all SysPar.carrier.Nfft REs
                                       "subcarrier_spacing", py.int(2^SysPar.carrier.mu*15000), ...
                                       "num_tx", py.int(1), "num_streams_per_tx", py.int(Chan.Nin), "dc_null",py.False);
    sionna_rg = py.sionna.ofdm.ResourceGrid(args_rg_cfg);
    args_OFDM_chan_cfg = pyargs("channel_model",sionna_chan, "resource_grid", sionna_rg,"normalize_channel", py.False, "add_awgn", py.False, "return_channel", py.True);%,Note that sionna by default has the normalization of the PDP as the 5G toolbox does. In addtion, we need to disable the CFR normalization as 5G toolbox does not have it.
    py_sionna_chan = py.sionna.channel.OFDMChannel(args_OFDM_chan_cfg);
    Chan.sionna_chan = py_sionna_chan;
end

function [a, b] = findTwoIntegerFactorsCloseToSquareRoot(x)
   sqrt_x = sqrt(x);
   a = floor(sqrt_x);
   while a > 0
        if mod(x, a) == 0
            b = x/a;
            break;
        else
            a = a - 1;
        end
   end
end