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

function Interf_Chan = cfgInterferenceChan

Interf_Chan.type = 'AWGN';                 % Supported interf. channel models: AWGN; P2P; TDL and CDL in time domain with 5G toolbx; TDL, CDL, UMi, UMa and RMa in freq domain with Sionna
Interf_Chan.DelayProfile = 'CDL-A';        % Interf_Channel delay profile. Valid for TDL and CDL Interf_Channel type.
Interf_Chan.DelaySpread = 30e-9;           % delay spread in second
Interf_Chan.MaximumDopplerShift = 10;      % max doppler freq in Hz
Interf_Chan.CDL_updateRateType = 'perSample';% 'perSample', 'perSymbol'. take effect only when Interf_Chan.type='CDL'
Interf_Chan.gNB_AntPolarizationAngles = [45, -45]; % vector of length 1 or 2. gNB polarization angles, valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.UE_AntPolarizationAngles = [45, -45]; % vector of length 1 or 2. UE polarization angles, valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.gNB_AntArraySize = [1,2,2];    % gNB antenna array size: [nuw_row_elements, num_col_elements, num_polarization], valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.UE_AntArraySize = [1,1,2];     % gNB antenna array size: [nuw_row_elements, num_col_elements, num_polarization], valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.gNB_AntPattern = '38.901';     % 'isotropic' or '38.901',gNB antenna pattern, valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.UE_AntPattern = 'isotropic';   % 'isotropic' or '38.901',UE antenna pattern, valid for 'CDL', 'UMi', 'UMa', 'RMa' Interf_Channel models
Interf_Chan.MIMOCorrelation = 'Low';       % only for TDL Interf_Channel model
Interf_Chan.SIR = 40;                      % signal to interference ratio, dB 
Interf_Chan.delay = 0e-6;                  % second
Interf_Chan.CFO = 0;                       % Hz
Interf_Chan.model_source = 'MATLAB5Gtoolbox';    % 'custom', 'MATLAB5Gtoolbox', 'sionna'
if verLessThan('matlab', '9.11')
    Interf_Chan.simplifiedDelayProfile = 0;    % not support simplified delay profile
else
    Interf_Chan.simplifiedDelayProfile = 1;    % use simplified delay profile
end
Interf_Chan.gain = 1;                      % Interf_Channel gain


return
