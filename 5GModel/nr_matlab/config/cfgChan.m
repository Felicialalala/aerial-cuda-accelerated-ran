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

function Chan = cfgChan

Chan.type = 'AWGN';                 % refer to initChan.m for supported channel model
Chan.DelayProfile = 'CDL-A';        % channel delay profile. Valid for TDL and CDL channel type.
Chan.DelaySpread = 30e-9;           % delay spread in second
Chan.MaximumDopplerShift = 10;      % max doppler freq in Hz
Chan.CDL_updateRateType = 'perSample';% 'perSample', 'perSymbol'. take effect only when Chan.type='CDL'
Chan.gNB_AntPolarizationAngles = [45, -45]; % vector of length 1 or 2. gNB polarization angles, valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.UE_AntPolarizationAngles = [45, -45]; % vector of length 1 or 2. UE polarization angles, valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.gNB_AntArraySize = [1,2,2];    % gNB antenna array size: [nuw_row_elements, num_col_elements, num_polarization], valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.UE_AntArraySize = [1,1,2];     % gNB antenna array size: [nuw_row_elements, num_col_elements, num_polarization], valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.gNB_AntPattern = '38.901';     % 'isotropic' or '38.901',gNB antenna pattern, valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.UE_AntPattern = 'isotropic';   % 'isotropic' or '38.901',UE antenna pattern, valid for 'CDL', 'UMi', 'UMa', 'RMa' channel models
Chan.gNB_AntSpacing = [0.5, 0.5];   % in unit of lambda (wavelength)
Chan.UE_AntSpacing = [0.5, 0.5];    % in unit of lambda (wavelength)
Chan.MIMOCorrelation = 'Low';       % only for TDL channel model
Chan.CDL_DPA = [];                      % CDL channel delay/power/angle, only valid for CDL-customized
Chan.CDL_PCP = [];                      % CDL channel per cluster parameters, only valid for CDL-customized

% Common for all Chan.type
Chan.SNR = 40;                      % dB 
Chan.delay = 0e-6;                  % second
Chan.CFO = 0;                       % Hz
Chan.model_source = 'MATLAB5Gtoolbox';    % 'custom', 'MATLAB5Gtoolbox', 'sionna'
if verLessThan('matlab', '9.11')
    Chan.simplifiedDelayProfile = 0;    % not support simplified delay profile
else
    Chan.simplifiedDelayProfile = 1;    % use simplified delay profile
end
Chan.gain = 1;                      % channel gain
   
% findLic = license('checkout', 'matlab_5g_toolbox');
% if findLic == 0 
%     Chan.use5Gtoolbox = 0;
% end

return
