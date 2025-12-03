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

function W_upper = derive_remainder_filter(

% function derives mmse interpolation filter for a single prb. Extracts
% additional dmrs signal upper and lower prbs.

% *-* (- indicates prb where channel estimated, * indicates prb where
% additional dmrs extracted)

% outputs:
% W_upper --> upper interpolation filter. Dim: 12 x 18

%%
%PARAMATERS

% numerology:
df = sp.gnb.numerology.df;           % subcarrier spacing (Hz);

% mmse ChEst paramaters:
if strcmp(sp.sim.opt.simType,'pusch')
    ChEst = sp.gnb.pusch.reciever.ChEst;
elseif strcmp(sp.sim.opt.simType,'pdsch')
    ChEst = sp.gnb.pdsch.reciever.ChEst;
end
N0 = ChEst.N0;                       % noise variance assumed by channel estimation block
delaySpread = ChEst.delaySpread;     % delay spread assumed by channel esitmation block (s)
channelEnergy = ChEst.channelEnergy; % average channel energy assumed by channel estimation block

%%
%SETUP

% sizes:
Nfc = 12;                     % number of subcarriers to estimate
Nfc_dmrs =  18;               % number of dmrs subcarriers
 
% data grid:
f = (12) : (12 + 12);
f = df * f';

% dmrs grid:
f_dmrs = 0 : (Nfc_dmrs - 1);
f_dmrs = 2 * f_dmrs + 1;
f_dmrs = df * f_dmrs';

% fOCC
fOCC = ones(Nfc_dmrs,1);
fOCC(mod(1:Nfc_dmrs,2) == 0) = -1;

%%
%COVARIANCE

% channel covariance:
[F1,F2] = meshgrid(f_dmrs,f_dmrs);
RYY = sinc_tbf(delaySpread*(F1 - F2));

% add fOCC:
RYY = (fOCC * fOCC' + ones(Nfc_dmrs)) .* RYY;

% add energy:
RYY = channelEnergy * RYY;

% add noise:
RYY = RYY + N0 * eye(Nfc_dmrs);

%%
%CORRELATION

% channel correlation:
[F1,F2] = meshgrid(f_dmrs,f);
RXY = sinc_tbf(delaySpread*(F1 - F2));

% add energy:
RXY = channelEnergy * RXY;

%%
%FILTER

W_upper = RXY * pinv(RYY);











