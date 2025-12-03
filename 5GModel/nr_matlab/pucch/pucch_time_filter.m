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

function Wt = pucch_time_filter(nSym, mu, pucchTable)

%function compute the pucch time channel estimation filter

%inputs:
% nSym --> number of pucch symbols

%output:
% Wt   --> time estimation filter. Dim: nSym_dmrs x nSym_data 

%%
%PARAMATERS

%modulation:
df = 15e3*2^mu; % subcarrier spacing (Hz);
dt = (1/df)*(2048+144)/2048;
nSym_data = floor(nSym / 2);  % number of data symbols
nSym_dmrs = ceil(nSym / 2);   % number of dmrs symbols

%mmse:
DopplerSpread = 300; % sp.gnb.pucch.reciever.ChEst.DopplerSpread;  % delay spread assumed by channel esitmation block (s)
N0 = 10; % sp.gnb.pucch.reciever.ChEst.N0;                        % noise variance assumed by channel estimation block

%%
%SETUP

t_dmrs = 0 : (nSym_dmrs - 1);
t_dmrs = 2*t_dmrs + 0;
t_dmrs = dt*t_dmrs';

t_data = 0 : (nSym_data - 1);
t_data = 2*t_data + 1;
t_data = dt*t_data';

tOCC = pucchTable.tOCC;

%%
%COVARIANCE

[T1_cov,T2_cov] = meshgrid(t_dmrs,t_dmrs);
RYY_base = sinc_tbf(DopplerSpread*(T1_cov - T2_cov));

RYY_tot = zeros(nSym_dmrs);
for i = 1 : nSym_dmrs
    s = tOCC{nSym_dmrs,i}.';
    RYY_tot = RYY_tot + (s*s') .* RYY_base;
end

RYY_tot = RYY_tot + sqrt(N0 / 4) * eye(nSym_dmrs);

%%
%CORRELATION

[T1_cor,T2_cor] = meshgrid(t_dmrs,t_data);
RXY = sinc_tbf(DopplerSpread*(T1_cor - T2_cor));

%%
%FILTER

Wt = RXY * pinv(RYY_tot);
Wt = Wt.';

%%
%ERROR

% REE = Wt*RYY_tot*Wt' - Wt*RXY' - RXY*Wt' + eye(nSym_data);
% SNR = -10*log10(diag(REE));

