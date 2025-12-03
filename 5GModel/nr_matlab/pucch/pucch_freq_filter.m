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

function [Wf] = pucch_freq_filter(mu)

%function computes the mmse frequency ChEst filter

%outputs:
%Wf --> mmse filter. Dim: 12 x 12

%%
%PARMATERS
df = 15e3*2^mu; % subcarrier spacing (Hz);
delaySpread = 2e-6; % sp.gnb.pucch.reciever.ChEst.delaySpread;  % delay spread assumed by channel esitmation block (s)
N0 = 10; % sp.gnb.pucch.reciever.ChEst.N0;                    % noise variance assumed by channel estimation block
% delaySpread = 0;

%%
%SETUP

%frequency grid:
f = 0 : 11;
f = f' * df;
[F1,F2] = meshgrid(f,f);

%neighbor cyclic shifts:
cs = [1 3 5 7 9 11];
cs = cs - 1;
cs = cs * (1 / df) / 12;

%%
%COVARIANCE

RYY_base = sinc_tbf(delaySpread*(F1 - F2));
RYY_tot = zeros(12);

for i = 1 : 6
    s = exp(-2*pi*1i*f*cs(i));
    RYY_tot = RYY_tot + (s*s') .* RYY_base;
end

RYY_tot = RYY_tot + N0*eye(12);

%%
%CORRELATION

RXY = RYY_base;
RXX = RYY_base;

%%
%FILTER

Wf = RXY * pinv(RYY_tot);

%%
%ERROR
% 
REE = Wf*RYY_tot*Wf' - Wf*RXY' - RXY*Wf' + RXX;

SNR = -10*log10(diag(REE));
