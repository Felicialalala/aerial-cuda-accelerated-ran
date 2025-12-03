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

function W = PF2_FreqChEst_filter(nPrb,mu)

%function compute PUCCH format 3 frequency channel estimation filter for a
%block of PRBs

%inputs:
% nPrb --> number of prbs in the block
% mu   --> numerology

%outputs:
% W --> frequency ChEst filter. Dim: 12*nPrb x 4*nPrb

%%
%PARAMATERS

N0 = 10^(-1.5);

switch mu
    case 0
        delaySpread = 4.69 * 10^(-6);
    case 1
        delaySpread = 2.34 * 10^(-6);
    case 2
        delaySpread = 1.17 * 10^(-6);
    case 3
        delaySpread = 0.59 * 10^(-6);
    case 4
        delaySpread = 0.29 * 10^(-6);
    otherwise
        error('mu = %d not supported',mu);
end

delayCenter = 0.9 * 0.5 * delaySpread;

scs = 15 * 2^mu * 10^3;
f   = 0 : (12*nPrb - 1);
f   = scs*f';

f_dmrs = 0 : 3 : (12*nPrb - 1);
f_dmrs = scs*(f_dmrs + 1)';

%%
%START

shiftSeq_dmrs = exp(-2*pi*1i*delayCenter*f_dmrs);
shiftSeq      = exp(-2*pi*1i*delayCenter*f);

K_dmrs = sinc_tbf((f_dmrs - f_dmrs')*delaySpread);
K_dmrs = (shiftSeq_dmrs*shiftSeq_dmrs') .* K_dmrs;

K = sinc_tbf((f - f_dmrs')*delaySpread);
K = (shiftSeq*shiftSeq_dmrs') .* K;

W = K*(K_dmrs + N0*eye(nPrb*4))^(-1);

end

