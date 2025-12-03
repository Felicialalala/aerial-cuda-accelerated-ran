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

function W = compute_srs_ChEst_filter(combOffset,srsChEst)

% function computes srs channel estimation filter

%inputs:
% combOffset --> comb offset 0-3.

%outputs:
% W          --> filter. Dim: nPrbPerThreadBlock/2 x 3*nPrbPerThreadBlock. Output_freq x Input_freq

%%
%PARAMATERS
nPrbPerThreadBlock = srsChEst.nPrbPerThreadBlock; % number of prb per threadblock
delaySpread        = srsChEst.delaySpread;        % delay spread assumed by channel estimation filters
N0                 = srsChEst.N0;                 % noise variance assumed by estimation filters
df                 = srsChEst.df;                 % subcarrier spacing (Hz)

%%
%GRIDS

%input grid:
f_in = 0 : 4 : (12*nPrbPerThreadBlock - 1);
f_in = df*f_in';
Nf_in = length(f_in);

%output grid:
f_out = 12 : 24 : (12*nPrbPerThreadBlock - 1);
f_out = df*(f_out' - combOffset);
Nf_out = length(f_out);

%%
%CYCLIC SHIFT

cs_seq = zeros(Nf_in,4);
cs_max = 12;

cs = 0 : 3;
cs = cs*cs_max/4;
alpha = 2*pi*cs/cs_max;

idx = 0 : (Nf_in - 1);
for i = 1 : 4
    cs_seq(:,i) = exp(1i*alpha(i)*idx');
end

%%
%STATISTICS

%Input covaraince:
[F1,F2] = meshgrid(f_in,f_in);

RYY = sinc_tbf(delaySpread*(F1 - F2));
RYY = (cs_seq*cs_seq').*RYY + N0*eye(Nf_in);

%Output-Input correlation:
[F1,F2] = meshgrid(f_in,f_out);
RXY = sinc_tbf(delaySpread*(F1 - F2));

%%
%FILTER

W = RXY*pinv(RYY);

% REE = W*RYY*W' - W*RXY' - RXY*W' + eye(Nf_out);
% SNR = -10*log10(diag(REE))

