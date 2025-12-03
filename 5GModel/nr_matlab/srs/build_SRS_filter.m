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

function [W, noiseEstDebias] = build_SRS_filter(combSize, nCyclicShifts, delayWidth, shift)

% function builds filter which seperates cyclic shift and estimates the
% channel

% inputs:
% combSize       --> SRS comb size. 2 or 4.
% nCyclicShifts  --> number of antenna ports to seperate. 1, 2, or 4.
% delayWidth     --> channel delay width (microseconds)

%%
% CONSTANTS

nPrbs  = 4;
df     = 30*10^3;
nSrsSc = nPrbs * 12 / combSize;

f = 0 : (nSrsSc - 1);
f = f * combSize * df;
f = f';

N0  = 10^(-2.0);

%%
%CENTERING SHIFT

tau = delayWidth * shift;
s   = exp(2*pi*1i*tau*f);

%%
% CYCLIC SHIFTS

cs_spacing = nSrsSc / nCyclicShifts;
CS         = zeros(nSrsSc,nCyclicShifts);

for csIdx = 0 : (nCyclicShifts - 1)
    CS(csIdx*cs_spacing + 1, csIdx + 1) = 1;
end

CS = fft(CS, nSrsSc);

%%
% KERNEL

K = diag(conj(s))*sinc_tbf((f - f')*delayWidth)*diag(s);
% K = sinc_tbf((f - f')*delayWidth);

R_YY = zeros(nSrsSc);
for csIdx = 0 : (nCyclicShifts - 1)
    R_YY = R_YY + (CS(:,csIdx + 1)*CS(:,csIdx + 1)') .* K;
end
R_YY = R_YY + N0*eye(nSrsSc);

%%
% BASE FILTER

global SimCtrl;
W = K*pinv(R_YY);
W = reshape(fp16nv(real(W), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(W), SimCtrl.fp16AlgoSel), [size(W)]);

%%
% COMPUTE DE-BIAS MATRIX

% biasMtrx = zeros(2);
% 
% F = zeros(nSrsSc);
% for csIdx = 0 : (nCyclicShifts - 1)
%     F = F + diag(CS(:,csIdx + 1)) * W * diag(conj(CS(:,csIdx + 1)));
% end
% A = F*K*F';
% biasMtrx(1,1) = nCyclicShifts * mean(diag(A));
% 
% B = F*F';
% biasMtrx(1,2) = mean(diag(B));

R_tot = -eye(nSrsSc);
for csIdx = 0 : (nCyclicShifts - 1)
    R_tot = R_tot + diag(CS(:,csIdx + 1)) * W * diag(conj(CS(:,csIdx + 1)));
end

% C = nCyclicShifts * R_tot * K * R_tot';
M = R_tot * R_tot';
noiseEstDebias = 1 / mean(diag(M));
% biasMtrx(2,1) = mean(diag(C));
% 
% 
% D = R_tot * R_tot';
% biasMtrx(2,2) = mean(diag(D));

% biasMtrx

% % deBiasMtrx = biasMtrx;
% deBiasMtrx = biasMtrx^(-1);












