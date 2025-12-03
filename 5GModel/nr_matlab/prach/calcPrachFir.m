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

function [prachFIRx2, prachFIRx3] = calcPrachFir(Nfft, L_RA, N_sc_RB, K)
%
% Calculate x2 and x3 LPF for PRACH using Matlab filter design tool
%

Nsig = ceil(L_RA/(K*N_sc_RB))*K*N_sc_RB;

rippleDB = 0.03;
stopDB = 50;
Dpass = 10^(rippleDB/20)-1; % Passband Ripple
Dstop = 10^(-stopDB/20);    % Stopband Attenuation
dens  = 20;                 % Density Factor

% x3 FIR
Fs = Nfft*3;                % Sampling Frequency
Fpass = Nsig/2;             % Passband Frequency
Fstop = Nfft - Nsig/2;      % Stopband Frequency

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);
% Calculate the coefficients using the FIRPM function.
prachFIRx3 = firpm(N, Fo, Ao, W, {dens});

% x2 FIR
Fs = 2;                     % Sampling Frequency
Fpass = 0.25;               % Passband Frequency
Fstop = 0.75;               % Stopband Frequency

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);
% Calculate the coefficients using the FIRPM function.
prachFIRx2 = firpm(N, Fo, Ao, W, {dens});

return
