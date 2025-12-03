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

function W_time = compute_1D_time_interpolator(PAR)

%function computes the 1D MMSE time interpolator

%outputs:
%W_time --> two options for the time interpolator. Dim: Nt x Nt_p x 2

%%
%PARAMETERS

%modulation parameters:
dt = PAR.mod.dt; %OFDM symbol duration (s)
Nt = PAR.mod.Nt; %number of symbols in subframe
df = PAR.mod.df; %subcarrier spacing (Hz)

%pilot parameters:
Nt_p = PAR.pilot.Nt_p; %number of time pilots
OCC_t = PAR.pilot.OCC_t; %Time OCC pilot. Dim: Nf_t x 1
pilot_grid_time = PAR.pilot.pilot_grid_time;

%MMSE parameters:
N0 = PAR.MMSE.N0; %noise variance (linear scale)
Doppler_spread = PAR.MMSE.Doppler_spread; %prior on channel Doppler spread (Hz)
delay_spread = PAR.MMSE.delay_spread; %prior on channel delay spread (s)

%%
%SETUP

data_grid_time = 1 : Nt;

PG = (1 / (df*6)) / delay_spread;

%%
%COVARIANCE

%covariance of pilot tones:
RYY = zeros(Nt_p);

for i = 1 : Nt_p
    for j = 1 : Nt_p
        RYY(i,j) = sinc_tbf(Doppler_spread*dt*(pilot_grid_time(i) - pilot_grid_time(j)));
    end
end

RYY = (OCC_t*OCC_t' + ones(Nt_p)) .* RYY + (N0 / PG) * eye(Nt_p);

%%
%CORRELATION

%correlation between pilots and total channel:
RXY = zeros(Nt,Nt_p);

for i = 1 : Nt
    for j = 1 : Nt_p
        RXY(i,j) = sinc_tbf(Doppler_spread*dt*(data_grid_time(i) - pilot_grid_time(j)));
    end
end

%%
%MMSE FILTER

W_time = RXY * pinv(RYY);

%%
%ADD OPTIONS

W_time = repmat(W_time,[1 1 2]);

W_time(:,:,2) = W_time(:,:,2) * diag(OCC_t);

x = 2;









