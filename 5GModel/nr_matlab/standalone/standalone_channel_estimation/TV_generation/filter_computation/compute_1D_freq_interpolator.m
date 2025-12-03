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

function W_freq = compute_1D_freq_interpolator(PAR)

%function computes the 1D MMSE frequency interpolator

%outputs:
%W_freq --> frequency interpolation filter. Dim: Nf x num_freq_tones x 3 x 2

%%
%PARAMETERS

%modulation parameters:
df = PAR.mod.df; %subcarrier spacing (Hz)

%cluster parameters:
Nf_a = PAR.cluster.Nf_a; %number of subcarriers in each allocation

%pilot parameters:
Nf_p = PAR.pilot.Nf_p; %number of freq pilots
pilot_grid_freq = PAR.pilot.pilot_grid_freq; %Freq pilot grid.
OCC_f = PAR.pilot.OCC_f; %Frequency OCC pilot. Dim: Nf_p x 1

%MMSE parameters:
N0 = PAR.MMSE.N0; %noise variance (linear scale)
delay_spread = PAR.MMSE.delay_spread; %prior on channel delay spread (s)

%%
%SETUP

data_grid_freq = 1 : Nf_a;

pilot_grid_freq = repmat(pilot_grid_freq,[3 1]);

for i = 1 : 3
    pilot_grid_freq(i,:) = pilot_grid_freq(i,:) + 2*(i-1);
end

%%
%COVARIANCE

%covariance of channel on pilot tones:
RYY = zeros(Nf_p,Nf_p);


for i = 1 : Nf_p
    for j = 1 : Nf_p
        RYY(i,j) = sinc_tbf(delay_spread*df*(pilot_grid_freq(1,i) - pilot_grid_freq(1,j)));
    end
end

%total covariance (includes interference due to OCC and AWGN):
RYY_tot = 2*(OCC_f*OCC_f' + ones(Nf_p)) .* RYY + N0*eye(Nf_p);

%compute precesion matrix:
RYY_tot_inv = pinv(RYY_tot);

%%
%CORRELATION

%correlation between channel on pilot tones to all tones:
RXY = zeros(Nf_a,Nf_p,3);

for s = 1 : 3
    for i = 1 : Nf_a
        for j = 1 : Nf_p
            RXY(i,j,s) = sinc_tbf(delay_spread*df*(data_grid_freq(i) - pilot_grid_freq(s,j)));
        end
    end
end

RXY = 2*RXY;

%%
%MMSE FILTER

W_freq = zeros(Nf_a,Nf_p,3);

for s = 1 : 3
    W_freq(:,:,s) = RXY(:,:,s) * RYY_tot_inv;
end

W_freq = repmat(W_freq,[1 1 1 2]);

for s = 1 : 3
    W_freq(:,:,s,2) = W_freq(:,:,s,2) * diag(OCC_f);
end
