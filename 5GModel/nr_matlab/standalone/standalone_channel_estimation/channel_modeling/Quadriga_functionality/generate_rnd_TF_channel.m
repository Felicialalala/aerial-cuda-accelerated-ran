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

function H = parameters_to_TF(PAR_chan,PAR_mod)

%%
%PARAMETERS

%modulation parameters:
df = PAR.mod.df; %subcarrier spacing (Hz)
dt = PAR.mod.dt; %symbol duration (s)
Nf = PAR.mod.Nf; %number of subcarriers in slot
Nt = PAR.mod.Nt; %number of OFDM symbols in slot

%channel parameters:
L_UE = PAR.chan.no_rxant; %number of UE antennas
L_HUB = PAR.chan.no_txant; %number of Hub antennas
no_path = PAR.chan.no_path; %number of propogation paths
coeff = PAR.chan.coeff; %antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt
delay = PAR.chan.delay; %delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt

%%
%SETUP

%build frequency grid:
f = 0 : (Nf - 1);
f = f*df;

%reshape antenna coeff:
L = L_UE*L_HUB; %total number of antennas
coeff = reshape(coeff,L_UE*L_HUB,no_path,Nt); %dim: L x no_paths x Nt

%simplify delay:
delay = squeeze(delay(1,1,:,1)); %dim: no_paths x 1

%%
%START

H = zeros(L,Nf,Nt);

for p = 1 : no_path
    freq_wave = exp(-2*pi*1i*f*delay(p));
    
    for t = 1 : Nt
        H(:,:,t) = H(:,:,t) + coeff(:,p,t) * freq_wave;
    end
    
end

%%
%NORMALIZE

E = abs(H).^2;
H = H / sqrt(mean(E(:)));

%%
%RESHAPE

H = reshape(H,L_UE,L_HUB,Nf,Nt);
H = permute(H,[2 1 3 4]);

E = abs(H);
E = squeeze(E(1,1,:,:));

figure
imagesc(abs(E));







