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

function H = taps_to_TF(CT,PAR)

%converts channel taps to a TF response

%outputs:
%H --> TF channel response. Dim: Nf x Nt x L_BS x 2

%%
%PARAMETERS

%modulation parameters:
df = PAR.mod.df; %subcarrier spacing (Hz)
Nf = PAR.mod.Nf; %number of subcarriers in slot
Nt = PAR.mod.Nt; %number of OFDM symbols in slot

%channel parameters:
L_UE = CT.no_rxant; %number of UE antennas
L_BS = CT.no_txant; %number of Hub antennas
no_path = CT.no_path; %number of propogation paths
coeff = CT.coeff; %antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt
delay = CT.delay; %delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt

%%
%SETUP

%build frequency grid:
f = 0 : (Nf - 1);
f = f*df;

%reshape antenna coeff:
L = L_UE*L_BS; %total number of antennas
coeff = reshape(coeff,L,no_path,Nt); %dim: L x no_paths x Nt

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
%PERMUTE

H = reshape(H,L_BS,L_UE,Nf,Nt);
H = permute(H,[3 4 1 2]); %dim: Nf x Nt x L_BS x 2









