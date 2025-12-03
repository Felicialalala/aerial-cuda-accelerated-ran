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

function H = generate_channel_wrapper

%function generate rnd TF channels for a large number of UEs

%outputs:
%H --> TF channel. Dim: Nf x Nt x L_BS x num_UE_ant

%%
%PARAMETERS

%modulation parameters:
PAR.mod.Nf = 12*8*13; %number of subcarriers
PAR.mod.df = 15*10^3; %subcarrier spacing
PAR.mod.Nt = 14; %number of OFDM symbols
PAR.mod.dt = 1 / PAR.mod.df; %symbol duration

%antenna parameters:
PAR.ant.L_BS = 16; %number of BS antennas
%using Quadriga class "qd_array" for details section 2.2.2 in manual
PAR.ant.tx_array = qd_arrayant.generate('3gpp-3d',1,8,4*10^9,3); %BS antenna array geometry
PAR.ant.rx_array = qd_arrayant('xpol'); %UE antenna array

%propogation parameters:
PAR.prop.carrier_frequency = 4.0*10^9; %carrier frequency (Hz)
PAR.prop.scenerio = '3GPP_3D_UMi_NLOS'; %propogation parameters
LOS_flag = 0; %LOS channel? LOS_flag = 0  --> no, LOS_flag = 1 --> yes

%geometry parameters:
PAR.geo.tx_position = [0 ; 0 ; 30]; %location of BS antenna (x,y,z)

%simulation parameters:
num_UE = 6; %number of UE drops
UE_speed = 30 * 1000 / 60^2; %UE speed (m/s)


%%
%LAYOUT 

%build Quadriga layout object:
QL = build_Quadriga_layout(PAR);

%%
%START

H = zeros(PAR.mod.Nf,PAR.mod.Nt,PAR.ant.L_BS,2,num_UE);


for i = 1 : num_UE
    
    %generate random UE position:
    QL = generate_rnd_UE_position(QL,LOS_flag);
    
    %generate random UE track:
    QL = generate_rnd_UE_track(QL,UE_speed,PAR);
    
    %generate rnd channel parameters:
    CT = generate_rnd_channel_taps(QL);
    
    %build TF channel:
    H(:,:,:,:,i) = taps_to_TF(CT,PAR);
     
end
    
H = format_UE_ant(H,num_UE,PAR);

