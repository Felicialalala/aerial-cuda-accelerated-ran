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

function QL = build_Quadriga_layout(PAR)

%build Quadriga layout object. (Models antenna arrays for UE and BS. Fixes the propogation evniorment.)

%outputs:
%QL --> Quadriga layout object

%%
%SIMULATION PARAMATERS

s = qd_simulation_parameters;
s.center_frequency = PAR.prop.carrier_frequency; %carrier frequency
s.sample_density = 2; %sample density (2 a good number)

%%
%LAYOUT PARAMATERS

QL = qd_layout(s); %build layout object

QL.tx_position = PAR.geo.tx_position; %location of BS antenna (x,y,z)

%build antennas for BS and UE, see class "qd_array" for details section
%2.2.2 in manual
QL.tx_array = PAR.ant.tx_array; %BS antenna array geometry
QL.rx_array = PAR.ant.rx_array; %UE antenna array

%fix the propogation enviorment:
QL.set_scenario(PAR.prop.scenerio);





