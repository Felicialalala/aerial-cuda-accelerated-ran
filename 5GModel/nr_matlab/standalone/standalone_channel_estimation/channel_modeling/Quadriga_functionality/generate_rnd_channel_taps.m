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

function CT = generate_rnd_channel_taps(QL)

%function generate random channel taps.

%inputs:
%QL --> Quadriga layout class

%outputs:
%CT.no_rxant --> number of UE antennas
%CT.no_txant --> number of Hub antennas
%CT.no_path --> number of propogation paths
%CT.coeff --> antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt
%CT.delay --> delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt

%%
%START

p = QL.init_builder;
p.gen_ssf_parameters;
CT = p.get_channels;

end

