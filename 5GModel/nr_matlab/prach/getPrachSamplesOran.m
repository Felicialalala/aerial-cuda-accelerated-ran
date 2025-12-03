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

function y_uv_rx = getPrachSamplesOran(PrachParams, y_uv_rx)

kBar = PrachParams.kBar;
L_RA = PrachParams.L_RA;
Nrep = PrachParams.N_rep;

Nsamp_left = kBar;
if L_RA == 139
    Nsamp_oran = 144;
else
    Nsamp_oran = 864;
end
Nsamp_right = Nsamp_oran - L_RA - kBar;
[Nant, Nsamp] = size(y_uv_rx);
Nsamp = Nsamp/Nrep;

for idxAnt = 1:Nant        
    y_uv_rx_0 = y_uv_rx(idxAnt,:);
    y_uv_rx_1 = reshape(y_uv_rx_0, [Nsamp, Nrep]);
    y_uv_rx_2 = [zeros(Nsamp_left, Nrep); y_uv_rx_1; zeros(Nsamp_right, Nrep)];
    y_uv_rx_3 = reshape(y_uv_rx_2, [1, Nsamp_oran*Nrep]);
    y_uv_rx_oran(idxAnt, :) = y_uv_rx_3;
end
y_uv_rx = y_uv_rx_oran;
    
return
