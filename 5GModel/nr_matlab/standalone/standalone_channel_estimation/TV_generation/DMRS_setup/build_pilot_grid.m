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

function PAR = build_pilot_grid(PAR)

%build time and frequency pilot grids


%outputs:
%PAR.pilot.pilot_grid_time --> Time pilot grid.
%PAR.pilot.pilot_grid_freq --> Freq pilot grid.


%%
%PARAMETERS

%pilot parameters:
Nf_p = PAR.pilot.Nf_p; %number of frequency pilots

%modulation parameters:
num_PRB = PAR.cluster.num_PRB; %number of PRB per cluster

%%
%FREQ GRID

pilot_grid_freq = zeros(1,Nf_p);

count = 0;

for i = 1 : num_PRB
    for j = 1 : 2
        for k = 1 : 2
            count = count + 1;
            pilot_grid_freq(count) = (i - 1)*12 + 6*(j - 1) + (k - 1);
        end
    end
end

pilot_grid_freq = pilot_grid_freq + 1;

%%
%TIME GRID

pilot_grid_time = [3 4 11 12];

%%
%WRAP

PAR.pilot.pilot_grid_time = pilot_grid_time;
PAR.pilot.pilot_grid_freq = pilot_grid_freq;

%%
%VISUALIZE

if PAR.visualize == 1
    visualize_pilot_grid(PAR);
end

