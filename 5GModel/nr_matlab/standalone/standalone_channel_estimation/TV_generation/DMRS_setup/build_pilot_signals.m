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

function PAR = build_pilot_signals(PAR)

%function builds frequency pilot signals using an orthogonal covering code
%(OCC)

%outputs:
%PAR.pilot.OCC_f --> Frequency OCC pilot. Dim: Nf_p x 1
%PAR.pilot.OCC_t --> Time OCC pilot. Dim: Nf_t x 1
%PAR.pilot.Pab  --> total pilot signal in TF domain. a = 0,1 b = 0,1 and
%indicate if the pilot has an OCC in frequency or time respectivally. Dim:
%Nf_p x Nt_p

%%
%PARAMETERS

%modulation parameters:
num_PRB = PAR.cluster.num_PRB; %number of PRB per cluster

%pilot parameters:
Nf_p = PAR.pilot.Nf_p; %number of frequnecy pilots.
Nt_p = PAR.pilot.Nt_p; %number of time pilots.

%%
%FREQUENCY OCC

OCC_f = ones(Nf_p,1);

count = 0;

for i = 1 : num_PRB
    for j = 1 : 2
        for k = 1 : 2
            count = count + 1;
            OCC_f(count) = (-1)^(k - 1);
        end
    end
end

%%
%TIME OCCC

OCC_t = [1 ; -1 ; 1 ; -1];

%%
%PILOT SIGNALS

P00 = ones(Nf_p,1) * ones(Nt_p,1)';
P10 = OCC_f * ones(Nt_p,1)';
P01 = ones(Nf_p,1) * OCC_t';
P11 = OCC_f * OCC_t';

P = zeros(Nf_p,Nt_p,2,2);
P(:,:,1,1) = P00;
P(:,:,1,2) = P01;
P(:,:,2,1) = P10;
P(:,:,2,2) = P11;

% P(:,:,1,2) = zeros(size(P00));
% P(:,:,2,1) = zeros(size(P00));
% P(:,:,2,2) = zeros(size(P00));

%%
%WRAP

PAR.pilot.OCC_f = OCC_f;
PAR.pilot.OCC_t = OCC_t;
PAR.pilot.P = P;

end


