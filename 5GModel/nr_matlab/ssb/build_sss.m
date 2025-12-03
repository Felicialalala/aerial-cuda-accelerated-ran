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

function d_sss = build_sss(N_id)

% function builds the Secondary Synchronization Sequence (SSS)

%inputs:
% N_id  --> physical cell id

%outputs:
% d_sss --> sss. Dim: 127 x 1

%%
%START

N_id2 = mod(N_id,3);
N_id1 = (N_id - N_id2) / 3;

load('sss_x_seq.mat');

m0 = 15*floor(N_id1/112) + 5*N_id2;
m1 = mod(N_id1,112);

idx0 = mod((0:126) + m0,127);
idx1 = mod((0:126) + m1,127);

d_sss = x0(idx0 + 1) .* x1(idx1 + 1);

end
