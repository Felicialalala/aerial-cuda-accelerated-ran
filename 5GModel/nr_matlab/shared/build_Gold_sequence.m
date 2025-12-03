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

function c = build_Gold_sequence(c_init,N)

%function builds a Gold sequence

%inputs:
%c_init --> initial seed to Gold sequence
%N      --> length of desired Gold sequence

%outputs:
%c --> Gold sequence. Dim: N x 1

%%
%INIT x1,x2

x1 = zeros(N,1);
x1(1) = 1;

% x2 = zeros(N,1);
% tmp = d2b(c_init).';
% x2(1:length(tmp)) = flip(tmp);
x2 = flip(dec2bin(c_init, N)-'0');

%%
%BUILD x1,x2

Nc = 1600;

for n = 1 : (N + Nc - 31)
    x1(n + 31) = mod(x1(n + 3) + x1(n),2);
    x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n),2);
end

%%
%BUILD GOLD

c = zeros(N,1);

for n = 1 : N
    c(n) = mod(x1(n + Nc) + x2(n + Nc),2);
end
