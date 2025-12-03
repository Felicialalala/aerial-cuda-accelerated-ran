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

function [W_noiseIso, noiseDim] = pucch_noiseIso_filter(mu)

%function computes filter which isolates noise from signal

%outputs:
%W_noiseIso --> noise isolation filter. Dim: 8 x 12

%%
%PARAMATERS

noiseDim = 8;
df       = 15e3*2^mu; % subcarrier spacing (Hz);

%%
%START

f = 0 : 11;
f = f' * df;

delaySpread = 2*10^(-6);

%%
%KERNEL

K = sinc_tbf((f - f') * delaySpread);

[V,~]      = eigs(K,12);
W_noiseIso = V(:,12 - noiseDim + 1 : 12);
W_noiseIso = W_noiseIso.';


end 

