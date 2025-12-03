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

function y = apply_channel(H,x,gnb,srs,sim)

%function apply channel to transmit signal.

%inputs:
% H    --> channel. Dim: L_gNB x 4 x Nf
% x    --> transmit signal. Dim: 4 x Nf x nSym x nRes
% Nf   --> number of subcarriers
% nSym --> number of SRS symbols
% nRes --> number of SRS resourses

%outputs:
% y    --> SRS recieved by gNB. Dim: Nf x nSym x L_gNB

%%
%PARAMATERS

% gnb paramaters:
Nf = gnb.Nf;      % number of subcarrier
L_gNB = gnb.L;    % number of gNB digital antennas

% srs paramaters:
nRes = srs.nRes;  % number of SRS resourses
nSym = srs.nSym;  % total number of srs symbols 

% sim paramaters:
N0 = sim.N0;      % input noise variance


%%
%START

y = zeros(L_gNB,Nf,nSym);

%apply channel:
for r = 1 : nRes
    for f = 1 : Nf
        for t = 1 : nSym
            y(:,f,t) = y(:,f,t) + H(:,:,f,r) * x(:,f,t,r);
        end
    end
end

%add noise:
y = y + sqrt(N0/2)*(randn(size(y)) + 1i*randn(size(y)));

%reshape:
y = permute(y,[2 3 1]); % now: Nf x nSym x L_gNB


end








