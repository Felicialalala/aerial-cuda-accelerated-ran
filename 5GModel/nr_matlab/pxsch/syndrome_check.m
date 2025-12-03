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

function [pass, nFail] = syndrome_check(APP,nC,Zc,TannerPar)

%function hard slices current APPs and checks if all parity equation are
%satisfied

%inputs:
%APP --> current APP of bits. Dim: Zc x nV
%nC  --> number of parity check nodes to be checked, total parity equations (nC * Zc rows)
%        For full check: nC should be 46 for BG1 and 42 for in 5G-NR
%        For core check: nC = 4, only check the first 4 * Zc rows
%Zc  --> lifting size

%outputs:
%pass   --> 0 or 1. Indicates if parity equations (nC * Zc rows) satisfied
%nFail  --> number of parity equations failures (nC * Zc rows)


%%
%START


%first hard slices data:
x_est = zeros(size(APP));
x_est(APP >= 0) = 0;
x_est(APP < 0) = 1;


%next compute parity checks:
Chk = zeros(Zc,nC);

for i = 1 : nC
    Chk(:,i) = compute_check(i,Zc,x_est,TannerPar);
end

Chk = mod(Chk,2);

% syndrome_check results
nFail = sum(Chk(:));
pass  = (nFail == 0);