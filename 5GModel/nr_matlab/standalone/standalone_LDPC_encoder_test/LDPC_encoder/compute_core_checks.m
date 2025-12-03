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

function coreChecks = compute_core_checks(cb,Zc,TannerPar)

%function use data bits to compute the core parity check equations.

%inputs:
%Zc --> lifting size
%cb --> coded codeblock, currently has values for systematic ...
%data bits fixed, but not yet the pairty bits. Dim: Zc x nV

%outputs:
%coreChecks --> current status of core check nodes. Dim: Zc x 4

%%
%START

coreChecks = zeros(Zc,4);

for i = 1 : 4
    coreChecks(:,i) = compute_check(i,cb,Zc,TannerPar);
end

end






