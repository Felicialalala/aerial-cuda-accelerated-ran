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

function TbCbs_est = MSA_layering(LLR,nC,Zc,nItr,TannerPar)

%function performs min-sum layering on recieved LLRs

%inputs:
%nC --> number of active check nodes
%nV --> number of active variable nodes
%Zc --> lifting size
%LLR --> recieved LLRs. Dim: Zc x nV
%nItr --> max number of iterations

%outputs:
%TbCbs_est --> hard estimate of transmited bits. Dim: Zc x nV

%%
%SETUP

APP = LLR;
C2V = zeros(Zc,nC,19);

%%
%START

for itr = 1 : nItr
    for c = 1 : nC
        
        V2C = compute_V2C(c,APP,C2V,Zc,TannerPar);
        cC2V = compute_cC2V(c,V2C,Zc,TannerPar);
        C2V = update_C2V(c,cC2V,C2V,V2C,TannerPar);
        APP = update_APP(c,APP,C2V,V2C,TannerPar);
        
    end
end

%%
%HARD

TbCbs_est = APP;
TbCbs_est(APP >= 0) = 0;
TbCbs_est(APP < 0) = 1;

        
        

