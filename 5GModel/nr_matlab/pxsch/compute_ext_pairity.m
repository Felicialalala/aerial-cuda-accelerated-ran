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

function ExtPairity = compute_ext_pairity(CodedCb,TannerPar,Zc)

%inputs:
%CodedCb --> coded codeblock, currently has values for systematic ...
%data and core pairty bits fixed, but not yet the extented pairty bits. Dim: Zc x nV

%outputs:
%ExtPairity --> extended pairity bits. Dim: Zc x nV_ext

%PARAMATERS

%coding paramaters:
% Zc = coding.Zc; %lifting size

%tanner graph paramaters:
nV = TannerPar.nV;          %number of variable nodes
nV_sym = TannerPar.nV_sym;  %number of systematic variable nodes
nV_ext = nV - (nV_sym + 4); %number of extended pairity nodes

%START

ExtPairity = zeros(Zc,nV_ext);

for i = 1 : nV_ext
    ExtPairity(:,i) = compute_check(4 + i,Zc,CodedCb,TannerPar);
end

return
