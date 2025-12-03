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

function ssbIdx = genSsbIdx_r001(N_id)

%function derivies and loads indicies for SS block

%inputs:
% N_id      --> physical cell id.

%ouputs:
% dmrs_idx  -->  indicies of dmrs payload. Dim: 144 x 1
% qam_idx   -->  indicies of qam payload.  Dim: 432 x 1
% pss_idx   -->  indicies for pss. Dim: 127 x 1 
% sss_idx   -->  indicies for sss. Dim: 127 x 1
% note: all indicies are 0 based

%%
%PSS/SSS

% these are always the same:
pss_idx = 56 : 182;
sss_idx = (56 : 182) + 240*2;


%%
%DMRS/QAMs

v = mod(N_id,4);

basic_dmrs_idx = [0 4 8] +  v;
basic_qam_idx = 0:11;
basic_qam_idx(basic_dmrs_idx+1) = [];

dmrs_idx = zeros(144,1);
qam_idx = zeros(432,1);

%compute indicies for 2nd SS block symbol:
for i = 1 : 20
    dmrs_idx(3*(i-1)+1:3*i) = basic_dmrs_idx + 12*(i-1) + 240;
    qam_idx(9*(i-1)+1:9*i) = basic_qam_idx + 12*(i-1) + 240;
end

% compute indicies for 3rd SS block symbol:
for i = 1 : 4
    dmrs_idx( (3*(i-1)+1:3*i) + 20*3) = basic_dmrs_idx + 12*(i-1) + 2*240;
    qam_idx( (9*(i-1)+1:9*i) + 20*9) = basic_qam_idx + 12*(i-1) + 2*240;
end

for i = 1 : 4
    dmrs_idx((3*(i-1)+1:3*i) + 24*3) = basic_dmrs_idx + 12*(i-1) + 2*240 + 192;
    qam_idx((9*(i-1)+1:9*i) + 24*9) = basic_qam_idx + 12*(i-1) + 2*240 + 192;
end

% compute indicies for 4th SS block symbol:
for i = 1 : 20
    dmrs_idx((3*(i-1)+1:3*i) + 28*3) = basic_dmrs_idx + 12*(i-1) + 3*240;
    qam_idx((9*(i-1)+1:9*i) + 28*9) = basic_qam_idx + 12*(i-1) + 3*240;
end

ssbIdx.dmrs_idx = dmrs_idx(:);
ssbIdx.qam_idx = qam_idx(:);
ssbIdx.pss_idx = pss_idx(:);
ssbIdx.sss_idx = sss_idx(:);

    
