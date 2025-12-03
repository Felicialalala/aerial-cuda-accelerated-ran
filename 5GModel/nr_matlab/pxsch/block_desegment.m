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

function [TbCrc_est,cbErr] = block_desegment(TbCbs_est, C, K_prime, table)

%function does the following:

%1.) Removes filler bits from ldpc output
%2.) Performs crc on ldpc output, removes crc bits
%3.) Reshapes bits to an estimate of the crc-encoded transport block

%inputs:
%TbCbs_est --> ldpc estimate of transmited data bits

%outputs:
%cbErr     --> crc results of each codeblock
%TbCrc_est --> estimate of crc encoded transport block

%%
%SETUP

if C == 1
    TbCrc_est = zeros(K_prime,1);
else
    TbCrc_est = zeros(K_prime - 24,C);
end

cbErr = zeros(C,1);

%%
%START

%first, remove filler bits:
TbCbs_est = TbCbs_est(1 : K_prime,:);

%next, if C > 1 decode crc
if C > 1
    for c = 1 : C
        [TbCrc_est(:,c),cbErr(c)] = CRC_decode(TbCbs_est(: , c),'24B',table);
%           [TbCrc_est(:,c),cbErr(c)] = crc_decode_mex(TbCbs_est(: , c),'24B');
    end
else
    TbCrc_est = TbCbs_est;
end


%finally, reshape bits into crc-encoded transport block:
TbCrc_est = TbCrc_est(:);

