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

function compute_bit_errors(TbCbs,TbCbs_est,PuschCfg)

%function computes the number of bits errors after LDPC decoding

%inputs:
%TbCbs     --> true transport block bits. Dim: (nV_sym*Zc) x C
%TbCbs_est --> estimated transport block bits. Dim: (nV_sym*Zc) x C

%%
%PARAMATERS

C = PuschCfg.coding.C;   % number of codeblocks

%%
%START

nBitErrors = sum(abs(TbCbs - TbCbs_est),1).';
CodeBlock = (1 : C).';

table(CodeBlock,nBitErrors)


