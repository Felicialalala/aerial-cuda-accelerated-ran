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

function [scrseq,LLR_descr] = descramble_bits(LLRseq, N_id, n_rnti)

%function computes user's scrambling sequence and applies it to bits.
%Following: TS 38.211 section 7.3.1.1

%inputs:
%TbRateMatCbs --> input bits

%outputs:
%scrseq       --> users scrambling sequence
%TbScramCbs   --> scrambled bits

%PARAMATERS

%gnb paramaters:
% N_id = alloc.dataScramblingId;       % data scrambling id

%PUSCH paramaters:
% n_rnti = alloc.RNTI;      % user rnti paramater

%BUILD SEQUENCE

%first compute seed:
c_init = n_rnti * 2^15 + N_id;

%compute sequence:
scrseq = build_Gold_sequence(c_init,length(LLRseq));

%APPLY SEQUENCE

LLR_descr =  (1 - 2 * scrseq) .*LLRseq;

return
