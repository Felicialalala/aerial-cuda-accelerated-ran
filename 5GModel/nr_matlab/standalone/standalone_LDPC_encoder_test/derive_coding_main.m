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

function PuschCfg = derive_coding_main(PuschCfg)

%function derives dmrs paramaters for each user. For each user updates PuschCfg:

%primary paramaters:
%PuschCfg.coding.qamstr     --> selected qam (string)
%PuschCfg.coding.qam        --> bits per qam
%PuschCfg.coding.codeRate   --> target code rate
%PuschCfg.coding.TBS        --> size of transport block
%PuschCfg.coding.BGN        --> 1 or 2. Indicates which base graph used
%PuschCfg.coding.CRC        --> '16' or '24A'. Indicates which CRC polynomial used to encode data
%PuschCfg.coding.B          --> size of CRC encoded transport block
%PuschCfg.coding.C          --> number of codeblocks
%PuschCfg.coding.Zc         --> lifting size
%PuschCfg.coding.i_LS       --> lifting set index
%PuschCfg.coding.K          --> number of systematic bits per codeblock
%PuschCfg.coding.F          --> number of filler bits per codeblock
%PuschCfg.coding.K_prime    --> number of CRC encoded bits per codeblock (no filler bits)
%PuschCfg.coding.nV_parity  --> number of parity nodes


%%
%START

%lookup mcs table (qamstr, qam, and codeRate)
PuschCfg = lookup_mcs_table(PuschCfg);

%Derive transport block size (TBS)
PuschCfg = derive_TB_size(PuschCfg);

%Derive base graph (BGN)
PuschCfg = derive_BGN(PuschCfg);

%Derive CRC (CRC and B)
PuschCfg = derive_CRC(PuschCfg);

%Derive lifting (C, Zc, i_LS, K, F, and K_prime)
PuschCfg = derive_lifting(PuschCfg);

%Derive number of parity nodes (nV_parity)
PuschCfg = derive_parity(PuschCfg);




    
end

