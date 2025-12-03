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

%script test LDPC encoder, done in seven steps:

%1.) Generate random Pusch configuration for user
%2.) Compute coding paramaters for user
%3.) Generate random transport block for user
%4.) Apply LDPC encoder
%5.) Apply white noise to encoded bits
%6.) Apply LDPC decoder
%7.) Check that data bits are recovered

%%
%START

addpath(genpath('./'));

%1.) Generate a randon pusch cfg for the user:
%PuschCfg = generate_rnd_cfg;
mcsTable = 1;

for mcs = 27:27
    for prballoc = 1
        for nsym=12:1:12
            
            PuschCfg = generate_fixed_cfg(mcs,prballoc,nsym,mcsTable);
            
            %2.) Generate coding paramaters for the user:
            PuschCfg = derive_coding_main(PuschCfg);
            %PuschCfg.coding.nV_parity = 46;
            
            %3.) Generate random transport block
            TbCbs = generate_rnd_TB(PuschCfg);
            
            if PuschCfg.coding.BGN == 1
                PuschCfg.coding.nV_parity = 46;
            else
                PuschCfg.coding.nV_parity = 42;
            end
            
            %4.) Apply LDPC encoder:
            TbCodedCbs = LDPC_encode_main(TbCbs,PuschCfg);
            
            %4.5.) Save output file:
            save_TbCodedCbs(TbCodedCbs, TbCbs, PuschCfg);
            
            %5.) Apply channel:
            LLR = apply_channel_main(TbCodedCbs);
            
            %6) Apply LDPC decoder:
            TbCbs_est = LDPC_decoder_main(LLR,PuschCfg);
            
            %7.) Evaluate results:
            compute_bit_errors(TbCbs,TbCbs_est,PuschCfg);
            
        end
    end
end






