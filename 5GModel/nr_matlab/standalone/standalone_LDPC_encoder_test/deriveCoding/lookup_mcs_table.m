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

function PuschCfg = lookup_mcs_table(PuschCfg)

%function uses table to lookup qam + coderate from mcs
%following TS 38.214 section 5.1.3.1

%outputs:
%PuschCfg.coding.qamstr   --> selected qam (string)
%PuschCfg.coding.qam      --> bits per qam
%PuschCfg.coding.codeRate --> target code rate

%%
%PARAMATERS

mcs = PuschCfg.coding.mcs;             %mcs index
mcsTable = PuschCfg.coding.mcsTable;   %Choice of mcs table

%%
%READ TABLE

    
switch(mcsTable)
    case 1
    % Uses MCS in Table 5.1.3.1.-1, TS 38.214
    switch(mcs)
        case 100    %Uncoded test mode
            qam = 2; codeRate = 1;

        case 0
            qam = 2; codeRate = 120/1024;
        case 1
            qam = 2; codeRate = 157/1024;
        case 2
            qam = 2; codeRate = 193/1024;
        case 3
            qam = 2; codeRate = 251/1024;
        case 4
            qam = 2; codeRate = 308/1024;
        case 5
            qam = 2; codeRate = 379/1024;
        case 6
            qam = 2; codeRate = 449/1024;
        case 7
            qam = 2; codeRate = 526/1024;
        case 8
            qam = 2; codeRate = 602/1024;
        case 9
            qam = 2; codeRate = 679/1024;

        case 10
            qam = 4; codeRate = 340/1024;
        case 11
            qam = 4; codeRate = 378/1024;
        case 12
            qam = 4; codeRate = 434/1024;
        case 13
            qam = 4; codeRate = 490/1024;
        case 14
            qam = 4; codeRate = 553/1024;
        case 15
            qam = 4; codeRate = 616/1024;
        case 16
            qam = 4; codeRate = 658/1024;

        case 17
            qam = 6; codeRate = 438/1024;
        case 18
            qam = 6; codeRate = 466/1024;
        case 19
            qam = 6; codeRate = 517/1024;
        case 20
            qam = 6; codeRate = 567/1024;
        case 21
            qam = 6; codeRate = 616/1024;
        case 22
            qam = 6; codeRate = 666/1024;
        case 23
            qam = 6; codeRate = 719/1024;
        case 24
            qam = 6; codeRate = 772/1024;
        case 25
            qam = 6; codeRate = 822/1024;
        case 26
            qam = 6; codeRate = 873/1024;
        case 27
            qam = 6; codeRate = 910/1024;
        case 28
            qam = 6; codeRate = 948/1024;
        otherwise
            error("Error! Unsupported MCS");
    end

    case 2
     % Uses MCS in Table 5.1.3.1-2, TS 38.214
     switch(mcs)
        case 100    %Uncoded test mode
            qam = 2; codeRate = 1;

        case 0
            qam = 2; codeRate = 120/1024;
        case 1
            qam = 2; codeRate = 193/1024;
        case 2
            qam = 2; codeRate = 308/1024;
        case 3
            qam = 2; codeRate = 449/1024;
        case 4
            qam = 2; codeRate = 602/1024;
        case 5
            qam = 4; codeRate = 378/1024;
        case 6
            qam = 4; codeRate = 434/1024;
        case 7
            qam = 4; codeRate = 490/1024;
        case 8
            qam = 4; codeRate = 553/1024;
        case 9
            qam = 4; codeRate = 616/1024;
        case 10
            qam = 4; codeRate = 658/1024;
        case 11
            qam = 6; codeRate = 466/1024;
        case 12
            qam = 6; codeRate = 517/1024;
        case 13
            qam = 6; codeRate = 567/1024;
        case 14
            qam = 6; codeRate = 616/1024;
        case 15
            qam = 6; codeRate = 666/1024;
        case 16
            qam = 6; codeRate = 719/1024;
        case 17
            qam = 6; codeRate = 772/1024;
        case 18
            qam = 6; codeRate = 822/1024;
        case 19
            qam = 6; codeRate = 873/1024;
        case 20
            qam = 8; codeRate = 682.5/1024;
        case 21
            qam = 8; codeRate = 711/1024;
        case 22
            qam = 8; codeRate = 754/1024;
        case 23
            qam = 8; codeRate = 797/1024;
        case 24
            qam = 8; codeRate = 841/1024;
        case 25
            qam = 8; codeRate = 885/1024;
        case 26
            qam = 8; codeRate = 916.5/1024;
        case 27
            qam = 8; codeRate = 948/1024;
        otherwise
            error("Error! Unsupported MCS");
     end
end

%%
%QAM

switch qam
    case 2
        qamstr = 'QPSK';
    case 4
        qamstr = '16QAM';
    case 6
        qamstr = '64QAM';
    case 8
        qamstr = '256QAM';
end

%%
%WRAP

PuschCfg.coding.qamstr = qamstr;
PuschCfg.coding.qam = qam;
PuschCfg.coding.codeRate = codeRate;

             
