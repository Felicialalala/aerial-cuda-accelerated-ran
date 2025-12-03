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

function [nCompChecks, nCompEncErrs, nCompScrErrs, nTvGen] = testCompGenTV_simplex(caseSet, compTvMode)

% evaluate BER achieved by 5G Toolbox functions
evalBERtoolbox = 1;

if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
elseif nargin == 1
    compTvMode = 'both';
end

switch compTvMode
    case 'both'
        genTvFlag = 1;
        testCompliance = 1;
        evalBER = 1;
    case 'genTV'
        genTvFlag = 1;
        testCompliance = 0;
        evalBER = 0;
    case 'testCompliance'
        genTvFlag = 0;
        testCompliance = 1;
        evalBER = 0;
    case 'evalBER'
        genTvFlag = 0;
        testCompliance = 0;
        evalBER = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

compact_TC  = [61000:61999];
full_TC     = [61000:61999]; % for performance evaluation of Simplex code
selected_TC = [61000:61043, 61088:61129, 61177:61190]; % for cuPHY/cuBB unit testing

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch caseSet
        case 'compact'
            TcToTest = selected_TC;
        case 'full'
            TcToTest = selected_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
end

CFG = {...
% 1-bit information
% TC#    payload      K      E      Qm    nRNTI    nID    SNR
  61000    0       1       1      1      11      1      20     % default
  61001    1       1       1      1      11      1      20     % vary payload
  61002    0       1       2      2      11      1      20     % vary Qm, E
  61003    0       1       4      4      11      1      20     % vary Qm, E
  61004    0       1       6      6      11      1      20     % vary Qm, E
  61005    0       1       8      8      11      1      20     % vary Qm, E
  61006    0       1       2      2      50      1      20     % vary nRNTI
  61007    0       1       2      2      11      99     20     % vary nID
  61008    1       1       2      2      11      1      20     % vary payload, Qm, E
  61009    1       1       4      4      11      1      20     % vary payload, Qm, E
  61010    1       1       6      6      11      1      20     % vary payload, Qm, E
  61011    1       1       8      8      11      1      20     % vary payload, Qm, E
  61012    1       1       2      2      47      67     20     % vary payload, Qm, nRNTI, nID, E
  61013    1       1       4      4      114     7      20     % vary payload, Qm, nRNTI, nID, E
  61014    1       1       6      6      999     462    20     % vary payload, Qm, nRNTI, nID, E
  61015    1       1       8      8      14      2      20     % vary payload, Qm, nRNTI, nID, E
  61016    0       1       8      8      7       111    20     % vary payload, Qm, nRNTI, nID, E
  61017    0       1       3      1      11      1      20     % vary E
  61018    0       1       6      2      11      1      20     % vary E, Qm
  61019    1       1       8      4      47      67     20     % vary payload, Qm, nRNTI, nID, E
  61020    0       1       24     6      999     6      20     % vary payload, Qm, nRNTI, nID, E
  61021    1       1       32     8      453     54     20     % vary payload, Qm, nRNTI, nID, E
  61022    0       1       1      1      11      1      10     % vary SNR
  61023    1       1       1      1      11      1      10     % vary payload, SNR
  61024    0       1       2      2      11      1      10     % vary Qm, E, SNR
  61025    0       1       4      4      11      1      10     % vary Qm, E, SNR
  61026    0       1       6      6      11      1      10     % vary Qm, E, SNR
  61027    0       1       8      8      11      1      10     % vary Qm, E, SNR
  61028    0       1       2      2      50      1      10     % vary nRNTI, SNR
  61029    0       1       2      2      11      99     10     % vary nID, SNR
  61030    1       1       2      2      11      1      10     % vary payload, Qm, E, SNR
  61031    1       1       4      4      11      1      10     % vary payload, Qm, E, SNR
  61032    1       1       6      6      11      1      10     % vary payload, Qm, E, SNR
  61033    1       1       8      8      11      1      10     % vary payload, Qm, E, SNR
  61034    1       1       2      2      47      67     10     % vary payload, Qm, nRNTI, nID, E, SNR
  61035    1       1       4      4      114     7      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61036    1       1       6      6      999     462    10     % vary payload, Qm, nRNTI, nID, E, SNR
  61037    1       1       8      8      14      2      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61038    0       1       8      8      7       111    10     % vary payload, Qm, nRNTI, nID, E, SNR
  61039    0       1       3      1      11      1      10     % vary E, SNR
  61040    0       1       6      2      11      1      10     % vary E, Qm, SNR
  61041    1       1       8      4      47      67     10     % vary payload, Qm, nRNTI, nID, E, SNR
  61042    0       1       24     6      999     6      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61043    1       1       32     8      453     54     10     % vary payload, Qm, nRNTI, nID, E, SNR
  61044    0       1       1      1      11      1       0     % vary SNR
  61045    1       1       1      1      11      1       0     % vary payload, SNR
  61046    0       1       2      2      11      1       0     % vary Qm, E, SNR
  61047    0       1       4      4      11      1       0     % vary Qm, E, SNR
  61048    0       1       6      6      11      1       0     % vary Qm, E, SNR
  61049    0       1       8      8      11      1       0     % vary Qm, E, SNR
  61050    0       1       2      2      50      1       0     % vary nRNTI, SNR
  61051    0       1       2      2      11      99      0     % vary nID, SNR
  61052    1       1       2      2      11      1       0     % vary payload, Qm, E, SNR
  61053    1       1       4      4      11      1       0     % vary payload, Qm, E, SNR
  61054    1       1       6      6      11      1       0     % vary payload, Qm, E, SNR
  61055    1       1       8      8      11      1       0     % vary payload, Qm, E, SNR
  61056    1       1       2      2      47      67      0     % vary payload, Qm, nRNTI, nID, E, SNR
  61057    1       1       4      4      114     7       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61058    1       1       6      6      999     462     0     % vary payload, Qm, nRNTI, nID, E, SNR
  61059    1       1       8      8      14      2       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61060    0       1       8      8      7       111     0     % vary payload, Qm, nRNTI, nID, E, SNR
  61061    0       1       3      1      11      1       0     % vary E, SNR
  61062    0       1       6      2      11      1       0     % vary E, Qm, SNR
  61063    1       1       8      4      47      67      0     % vary payload, Qm, nRNTI, nID, E, SNR
  61064    0       1       24     6      999     6       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61065    1       1       32     8      453     54      0     % vary payload, Qm, nRNTI, nID, E, SNR
  61066    0       1       1      1      11      1     -10     % vary SNR
  61067    1       1       1      1      11      1     -10     % vary payload, SNR
  61068    0       1       2      2      11      1     -10     % vary Qm, E, SNR
  61069    0       1       4      4      11      1     -10     % vary Qm, E, SNR
  61070    0       1       6      6      11      1     -10     % vary Qm, E, SNR
  61071    0       1       8      8      11      1     -10     % vary Qm, E, SNR
  61072    0       1       2      2      50      1     -10     % vary nRNTI, SNR
  61073    0       1       2      2      11      99    -10     % vary nID, SNR
  61074    1       1       2      2      11      1     -10     % vary payload, Qm, E, SNR
  61075    1       1       4      4      11      1     -10     % vary payload, Qm, E, SNR
  61076    1       1       6      6      11      1     -10     % vary payload, Qm, E, SNR
  61077    1       1       8      8      11      1     -10     % vary payload, Qm, E, SNR
  61078    1       1       2      2      47      67    -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61079    1       1       4      4      114     7     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61080    1       1       6      6      999     462   -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61081    1       1       8      8      14      2     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61082    0       1       8      8      7       111   -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61083    0       1       3      1      11      1     -10     % vary E, SNR
  61084    0       1       6      2      11      1     -10     % vary E, Qm, SNR
  61085    1       1       8      4      47      67    -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61086    0       1       24     6      999     6     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61087    1       1       32     8      453     54    -10     % vary payload, Qm, nRNTI, nID, E, SNR
% 2-bit information
% TC#    payload     K     E      Qm    nRNTI    nID
  61088  [0 1]       2     3      1      11      1      20     % default
  61089  [1 0]       2     3      1      11      1      20     % vary payload
  61090  [0 1]       2     6      2      11      1      20     % vary Qm, E
  61091  [0 1]       2     12     4      11      1      20     % vary Qm, E
  61092  [0 1]       2     18     6      11      1      20     % vary Qm, E
  61093  [0 1]       2     24     8      11      1      20     % vary Qm, E
  61094  [0 1]       2     3      1      50      1      20     % vary nRNTI
  61095  [0 1]       2     3      1      11      99     20     % vary nID
  61096  [1 0]       2     6      2      11      1      20     % vary payload, Qm, E
  61097  [1 1]       2     12     4      11      1      20     % vary payload, Qm, E
  61098  [0 0]       2     18     6      11      1      20     % vary payload, Qm, E
  61099  [1 0]       2     24     8      11      1      20     % vary payload, Qm, E
  61100  [1 0]       2     6      2      47      67     20     % vary payload, Qm, nRNTI, nID, E
  61101  [1 1]       2     12     4      114     7      20     % vary payload, Qm, nRNTI, nID, E
  61102  [0 0]       2     18     6      999     462    20     % vary payload, Qm, nRNTI, nID, E
  61103  [1 0]       2     24     8      14      2      20     % vary payload, Qm, nRNTI, nID, E
  61104  [0 1]       2     5      1      11      1      20     % vary E
  61105  [0 1]       2     18     2      11      1      20     % vary E? Qm
  61106  [1 1]       2     28     4      114     7      20     % vary payload, Qm, nRNTI, nID, E
  61107  [0 0]       2     48     6      999     462    20     % vary payload, Qm, nRNTI, nID, E
  61108  [1 0]       2     80     8      14      2      20     % vary payload, Qm, nRNTI, nID, E
  61109  [0 1]       2     3      1      11      1      10     % vary SNR
  61110  [1 0]       2     3      1      11      1      10     % vary payload, SNR
  61111  [0 1]       2     6      2      11      1      10     % vary Qm, E, SNR
  61112  [0 1]       2     12     4      11      1      10     % vary Qm, E, SNR
  61113  [0 1]       2     18     6      11      1      10     % vary Qm, E, SNR
  61114  [0 1]       2     24     8      11      1      10     % vary Qm, E, SNR
  61115  [0 1]       2     3      1      50      1      10     % vary nRNTI, SNR
  61116  [0 1]       2     3      1      11      99     10     % vary nID, SNR
  61117  [1 0]       2     6      2      11      1      10     % vary payload, Qm, E, SNR
  61118  [1 1]       2     12     4      11      1      10     % vary payload, Qm, E, SNR
  61119  [0 0]       2     18     6      11      1      10     % vary payload, Qm, E, SNR
  61120  [1 0]       2     24     8      11      1      10     % vary payload, Qm, E, SNR
  61121  [1 0]       2     6      2      47      67     10     % vary payload, Qm, nRNTI, nID, E, SNR
  61122  [1 1]       2     12     4      114     7      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61123  [0 0]       2     18     6      999     462    10     % vary payload, Qm, nRNTI, nID, E, SNR
  61124  [1 0]       2     24     8      14      2      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61125  [0 1]       2     5      1      11      1      10     % vary E, SNR
  61126  [0 1]       2     18     2      11      1      10     % vary E, Qm, SNR
  61127  [1 1]       2     28     4      114     7      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61128  [0 0]       2     48     6      999     462    10     % vary payload, Qm, nRNTI, nID, E, SNR
  61129  [1 0]       2     80     8      14      2      10     % vary payload, Qm, nRNTI, nID, E, SNR
  61130  [0 1]       2     3      1      11      1       0     % vary SNR
  61131  [1 0]       2     3      1      11      1       0     % vary payload, SNR
  61132  [0 1]       2     6      2      11      1       0     % vary Qm, E, SNR
  61133  [0 1]       2     12     4      11      1       0     % vary Qm, E, SNR
  61134  [0 1]       2     18     6      11      1       0     % vary Qm, E, SNR
  61135  [0 1]       2     24     8      11      1       0     % vary Qm, E, SNR
  61136  [0 1]       2     3      1      50      1       0     % vary nRNTI, SNR
  61137  [0 1]       2     3      1      11      99      0     % vary nID, SNR
  61138  [1 0]       2     6      2      11      1       0     % vary payload, Qm, E, SNR
  61139  [1 1]       2     12     4      11      1       0     % vary payload, Qm, E, SNR
  61140  [0 0]       2     18     6      11      1       0     % vary payload, Qm, E, SNR
  61141  [1 0]       2     24     8      11      1       0     % vary payload, Qm, E, SNR
  61142  [1 0]       2     6      2      47      67      0     % vary payload, Qm, nRNTI, nID, E, SNR
  61143  [1 1]       2     12     4      114     7       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61144  [0 0]       2     18     6      999     462     0     % vary payload, Qm, nRNTI, nID, E, SNR
  61145  [1 0]       2     24     8      14      2       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61146  [0 1]       2     5      1      11      1       0     % vary E, SNR
  61147  [0 1]       2     18     2      11      1       0     % vary E, Qm, SNR
  61148  [1 1]       2     28     4      114     7       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61149  [0 0]       2     48     6      999     462     0     % vary payload, Qm, nRNTI, nID, E, SNR
  61150  [1 0]       2     80     8      14      2       0     % vary payload, Qm, nRNTI, nID, E, SNR
  61151  [0 1]       2     3      1      11      1     -10     % vary SNR
  61152  [1 0]       2     3      1      11      1     -10     % vary payload, SNR
  61153  [0 1]       2     6      2      11      1     -10     % vary Qm, E, SNR
  61154  [0 1]       2     12     4      11      1     -10     % vary Qm, E, SNR
  61155  [0 1]       2     18     6      11      1     -10     % vary Qm, E, SNR
  61156  [0 1]       2     24     8      11      1     -10     % vary Qm, E, SNR
  61157  [0 1]       2     3      1      50      1     -10     % vary nRNTI, SNR
  61158  [0 1]       2     3      1      11      99    -10     % vary nID, SNR
  61159  [1 0]       2     6      2      11      1     -10     % vary payload, Qm, E, SNR
  61160  [1 1]       2     12     4      11      1     -10     % vary payload, Qm, E, SNR
  61161  [0 0]       2     18     6      11      1     -10     % vary payload, Qm, E, SNR
  61162  [1 0]       2     24     8      11      1     -10     % vary payload, Qm, E, SNR
  61163  [1 0]       2     6      2      47      67    -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61164  [1 1]       2     12     4      114     7     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61165  [0 0]       2     18     6      999     462   -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61166  [1 0]       2     24     8      14      2     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61167  [0 1]       2     5      1      11      1     -10     % vary E, SNR
  61168  [0 1]       2     18     2      11      1     -10     % vary E, Qm, SNR
  61169  [1 1]       2     28     4      114     7     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61170  [0 0]       2     48     6      999     462   -10     % vary payload, Qm, nRNTI, nID, E, SNR
  61171  [1 0]       2     80     8      14      2     -10     % vary payload, Qm, nRNTI, nID, E, SNR
  % multiple codewords
  61172  [1 0 1 1 1 0 1 0 0 1]    [2 1 2 1 2 2]   [3 1 6 4 12 18]   [1 1 2 4 4 6]   [1 2 3 4 5 6]  [9 10 11 12 13 14]  [10 10 10 10 10 10]
  61173  [1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,1,...
          1,1,0,1,1,1,1,0,0,0,1,1,0,1,1,0,1,0,1,1,...
          1,0,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,1,...
          0,0,1,0,0,1,1,1,1,0,1,1,0,0,0,1,0,0,0,1,...
          0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,0,1,0,...
          0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,...
          0,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,...
          1,0,0,1,1,1,1,1,1,1] ...
         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,...
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,...
          1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,...
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,...
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] ...
         [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,...
          4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,...
          8,8,8,8,8,8,8,8,8,8,3,3,3,3,3,3,3,3,3,3,...
          6,6,6,6,6,6,6,6,6,6,12,12,12,12,12,12,12,12,12,12,...
          18,18,18,18,18,18,18,18,18,18,24,24,24,24,24,24,24,24,24,24] ...
         [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,...
          4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,...
          8,8,8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,...
          2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,...
          6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8] ...
         [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,...
          21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,...
          41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,...
          61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,...
          81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100] ...
         [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,...
          121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,...
          141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,...
          161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,...
          181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200] ...
         [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,...
          10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,...
          10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,...
          10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,...
          10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]

% For cuPHY performance benchmarking
% 1-bit information
% TC#    payload    K      E            Qm    nRNTI    nID      SNR
  61174  [0 1]      [1 1]  [1536 1536]  [1 1] [1 1]    [11 12]  [0 0]
  61175  [0 1]      [1 1]  [1536 1536]  [8 8] [1 1]    [11 12]  [0 0]
% 2-bit information
  61176  [0 1 1 0 1 1 0 0]  [2 2 2 2]  [1536 1536 1536 1536]  [1 1 1 1] [1 1 1 1]    [11 12 13 14]  [0 0 0 0]

% 2-bit tests with E < 3*Qm
  61177  [0 1]       2     4      4      11      1      20     % default for E == Qm, SNR == 20 dB
  61178  [1 1]       2     6      6      11      1      20     % vary Qm, E
  61179  [0 1]       2     8      8      11      1      20     % vary Qm, E
  61180  [0 0]       2     4      2      11      1      20     % default for E == 2*Qm, SNR == 20 dB
  61181  [0 1]       2     8      4      11      1      20     % vary Qm, E
  61182  [1 0]       2     12     6      11      1      20     % vary Qm, E
  61183  [0 1]       2     16     8      11      1      20     % vary Qm, E
  61184  [1 1]       2     4      4      11      1      10     % default for E == Qm, SNR == 10 dB
  61185  [1 0]       2     6      6      11      1      10     % vary Qm, E
  61186  [0 0]       2     8      8      11      1      10     % vary Qm, E
  61187  [0 1]       2     4      2      11      1      10     % default for E == 2*Qm, SNR == 10 dB
  61188  [1 0]       2     8      4      11      1      10     % vary Qm, E
  61189  [1 1]       2     12     6      11      1      10     % vary Qm, E
  61190  [1 1]       2     16     8      11      1      10     % vary Qm, E
  61191  [0 1]       2     4      4      11      1      0      % default for E == Qm, SNR == 0 dB
  61192  [1 0]       2     6      6      11      1      0      % vary Qm, E
  61193  [0 1]       2     8      8      11      1      0      % vary Qm, E
  61194  [0 0]       2     4      2      11      1      0      % default for E == 2*Qm, SNR == 0 dB
  61195  [0 1]       2     8      4      11      1      0      % vary Qm, E
  61196  [1 1]       2     12     6      11      1      0      % vary Qm, E
  61197  [0 1]       2     16     8      11      1      0      % vary Qm, E
  61198  [0 0]       2     4      4      11      1      -10    % default for E == Qm, SNR == -10 dB
  61199  [1 0]       2     6      6      11      1      -10    % vary Qm, E
  61200  [1 1]       2     8      8      11      1      -10    % vary Qm, E
  61201  [0 1]       2     4      2      11      1      -10    % default for E == 2*Qm, SNR == -10 dB
  61202  [1 1]       2     8      4      11      1      -10    % vary Qm, E
  61203  [0 1]       2     12     6      11      1      -10    % vary Qm, E
  61204  [1 0]       2     16     8      11      1      -10    % vary Qm, E
};

[NallTest, ~] = size(CFG);
nCompEncErrs    = 0;
nCompScrErrs    = 0;
nCompChecks     = 0;
nTvGen          = 0;

tvDirName = 'GPU_test_input';
fp16AlgoSel = 0;

parfor i = 1:NallTest
    caseNum = CFG{i, 1};
    if ismember(caseNum, TcToTest)
        rng(caseNum);
        nTvGen = nTvGen + genTvFlag;

        numCW = length(CFG{i, 3});

        payload = CFG{i, 2};
        K       = CFG{i, 3};
        E       = CFG{i, 4};
        Qm      = CFG{i, 5};
        nRNTI   = CFG{i, 6};
        nID     = CFG{i, 7};
        snr = CFG{i, 8};

        PLstart = 0;

        dscrmLLR_cell =  cell(numCW,1);
        cbUint_vec    =  zeros(numCW,1);

        for cwIdx = 1:numCW
            K_CW = K(cwIdx);
            E_CW = E(cwIdx);
            Qm_CW = Qm(cwIdx);
            payload_CW = payload(PLstart+1:PLstart+K_CW);
            PLstart = PLstart + K_CW;
            nRNTI_CW = nRNTI(cwIdx);
            nID_CW = nID(cwIdx);
            snr_CW = snr(cwIdx);

            outEncode = simplexEncode(payload_CW, K_CW, E_CW, Qm_CW);
            outScr = scramble(outEncode, E_CW, nRNTI_CW, nID_CW, 0, 0);
            outScrLLR = apply_simplex_channel(snr_CW, Qm_CW, outScr);
            outDescrLLR = SimplexDescramble(outScrLLR, K_CW, E_CW, Qm_CW, nRNTI_CW, nID_CW, 0, 0);
          
            % save buffers
            cb = 0;
            for bitIdx = 0 : (K_CW - 1)
                cb = cb + 2^bitIdx * payload_CW(bitIdx + 1);
            end
            cbUint_vec(cwIdx)    =  cb;
            dscrmLLR_cell{cwIdx} =  outDescrLLR;
                
            
            if evalBER

                QmSeq = [];
                switch Qm_CW
                    case 1
                        QmSeq = 'pi/2-BPSK';
                    case 2
                        QmSeq = 'QPSK';
                    case 4
                        QmSeq = '16QAM';
                    case 6
                        QmSeq = '64QAM';
                    case 8
                        QmSeq = '256QAM';
                end

                [berImpl, ber5GToolbox] = testSimplex(Qm_CW, K_CW, E_CW, snr_CW, 5000, 0, nRNTI_CW, nID_CW, evalBERtoolbox);

                fprintf('\n BER evaluation for test case %d: %s, K = %d, E = %d, SNR = %d dB, BER = %f%%, BER(toolbox) = %f%% \n', ...
                    caseNum, QmSeq, K_CW, E_CW, snr_CW, berImpl*100, ber5GToolbox*100);
            end


            if(testCompliance)
                nCompChecks = nCompChecks + 1;
                QmSeq = [];
                switch Qm_CW
                    case 1
                        QmSeq = 'pi/2-BPSK';
                    case 2
                        QmSeq = 'QPSK';
                    case 4
                        QmSeq = '16QAM';
                    case 6
                        QmSeq = '64QAM';
                    case 8
                        QmSeq = '256QAM';
                end

                ToolboxOutEnc = nrUCIEncode(int8(transpose(payload_CW)), E_CW, QmSeq);


                err_bit = sum(abs(int8(outEncode)-ToolboxOutEnc));
                if err_bit > 0
                    nCompEncErrs = nCompEncErrs + 1;
                    fprintf('\n complience error with simplex encoder in test case: %d, codeword number: %d \n', caseNum, cwIdx);
                end

                ToolboxOutScr = nrPUSCHScramble(int8(outEncode), nID_CW, nRNTI_CW);

                err_bit = sum(abs(outScr-ToolboxOutScr));
                if err_bit > 0
                    nCompScrErrs = nCompScrErrs + 1;
                    fprintf('\n complience error with scrambler in test case: %d, codeword number: %d \n', caseNum, cwIdx);
                end
            end
        end

        if genTvFlag
            TVname = [sprintf('TVnr_%05d', caseNum), '_SIMPLEX_gNB_CUPHY_s0p0'];
            saveTV_Simplex_cuphy(tvDirName, TVname, fp16AlgoSel, numCW, cbUint_vec, K, E, Qm, dscrmLLR_cell);
        end

    end
end

fprintf('\n Generated %d test vectors', nTvGen);
fprintf('\n Performed %d complience tests, finding %d encoder errors, and %d scrambler errors \n\n', nCompChecks, nCompEncErrs, nCompScrErrs);

end


function saveTV_Simplex_cuphy(tvDirName, TVname, fp16AlgoSel, numCW, cbUint_vec, K, E, Qm, dscrmLLR_cell)
%%create h5 file
[status,msg]  = mkdir(tvDirName);
h5File        = H5F.create([tvDirName filesep TVname '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

SimplexCwPar.numCW = uint32(numCW);

hdf5_write_nv_exp(h5File, 'SimplexCwPar', SimplexCwPar);

SimplexPar = [];

for cwIdx = 1:numCW
    SimplexPar(cwIdx).K           = uint8(K(cwIdx));
    SimplexPar(cwIdx).E           = uint32(E(cwIdx));
    SimplexPar(cwIdx).nBitsPerQam = uint8(Qm(cwIdx));
    
    
    dscrmLLR_cw = dscrmLLR_cell{cwIdx};
    dscrmLLR_cw = single(dscrmLLR_cw);
    cwNameStr = strcat('cwLLRs',num2str(cwIdx - 1));
    hdf5_write_nv(h5File, cwNameStr, dscrmLLR_cw);
end


hdf5_write_nv_exp(h5File, 'SimplexPar', SimplexPar);
hdf5_write_nv(h5File, 'cb_vec', uint32(cbUint_vec));
end
