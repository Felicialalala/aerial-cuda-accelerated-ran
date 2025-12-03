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

function [nComp, errCnt, nTV, detErr] = testCompGenTV_pdcch(caseSet, compTvMode, subSetMod, relNum)

tic;
if nargin == 0
    caseSet = 'full';
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 1
    compTvMode = 'both';
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 2
    subSetMod = [0, 1];
    relNum = 10000;
elseif nargin == 3
    relNum = 10000;
end

switch compTvMode
    case 'both'
        genTV = 1;
        testCompliance = 1;
    case 'genTV'
        genTV = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTV = 0;
        testCompliance = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

selected_TC = [2001:2999];
disabled_TC = [];
[~,TcIdx] = ismember(disabled_TC, selected_TC);
selected_TC(TcIdx) = [];

compact_TC = [2001:2999];
full_TC = [2001:2999];

if isnumeric(caseSet)
    TcToTest = caseSet;
else    
    switch caseSet
        case 'compact'
            TcToTest = compact_TC;
        case 'full'
            TcToTest = full_TC;
        case 'selected'
            TcToTest = selected_TC;
        otherwise
            error('caseSet is not supported...\n');
    end
end

CFG = {...
    % change only one parameter from the base case    
    % TC# slotIdx nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS rnti scrbId scrbRnti aggrL dbQam dbDmrs Npaylod
    2001,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % base case
    2002,   1,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % slotIdx > 0
    2003,   0,      48,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nBWP < 273
    2004,   0,      48,114,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nBWP0 > 0
    2005,   0,     273,  0,   1,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % sym0 > 0
    2006,   0,     273,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % Nsym = 2
    2007,   0,     273,  0,   0,    3,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % Nsym = 3
    2008,   0,     273,  0,   0,    1,    0,    1,    6,    2,  41,     1,    1,65535,    41,     0,     2,      0,     0,   39; % crstIdx = 0
    2009,   0,     273,  0,   0,    1,    2,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % crstIdx = 2    
    2010,   0,     273,  0,   0,    1,    1,    1,    6,    2,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % intl = 1   
    2011,   0,     273,  0,   0,    3,    1,    1,    3,    2,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nBndl = 3    
    2012,   0,     273,  0,   0,    1,    1,    1,    2,    2,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nBnd = 2       
    2013,   0,     273,  0,   0,    3,    1,    1,    6,    3,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nIntl = 3   
    2014,   0,     273,  0,   0,    3,    1,    1,    6,    6,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nIntl = 6        
    2015,   0,     273,  0,   0,    1,    1,    1,    6,    2,  41,     1,    0,    0,     0,     0,     2,      0,     0,   39; % nShift > 0
    2016,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,    0,     0,     0,     2,      0,     0,   39; % isCSS = 1       
    2017,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0, 2000,     0,     0,     2,      0,     0,   39; % rnti > 0
    2018,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,    41,     0,     2,      0,     0,   39; % scrbId > 0
    2019,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     3,     2,      0,     0,   39; % scrbRnti > 0
    2020,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     1,      0,     0,   39; % aggrL = 1
    2021,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     4,      0,     0,   39; % aggrL = 4
    2022,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     8,      0,     0,   39; % aggrL = 8
    2023,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      3,     3,   39; % dbQam
    2024,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,     -3,    -3,   39; % dbDmrs
    2025,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     8,      0,     0,  140; % Npayload = 140  
    2026,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     8,      0,     0,   39; % coresetMap
    2027,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     8,      0,     0,   39; % coresetMap
    2028,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     2,    0,    0,     0,     0,     0,      0,     0,   39; % nDCI > 1
    2029,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     8,      0,     0,   12; % Npayload = 12   
    2030,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,    16,      0,     0,   39; % aggrL = 16
    2031,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % precoding
    2032,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      3,     3,   39; % modComp
    2033,   0,      48,108,   0,    1,    0,    1,    6,    2,  41,     1,    1,65535,    41,     0,     8,      0,     0,   39; % batching w/ multiple coresets
    2034,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 4-BWP
    2035,   0,      48,108,   0,    2,    0,    1,    6,    2,  41,     1,    1,65535,    41,     0,     8,      0,     0,   39; % 1+16+16 DCIs
    2036,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,     1,    0,   30,    10,    20,     1,      0,     0,   39; % testModel
    2037,   0,     273,  0,   0,    1,    1,    1,    6,    2,   0,     1,    0,    0,     0,     0,     4,      3,     3,   39; % modComp, intl = 1       

    % change multiple parameters from the base case 
    % TC# slotIdx nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS rnti scrbId scrbRnti aggrL dbQam dbDmrs Npaylod    
    2101,   2,      90, 12,   2,    2,    2,    1,    2,    3,   1,     1,    1,   30,    10,    20,     1,      0,     0,   40; % intl = 1, aggrL = 1
    2102,   3,     273,  0,   3,    1,    1,    1,    2,    2,   3,     1,    0,   30,    10,    20,     2,      0,     0,   43; % intl = 1, aggrL = 2
    2103,   0,      90, 12,   1,    3,    2,    1,    3,    6,   1,     1,    0,   30,    10,    20,     4,      0,     0,   40; % intl = 1, aggrL = 4
    2104,   1,      90, 12,   0,    2,    0,    1,    6,    2,  41,     1,    1,   30,    10,     0,     8,      0,     0,   43; % intl = 1, aggrL = 8
    2105,   1,     273,  0,   0,    2,    1,    1,    6,    2,  41,     8,    1,   30,    10,     0,     0,      0,     0,   43; % intl = 1, nDCI = 8
    2106,   2,      90, 12,   2,    2,    2,    0,    6,    0,   1,     1,    1,   30,    10,    20,     1,      0,     0,   40; % intl = 0, aggrL = 1
    2107,   3,     273,  0,   3,    1,    1,    0,    6,    0,   0,     1,    0,   30,    10,    20,     2,      0,     0,   43; % intl = 0, aggrL = 2
    2108,   0,      90, 12,   1,    3,    2,    0,    6,    0,   0,     1,    0,   30,    10,    20,     4,      0,     0,   40; % intl = 0, aggrL = 4
    2109,   1,      90, 12,   0,    2,    1,    0,    6,    0,   0,     1,    1,   30,    10,     0,     8,      0,     0,   43; % intl = 0, aggrL = 8
    2110,   1,     273,  0,   0,    2,    1,    0,    6,    0,   0,     8,    1,   30,    10,     0,     0,      0,     0,   43; % intl = 0, nDCI = 8
    2111,   1,      90, 12,   0,    2,    0,    1,    6,    2,  41,     1,    1,   30,    10,     0,    16,      0,     0,   43; % intl = 1, aggrL = 16
    2112,   1,     260, 12,   0,    2,    1,    0,    6,    0,   0,     1,    1,   30,    10,     0,    16,      0,     0,   43; % intl = 0, aggrL = 16

    % different BW
    % mu = 1
    % TC# slotIdx nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS rnti scrbId scrbRnti aggrL dbQam dbDmrs Npaylod    
    2201,   0,      11,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 5 MHz
    2202,   0,      24,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 10 MHz
    2203,   0,      38,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 15 MHz
    2204,   0,      51,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 20 MHz
    2205,   0,      65,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 25 MHz
    2206,   0,      78,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 30 MHz
    2207,   0,     106,  0,   0,    2     1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 40 MHz
    2208,   0,     133,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 50 MHz
    2209,   0,     162,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 60 MHz
    2210,   0,     189,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 70 MHz
    2211,   0,     217,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 80 MHz
    2212,   0,     245,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 90 MHz
    2213,   0,     273,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 100 MHz
    % mu = 0
    2214,   0,      25,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 5 MHz
    2215,   0,      52,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 10 MHz
    2216,   0,      79,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 15 MHz
    2217,   0,     106,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 20 MHz
    2218,   0,     133,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 25 MHz
    2219,   0,     160,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 30 MHz
    2220,   0,     216,  0,   0,    2     1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 40 MHz
    2221,   0,     270,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 50 MHz
    % additional BW test cases
    2222,   0,     100,  0,   0,    2     1,    0,    6,    0,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % 40 MHz
    2223,   1,      24,  0,   0,    2,    1,    0,    6,    0,   0,     1,    1,  211,    41,     0,     8,      0,     0,   39; % 10 MHz
    2224,  11,      24,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,20000,    41,     0,     2,      0,     0,   39; % 10 MHz
    2225,   2,      18,  6,   0,    2,    0,    1,    6,    2,  41,     1,    1,65535,    41,     0,     4,      0,     0,   39; % 10 MHz
    2226,   0,      24,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,    3,     7,     4,     1,      0,     0,   45; % 10 MHz
    2227,   1,      24,  0,   0,    2,    1,    0,    6,    0,   0,     1,    0,20001,     0,     4,     4,      3,     3,   44; % 10 MHz
    
    % Perf test vectors
    % F14 PDCCH
    2801,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,    91,    0,    0,     0,     0,     0,      0,     0,   39; % full cell (91 DCIs)
    2802,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,    25,    0,    0,     0,     0,     0,      0,     0,   39; % average cell (25 DCIs)
    % F08 PDCCH
    2803,   0,     273,  0,   0,    1,    2,    0,    6,    0,   0,    40,    0,    0,     0,     0,     0,      0,     0,   39; % full cell (40 DCIs)
    2804,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,    12,    0,    0,     0,     0,     0,      0,     0,   39; % average cell (12 DCIs)
    % F09 PDCCH
    2805,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,     6,    0,    0,     0,     0,     0,      0,     0,   39; % full cell (6 DCIs)
%     2806,   0,     273,  0,   0,    2,    2,    0,    6,    0,   0,    12,    0,    0,     0,     0,     0,      0,     0,   39; % average cell (12 DCIs)
    % F08 but 20 DCIs for half traffic
    2807,   0,     273,  0,   0,    1,    2,    0,    6,    0,   0,    20,    0,    0,     0,     0,     0,      0,     0,   39; 
    % F08 relaxed complexity, 12 DCIs
    2808,   0,     273,  0,   0,    1,    2,    0,    6,    0,   0,    12,    0,    0,     0,     0,     0,      0,     0,   39; 
    % F08 FDD
    2809,   0,      53,  0,   0,    2,    2,    0,    6,    0,   0,    12,    0,    0,     0,     0,     0,      0,     0,   39; % full cell (12 DCIs)
    % F14 PDCCH 16 DCI
    2810,   0,     273,  0,   0,    1,    2,    0,    6,    0,   0,    16,    0,    0,     0,     0,     0,      0,     0,   39; 
    % F14 PDCCH 16 DCI, 20 MHz, 2 symbols
    2811,   0,      51,  0,   0,    2,    2,    0,    6,    0,   0,    16,    0,    0,     0,     0,     0,      0,     0,   39; 
   
    
    % specific configuration for TV generation
    % TC# slotIdx nBWP BWP0 sym0  Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS rnti scrbId scrbRnti aggrL dbQam dbDmrs Npaylod    
    2901,   1,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,  211,    41,     0,     4,      0,     0,   39; % demo_msg2
    2902,  11,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,20000,    41,     0,     4,      0,     0,   39; % demo_msg4
    2903,   2,      48,108,   0,    1,    0,    1,    6,    2,  41,     1,    1,65535,    41,     0,     8,      0,     0,   39; % demo_coreset0
    2904,   0,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    1,    3,     7,     4,     4,      0,     0,   45; % demo_traffic_dl
    2905,   1,     273,  0,   0,    1,    1,    0,    6,    0,   0,     1,    0,20001,     0,     4,     4,      3,     3,   44; % demo_msg5
    % requested TCs
    2906,   0,     273,  0,   0,    1,    1,    1,    6,    3,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % intl = 3
    2907,   0,     273,  0,   0,    1,    1,    1,    6,    6,   0,     1,    0,    0,     0,     0,     2,      0,     0,   39; % intl = 6
    2908,   0,     273,  0,   0,    2,    1,    1,    6,    3,  41,     8,    0,   30,    10,     0,     0,      0,     0,   43; % intl = 3, nDCI = 8
    2909,   0,     273,  0,   0,    2,    1,    1,    6,    6,  41,     8,    0,   30,    10,     0,     0,      0,     0,   43; % intl = 6, nDCI = 8
    };

[NallTest, ~] = size(CFG);
errCnt = 0;
detErr = 0;
nTV = 0;
nComp = 0;

if (isnumeric(caseSet))
    caseSetStr = num2str(caseSet);
else
    caseSetStr = caseSet;
end
fprintf('PDCCH: genTV = %d, testCompliance = %d, caseSet = %s', genTV, testCompliance, caseSetStr);
fprintf('\nTC# slotIdx nBWP BWP0 sym0 Nsym crstIdx intl nBndl nIntl nShift nDCI isCSS  rnti scrbId scrbRnti aggrL dbQam dbDmrs Npaylod PASS Det\n');
fprintf('------------------------------------------------------------------------------------------------------------------------------------\n');

parfor n = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.pdcch = 1;
    idxSet = n;
    caseNum = CFG{idxSet, 1};
    aggrL_display = '0';
    if ismember(caseNum, TcToTest) && (mod(caseNum, subSetMod(2)) == subSetMod(1))
        rng(caseNum);
        SysPar = initSysPar(testAlloc);
        SysPar.SimCtrl.relNum = relNum;
        SysPar.SimCtrl.N_frame = 1;
        if genTV
            nTV = nTV + 1;
            SysPar.SimCtrl.genTV.enable = 1;
            SysPar.SimCtrl.genTV.TVname = sprintf('TVnr_%04d', caseNum);
            if ismember(caseNum, disabled_TC)
                SysPar.SimCtrl.genTV.TVname = sprintf('disabled_TVnr_%04d', caseNum);
            end
            SysPar.SimCtrl.N_slot_run = CFG{idxSet, 2} + 1;
%             SysPar.SimCtrl.genTV.forceSlotIdxFlag = 1;
            SysPar.SimCtrl.genTV.slotIdx = CFG{idxSet, 2};
        end                
        SysPar.pdcch{1}.BWPSize = CFG{idxSet, 3};
        SysPar.pdcch{1}.BWPStart = CFG{idxSet, 4};
        SysPar.pdcch{1}.StartSymbolIndex = CFG{idxSet, 5};
        SysPar.pdcch{1}.DurationSymbols = CFG{idxSet, 6};
        SysPar.pdcch{1}.coresetIdx = CFG{idxSet, 7};
        SysPar.pdcch{1}.CceRegMappingType = CFG{idxSet, 8};
        SysPar.pdcch{1}.RegBundleSize =  CFG{idxSet, 9};
        SysPar.pdcch{1}.InterleaverSize =  CFG{idxSet, 10};
        SysPar.pdcch{1}.ShiftIndex =  CFG{idxSet, 11};
        SysPar.pdcch{1}.numDlDci =  CFG{idxSet, 12};      
        SysPar.pdcch{1}.isCSS = CFG{idxSet, 13};    
     
        for idxDCI = 1:SysPar.pdcch{1}.numDlDci
            SysPar.pdcch{1}.DCI{idxDCI}.RNTI = CFG{idxSet, 14}+idxDCI-1;
            SysPar.pdcch{1}.DCI{idxDCI}.ScramblingId = CFG{idxSet, 15}+idxDCI-1;
            SysPar.pdcch{1}.DCI{idxDCI}.ScramblingRNTI = CFG{idxSet, 16}+idxDCI-1;
            if SysPar.pdcch{1}.numDlDci > 1
                if caseNum == 2801
                    SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 1;
                    SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = idxDCI-1;
                elseif caseNum == 2802
                    if idxDCI <= 14
                        SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 2;
                        SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 2*(idxDCI-1);
                    elseif idxDCI <= 14+8
                        SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 4;
                        SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 4*(idxDCI-1-14) + 28;
                    else
                        SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 8;
                        SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 8*(idxDCI-1-14-8) + 28 + 32;
                    end
                elseif ismember(caseNum, [2803, 2805, 2807, 2808, 2810, 2811])
                    SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 1;
                    SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = idxDCI-1;
                elseif ismember(caseNum, [2804, 2806])
                    if idxDCI <= 5
                        SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 2;
                        SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 2*(idxDCI-1);
                    elseif idxDCI <= 5+5
                         SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 4;
                         SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 4*(idxDCI-1-5) + 10;
                    else
                         SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 8;
                         SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = 8*(idxDCI-1-5-5) + 10 + 20;
                    end
                else
                    SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 2^mod(idxDCI,4);
                end
                aggrL_display = 'M';
            else
                SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = CFG{idxSet, 17};
                aggrL_display = num2str(CFG{idxSet, 17});
            end
            
            SysPar.pdcch{1}.DCI{idxDCI}.beta_PDCCH_1_0 = CFG{idxSet, 18} + 8; 
            SysPar.pdcch{1}.DCI{idxDCI}.powerControlOffsetSS = CFG{idxSet, 18}/3 + 1; 
            SysPar.pdcch{1}.DCI{idxDCI}.powerControlOffsetSSProfileNR = CFG{idxSet, 18};
            SysPar.pdcch{1}.DCI{idxDCI}.PayloadSizeBits =  CFG{idxSet, 20} + (idxDCI-1);
 
            if caseNum == 2026 
                SysPar.pdcch{1}.coresetMap = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
            elseif caseNum == 2027 
                SysPar.pdcch{1}.coresetMap = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];                
            elseif caseNum == 2030 
                SysPar.pdcch{1}.coresetMap = ones(1, 32);       
            elseif caseNum == 2031 
                SysPar.pdcch{1}.DCI{idxDCI}.prcdBf = mod(idxDCI, 2) + 3;
            elseif caseNum == 2032
                SysPar.SimCtrl.genTV.fhMsgMode = 2; 
            elseif caseNum == 2037
                SysPar.SimCtrl.genTV.fhMsgMode = 2; 
                SysPar.pdcch{1}.coresetMap = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]; 
            elseif caseNum == 2033 % coreset0 (1 DCI) + coreset1 (2 DCIs) 
                SysPar.pdcch{2} = SysPar.pdcch{1};
                SysPar.pdcch{2}.BWPSize = 273;
                SysPar.pdcch{2}.BWPStart = 0;
                SysPar.pdcch{2}.StartSymbolIndex = 0;
                SysPar.pdcch{2}.DurationSymbols = 1;
                SysPar.pdcch{2}.coresetIdx = 1;
                SysPar.pdcch{2}.CceRegMappingType = 0;
                SysPar.pdcch{2}.RegBundleSize =  6;
                SysPar.pdcch{2}.InterleaverSize =  0;
                SysPar.pdcch{2}.ShiftIndex =  0;
                SysPar.pdcch{2}.isCSS = 0;
                SysPar.pdcch{2}.coresetMap = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
                SysPar.pdcch{2}.numDlDci =  2;
                SysPar.pdcch{2}.idxUE = 1;
                SysPar.pdcch{2}.DCI{1}.RNTI = 3;
                SysPar.pdcch{2}.DCI{1}.ScramblingId = 7;
                SysPar.pdcch{2}.DCI{1}.ScramblingRNTI = 4;
                SysPar.pdcch{2}.DCI{1}.AggregationLevel = 4;
                SysPar.pdcch{2}.DCI{1}.PayloadSizeBits = 45;
                SysPar.pdcch{2}.DCI{1}.Payload = round(rand(1, SysPar.pdcch{2}.DCI{1}.PayloadSizeBits));
                SysPar.pdcch{2}.DCI{2} = SysPar.pdcch{2}.DCI{1};
                SysPar.pdcch{2}.DCI{2}.RNTI = 13;
                SysPar.pdcch{2}.DCI{2}.ScramblingId = 17;
                SysPar.pdcch{2}.DCI{2}.ScramblingRNTI = 14;
                SysPar.pdcch{2}.DCI{2}.AggregationLevel = 2;
                SysPar.pdcch{2}.DCI{2}.PayloadSizeBits = 37;
                SysPar.pdcch{2}.DCI{2}.Payload = round(rand(1, SysPar.pdcch{2}.DCI{2}.PayloadSizeBits));
                SysPar.testAlloc.pdcch = 2;
                SysPar.SimCtrl.N_UE = 2;
            elseif caseNum == 2034 % 4 BWPs
                SysPar.pdcch{2} = SysPar.pdcch{1};
                SysPar.pdcch{3} = SysPar.pdcch{1};
                SysPar.pdcch{4} = SysPar.pdcch{1};
                SysPar.pdcch{1}.BWPSize = 72;
                SysPar.pdcch{1}.BWPStart = 0;
                SysPar.pdcch{1}.DCI{1}.PayloadSizeBits = 45;
                SysPar.pdcch{1}.DCI{1}.Payload = round(rand(1, SysPar.pdcch{1}.DCI{1}.PayloadSizeBits));
                SysPar.pdcch{1}.idxUE = 0;
                SysPar.pdcch{2}.BWPSize = 60;
                SysPar.pdcch{2}.BWPStart = 72;
                SysPar.pdcch{2}.DCI{1}.PayloadSizeBits = 43;
                SysPar.pdcch{2}.DCI{1}.Payload = round(rand(1, SysPar.pdcch{2}.DCI{1}.PayloadSizeBits));
                SysPar.pdcch{2}.idxUE = 1;
                SysPar.pdcch{3}.BWPSize = 66;
                SysPar.pdcch{3}.BWPStart = 132;
                SysPar.pdcch{3}.DCI{1}.PayloadSizeBits = 41;
                SysPar.pdcch{3}.DCI{1}.Payload = round(rand(1, SysPar.pdcch{3}.DCI{1}.PayloadSizeBits));
                SysPar.pdcch{3}.idxUE = 2;
                SysPar.pdcch{4}.BWPSize = 75;
                SysPar.pdcch{4}.BWPStart = 198;
                SysPar.pdcch{4}.DCI{1}.PayloadSizeBits = 39;
                SysPar.pdcch{4}.DCI{1}.Payload = round(rand(1, SysPar.pdcch{4}.DCI{1}.PayloadSizeBits));
                SysPar.pdcch{4}.idxUE = 3;
                SysPar.testAlloc.pdcch = 4;
                SysPar.SimCtrl.N_UE = 4;
            elseif caseNum == 2035 % coreset0 (1 DCI) + coreset1 (16 DCIs) + coreset2 (16 DCIs)
                for idxPdcch = 2:3
                    SysPar.pdcch{idxPdcch} = SysPar.pdcch{1};
                    SysPar.pdcch{idxPdcch}.forceCceIndex = 1;
                    SysPar.pdcch{idxPdcch}.BWPSize = 273;
                    SysPar.pdcch{idxPdcch}.BWPStart = 0;
                    SysPar.pdcch{idxPdcch}.StartSymbolIndex = 0;
                    SysPar.pdcch{idxPdcch}.DurationSymbols = 2;
                    SysPar.pdcch{idxPdcch}.coresetIdx = idxPdcch-1;
                    SysPar.pdcch{idxPdcch}.CceRegMappingType = 0;
                    SysPar.pdcch{idxPdcch}.RegBundleSize =  6;
                    SysPar.pdcch{idxPdcch}.InterleaverSize =  0;
                    SysPar.pdcch{idxPdcch}.ShiftIndex =  0;
                    SysPar.pdcch{idxPdcch}.isCSS = 0;
                    if idxPdcch == 2
                        SysPar.pdcch{idxPdcch}.coresetMap = ones(1, 18);
                        SysPar.pdcch{idxPdcch}.dciUL = 0;
                    else
                        SysPar.pdcch{idxPdcch}.coresetMap = [zeros(1,26), ones(1, 18)];
                        SysPar.pdcch{idxPdcch}.dciUL = 1;
                    end
                    SysPar.pdcch{idxPdcch}.numDlDci =  16;
                    SysPar.pdcch{idxPdcch}.idxUE = idxPdcch-1;
                    for idxDci = 1:SysPar.pdcch{idxPdcch}.numDlDci
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.RNTI = idxDci + 1;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.ScramblingId = idxDci*2 + 1;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.ScramblingRNTI = idxDci*3 - 1;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.beta_PDCCH_1_0 = 8;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.powerControlOffsetSS = 1;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.powerControlOffsetSSProfileNR = 0;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.AggregationLevel = 2;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.cceIndex = 2*(idxDci - 1);
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.prcdBf = 0;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.seed = 0;
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.PayloadSizeBits = 35 + round(rand(1)*10);
                        SysPar.pdcch{idxPdcch}.DCI{idxDci}.Payload = round(rand(1, ...
                            SysPar.pdcch{idxPdcch}.DCI{idxDci}.PayloadSizeBits));
                    end
                end
                SysPar.testAlloc.pdcch = 3;
                SysPar.SimCtrl.N_UE = 3;
            elseif caseNum == 2036
                SysPar.pdcch{1}.testModel = 1;
                SysPar.pdcch{1}.coresetMap = [1];  
            elseif caseNum == 2105
                SysPar.pdcch{1}.coresetMap = ones(1, 32);    
            elseif caseNum == 2110
                SysPar.pdcch{1}.coresetMap = ones(1, 32);  
            elseif caseNum == 2111
                SysPar.pdcch{1}.coresetMap = ones(1, 32);    
            elseif caseNum == 2112
                SysPar.pdcch{1}.coresetMap = ones(1, 32);     
            elseif ismember(caseNum, [2201:2213, 2222])
                SysPar.pdcch{1}.coresetMap = 1; 
                SysPar.carrier.N_grid_size_mu = SysPar.pdcch{1}.BWPSize;
            elseif ismember(caseNum, [2223:2227])
                SysPar.pdcch{1}.coresetMap = ones(1, 4);
                SysPar.carrier.N_grid_size_mu = SysPar.pdcch{1}.BWPSize;
                if caseNum == 2225
                    SysPar.pdcch{1}.coresetMap = ones(1, 3);
                    SysPar.carrier.N_grid_size_mu = 24;
                end
            elseif ismember(caseNum, [2214:2221])
                SysPar.pdcch{1}.coresetMap = 1;
                SysPar.carrier.N_grid_size_mu = SysPar.pdcch{1}.BWPSize;
                SysPar.carrier.mu = 0;
            elseif caseNum == 2901           
                SysPar.pdcch{1}.coresetMap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
                       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
                SysPar.pdcch{1}.DCI{idxDCI}.Payload = [
                       0,0,0,0,1,0,1,1,1,1,1,...          % 95d == 48 PRBs
                       0,0,1,1,...                        % 3d == row4 == 7 symbols
                       0,...                              % non-interleaved
                       0,0,1,0,1,...                      % mcs = 5
                       0,0,...                            % tb scaling == 0
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];  % 16 bits reserved           
            elseif caseNum == 2902
                SysPar.pdcch{1}.coresetMap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
                       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
                SysPar.pdcch{1}.DCI{idxDCI}.Payload = ...
                     [ 1, 0, 0, 0, 0, 1, 0, 1,...
                       1, 1, 1, 1, 0, 0, 0, 0,...
                       0, 0, 0, 1, 0, 1, 0, 0,...
                       0, 0, 0, 0, 0, 0, 0, 0,...
                       1, 0, 0, 1, 0, 1, 0];                    
            elseif caseNum == 2903
                SysPar.pdcch{1}.coresetMap = [1, 1, 1, 1, 1, 1, 1, 1];
                SysPar.pdcch{1}.DCI{idxDCI}.Payload = ...
                    [0, 1, 1, 1, 0, 0, 0, 1,...
                     1, 1, 1, 0, 1, 0, 0, 0,...
                     0, 0, 1, 0, 1, 0, 0, 0,...
                     0, 0, 0, 0, 0, 0, 0, 0,...
                     0, 0, 0, 0, 0, 0, 0];
            elseif caseNum == 2904
                SysPar.pdcch{1}.coresetMap = [0,0,0,0,0,0,0,0,0,0,0,0,0,...
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0];
                SysPar.pdcch{1}.DCI{idxDCI}.Payload = ...
                    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,...
                    1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,...
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0];  
            elseif caseNum == 2905
                SysPar.pdcch{1}.coresetMap = [...
                      0, 0, 0, 0, 0, 0, 0, 0,...
                      0, 0, 0, 0, 0, 0, 0, 0,...
                      0, 0, 0, 1, 1, 1, 1, 1,...
                      1, 1, 1, 1, 1, 1, 1, 1,...
                      1, 1, 1];
                SysPar.pdcch{1}.DCI{idxDCI}.Payload = [...
                    0, 0, 0, 0, 0, 0, 1, 1,...
                    1, 1, 0, 0, 0, 0, 1, 1,...
                    0, 0, 0, 0, 0, 0, 1, 1,...
                    0, 1, 1, 1, 0, 0, 0, 0,...
                    0, 0, 0, 1, 0, 1, 1, 0,...
                    0, 0, 0, 0];
            elseif ismember(caseNum, [2801:2808,2810, 2811])
                SysPar.pdcch{1}.forceCceIndex = 1;
                SysPar.pdcch{1}.coresetMap = ones(1, 46);
                SysPar.pdcch{1}.DCI{idxDCI}.PayloadSizeBits =  CFG{idxSet, 20}; 
            elseif ismember(caseNum, [2809])
                SysPar.pdcch{1}.DCI{idxDCI}.AggregationLevel = 1;
                SysPar.pdcch{1}.DCI{idxDCI}.cceIndex = idxDCI-1;
                SysPar.pdcch{1}.forceCceIndex = 1;
                SysPar.pdcch{1}.coresetMap = ones(1, 6);
                SysPar.pdcch{1}.DCI{idxDCI}.PayloadSizeBits =  CFG{idxSet, 20}; 
                
                SysPar.carrier.N_grid_size_mu = 106; 
                SysPar.carrier.duplex = 0;
                
                SysPar.prach{1}.configurationIndex = 198;
                SysPar.testAlloc.prach = 0;
            elseif ismember(caseNum, [2906:2909])
                SysPar.pdcch{1}.coresetMap = ones(1, 24);
            end
            SysPar.pdcch{1}.DCI{idxDCI}.prcdBf = SysPar.pdcch{1}.DCI{1}.prcdBf;
            SysPar.pdcch{1}.DCI{idxDCI}.seed = SysPar.pdcch{1}.DCI{1}.seed;
            
            SysPar.pdcch{1}.DCI{idxDCI}.Payload = round(rand(1, SysPar.pdcch{1}.DCI{idxDCI}.PayloadSizeBits));            
        end
        
        for idxUe = 1:length(SysPar.pdcch)
            for idxDCI = 1:SysPar.pdcch{idxUe}.numDlDci
                digBFInterfaces = 1;
                switch SysPar.pdcch{idxUe}.DCI{idxDCI}.prcdBf
                    case 0
                        digBFInterfaces = 1;
                    case {1, 2, 5, 6}
                        digBFInterfaces = 2;
                    case {3, 4, 7, 8, 9, 10, 11, 12}
                        digBFInterfaces = 4;
                    otherwise
                        error('prcdBf is not supported ... \n');
                end
                SysPar.pdcch{idxUe}.DCI{idxDCI}.digBFInterfaces = digBFInterfaces;
                SysPar.pdcch{idxUe}.DCI{idxDCI}.beamIdx = [1:digBFInterfaces];
            end
        end
        
        % update Nre_max which is used to calculate beta value in BFP
        SysPar.SimCtrl.oranComp.Nre_max = SysPar.carrier.N_grid_size_mu*12;
        
        % save SysPar into Cfg_<TC#>.yaml config file
        if SysPar.SimCtrl.genTV.genYamlCfg
            fileName = sprintf('Cfg_%04d.yaml', caseNum);
            WriteYaml(fileName, SysPar);
        end        
        
        if SysPar.SimCtrl.genTV.enable && SysPar.SimCtrl.genTV.launchPattern
            if ~ismember(caseNum, disabled_TC)
                LPFileName = 'launch_pattern_nrSim';
            else
                LPFileName = 'disabled_launch_pattern_nrSim';
            end
            slotIdx =  SysPar.SimCtrl.genTV.slotIdx;
            genSingleSlotLPFile(LPFileName, caseNum, slotIdx);
        end

        if ismember(caseNum, [2031]) % precoding TCs
            SysPar.SimCtrl.alg.enablePrcdBf = 1;
        end
        
        bypassDet = 0;
        if strcmp(caseSet, 'full') || strcmp(caseSet, 'compact')
            SysPar.SimCtrl.enableUeRx = 1;
            if ismember(caseNum, [2036])
                bypassDet = 1;
            end
        end
        [SysPar, UE, gNB] = nrSimulator(SysPar);
        
        Detected = 1;
        if SysPar.SimCtrl.enableUeRx
            results = SysPar.SimCtrl.results.pdcch;
            nPdcch = length(results);
            for idxPdcch = 1:nPdcch
                if (results{idxPdcch}.errCnt > 0)
                    Detected = 0;
                end
            end
            
            if bypassDet
                Detected = 1;
            end
            
            if ~Detected
                detErr = detErr + 1;
            end
        end        
        
        testPass = 1;        
        if ismember(caseNum, [2031, 2035, 2036, 2801:2811]) % precoding, 16-UE, perf TCs
            bypassCompTest = 1;
        else
            bypassCompTest = 0;
        end        
        if testCompliance && ~bypassCompTest
            nComp = nComp + 1;
            
            [dcicw, sym, cgrid] = hPDCCHGen(SysPar.pdcch, gNB.Phy.Config.carrier);
            
            Xtf_5G = cgrid;
            Xtf_nr = gNB.Phy.tx.Xtf(:,:,1);
            err_Xtf = sum(sum(abs(Xtf_5G - Xtf_nr)));
            
            testPass = (err_Xtf < 1e-4);
            if ~testPass
                errCnt = errCnt + 1;
            end            
        end
        fprintf('%4d %4d   %4d %4d %4d %4d %4d   %4d %4d  %4d  %4d  %4d  %4d  %5d  %4d   %4d    %4s   %4d   %4d  %4d  %4d %4d\n',...
            CFG{idxSet, 1}, CFG{idxSet, 2}, CFG{idxSet, 3}, CFG{idxSet, 4}, ...
            CFG{idxSet, 5}, CFG{idxSet, 6}, CFG{idxSet, 7}, CFG{idxSet, 8}, ...
            CFG{idxSet, 9}, CFG{idxSet, 10}, CFG{idxSet, 11}, CFG{idxSet, 12}, ...
            CFG{idxSet, 13}, CFG{idxSet, 14}, CFG{idxSet, 15}, CFG{idxSet, 16}, ...
            aggrL_display, CFG{idxSet, 18}, CFG{idxSet, 19}, CFG{idxSet, 20}, ...
            testPass, Detected);        
    end
end
fprintf('------------------------------------------------------------------------------------------------------------------------------------\n');
fprintf('Total Compliance TC = %d, PASS = %d, FAIL = %d, Total TV generated = %d\n\n', nComp, nComp-errCnt, errCnt, nTV);
toc; 
fprintf('\n');