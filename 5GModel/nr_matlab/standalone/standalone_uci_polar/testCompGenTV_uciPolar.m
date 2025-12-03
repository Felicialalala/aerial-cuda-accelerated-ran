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

function  testCompGenTV_uciPolar(caseSet, compTvMode)

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
    case 'genTV'
        genTvFlag = 1;
        testCompliance = 0;
    case 'testCompliance'
        genTvFlag = 0;
        testCompliance = 1;
    otherwise
        error('compTvMode is not supported...\n');
end

compact_TC  = [60001:70000];
full_TC     = [60001:70000];
selected_TC = [60001:70000];
% selected_TC = 60004;

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
% single polar uci segment tests
%   TC#      nUciSegs A_seg     E_seg  SNR
    60001    1        12        22     30
    60002    1        19        29     30
    60003    1        200       212    30
    60004    1        900       930    30
    60005    1        1012      1024   30
    60006    1        1015      1200   30
    60007    1        1014      1200   30
    60008    1        200       830    30
    60009    1        1706      2000   30
    60010    1        1706      8192   30
    60011    1        1013      1043   30
    60012    1        1013      16385  30
    60013    1        20        32     30
    
% Parity bits, wmFlag = 0. 11 <= A <= 19. (A + 9) <= E <= (195 + A).
60014  1    12    22     30
60015  1    12    57     30
60016  1    12    112    30
60017  1    12    156    30
60018  1    12    207    30
60019  1    19    29     30
60020  1    19    57     30
60021  1    19    112    30
60022  1    19    156    30
60023  1    19    214    30

% Parity bits, wmFlag = 1. 11 <= A <= 19. (195 + A) < E.
60024  1    12    208    30
60025  1    12    402    30
60026  1    12    4002   30
60027  1    19    215    30
60028  1    19    508    30
60029  1    19    5008   30

% No segmentation with (E < 1088) and (A < 1013). No parity. Have: E >= (A + 11)
60030  1    20     32    30
60031  1    20     301   30
60032  1    20     601   30
60033  1    20     1087  30
60034  1    1012   1024  30
60035  1    1012   1087  30
60036  1    200    212   30
60037  1    200    511   30
60038  1    200    1087  30

% No segmentation with (E >= 1088) and (A < 360). No parity. Have: (E >= A + 11)
60039  1    20    1088  30
60040  1    200   1088  30
60041  1    359   1088  30

% Segmentation with (E >= 1088) and (A >= 360). No zero-insertion. Have: (E >= A + 11)
60042   1    360     1088  30
60043   1    560     1088  30
60044   1    1076    1100  30
60045   1    360     2088  30
60046   1    360     3088  30

% Segmentation with (E >= 1088) and (A >= 360). Yes zero-insertion. Have: (E >= A + 11)
60047   1    361     1088  30
60048   1    561     1088  30
60049   1    1077    1102  30
60050   1    361     2088  30
60051   1    361     3088  30

% test crc errors
60052  1    12    208   -20  % crc 6  fail test. Small payload.
60053  1    19    508   -20  % crc 6  fail test. Large payload.
60054  1    20    601   -20  % crc 11 fail test. No seg. Small payload.
60055  1    1012  1087  -20; % crc 11 fail test. No seg. Large payload.
60056  1    360   1088  -20  % crc 11 fail test. Yes seg. Small payload.
60057  1    1077  1102  -20  % crc 11 fail test. Yes seg. Large payload.

% Segmentation with (E >= 1088) and (A >= 360). 15 words, and different remainders
60058   1    480     1088  30
60059   1    481     1088  30
60060   1    482     1088  30
60061   1    483     1088  30
60062   1    479     1088  30
60063   1    478     1088  30

% Segmentation with (E >= 1088) and (A >= 360). 16 words, and different remainders
60064   1    512     1088  30
60065   1    513     1088  30
60066   1    514     1088  30
60067   1    515     1088  30
60068   1    516     1088  30
60069   1    517     1088  30
60070   1    518     1088  30
60071   1    519     1088  30
60072   1    520     1088  30
60073   1    521     1088  30
60074   1    522     1088  30
60075   1    523     1088  30
60076   1    524     1088  30
60077   1    525     1088  30
60078   1    526     1088  30
60079   1    527     1088  30
60080   1    528     1088  30
60081   1    529     1088  30
60082   1    530     1088  30
60083   1    531     1088  30
60084   1    532     1088  30
60085   1    533     1088  30
60086   1    534     1088  30
60087   1    535     1088  30
60088   1    536     1088  30
60089   1    537     1088  30
60090   1    538     1088  30
60091   1    539     1088  30
60092   1    540     1088  30
60093   1    541     1088  30
60094   1    542     1088  30
60095   1    543     1088  30
60096   1    544     1088  30
60097   1    545     1088  30
60098   1    546     1088  30
60099   1    547     1088  30
60100   1    548     1088  30
60101   1    549     1088  30

% multiple codeblocks for GPU capacity test
60102  100    repmat(20,1,100)     repmat(32,1,100)    30
60103  100    repmat(20,1,100)     repmat(301,1,100)   30
60104  100    repmat(20,1,100)     repmat(601,1,100)   30
60105  100    repmat(20,1,100)     repmat(1087,1,100)  30
60106  100    repmat(1012,1,100)   repmat(1024,1,100)  30
60107  100    repmat(1012,1,100)   repmat(1087,1,100)  30
60108  100    repmat(200,1,100)    repmat(212,1,100)   30
60109  100    repmat(200,1,100)    repmat(511,1,100)   30
60110  100    repmat(200,1,100)    repmat(1087,1,100)  30

% 3ggp illegal configurations:
60121  1  25  25  30
60122  1  18  25  30
60123  1  25  20  30


};


[NallTest, ~] = size(CFG);
nCompErrs             = 0;
nCompChecks           = 0;
nDecoderErrs          = 0;
nDecoderChecks        = 0;
nExpectedDecodeErrors = 0;
nTvGen                = 0;


for i = 1:NallTest
    caseNum = CFG{i, 1};
    
    if ismember(caseNum, TcToTest)
        rng(caseNum);
        nTvGen = nTvGen + genTvFlag;
        
        nPolUciSegs = CFG{i, 2};
        A_seg       = CFG{i, 3};
        E_seg       = CFG{i, 4};
        inputSnr    = CFG{i, 5};
        
        if(inputSnr < -10)
            nExpectedDecodeErrors = nExpectedDecodeErrors + 1;
        end
            
        
        % Transmit chain:
        [uciSegPayloads_cell, polCbs_cell, polCbsCrcEncoded_cell, polCws_cell, polCwsRmItl_cell, polUciSegsEncoded_cell] = uci_polar_tx(nPolUciSegs, A_seg, E_seg);
        
        % Channel:
        polUciSegLLRs_cell = apply_uciPolar_channel(inputSnr, nPolUciSegs, polUciSegsEncoded_cell);
        
        % Recieve chain:
        listLength = 8;
        [cuphyCwBitTypes_cell, cwTreeTypes_cell, cwLLRs_cell, uciSegEsts_cell] = uci_polar_rx(nPolUciSegs, A_seg, E_seg, listLength, polUciSegLLRs_cell, genTvFlag, caseNum);

        % Complience check: 
        caseNum
        if(testCompliance)
            nCompChecks = nCompChecks + 1;
            mathworksEncodedUci = nrUCIEncode(uciSegPayloads_cell{1},E_seg);
            modelEncodedUci     = polUciSegsEncoded_cell{1};

            if(sum(abs(mathworksEncodedUci - modelEncodedUci)) > 0)
                errorFlag = 1;
            else
                errorFlag = 0;
            end
            nCompErrs = nCompErrs + errorFlag;
            
             if(errorFlag)
                 fprintf('\n complience error with test-case: %d \n', caseNum);
             end
        end
        
        % check decoder output
        nDecoderChecks  = nDecoderChecks + 1;
        [errorFlag,snr] = compare_cells(uciSegEsts_cell, uciSegPayloads_cell);
        if(errorFlag)
            fprintf('\n decoder error with test-case: %d \n', caseNum);
            nDecoderErrs = nDecoderErrs + 1;
        end      
    end
end

fprintf('\n Generated %d test-vectors', nTvGen);
fprintf('\n Performed %d complience tests, finding %d errors', nCompChecks, nCompErrs);
fprintf('\n Performed %d decoder tests   , finding %d errors (expected %d errors) \n\n', nDecoderChecks, nDecoderErrs, nExpectedDecodeErrors);

end


function [compErrorFlag,snr] = compare_cells(cell_1, cell_2)
    n = length(cell_1);
    snr = zeros(n,1);
    
    compErrorFlag = 0;
    for i = 1 : n
        A = cell_1{i};
        A(A >= 20) = 20;
        
        B = cell_2{i};
        B(B >= 20) = 20;
        
        E = abs(A - B).^2;
        S = abs(B).^2;
        
        snr(i) = 10*log10(mean(S) / mean(E));
        
        if(snr(i) < 100)
            compErrorFlag = 1;
            break;
        end
    end
end
        
           
            
            
        
        
        





