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

function detErr = checkDetError(testAlloc, SimCtrl, printFlag)

if nargin < 3
    printFlag = 1;
end

negativeTC_flag = SimCtrl.negTV.enable > 0;
detErr = 0;

if testAlloc.dl && SimCtrl.enableUeRx
    if testAlloc.ssb
        ssbResults = SimCtrl.results.ssb;
        nResults = length(ssbResults);
        for idx = 1:nResults
            if ~isempty(ssbResults{idx})
                if ssbResults{idx}.errCnt && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('UE # %d: SSB MIB error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('UE # %d: %d SSB MIB detected \n', idx-1, ssbResults{idx}.totalCnt);
                    end
                end
            end
        end
    end
    if testAlloc.pdcch
        pdcchResults = SimCtrl.results.pdcch;
        nResults = length(pdcchResults);
        for idx = 1:nResults
            if ~isempty(pdcchResults{idx})
                if pdcchResults{idx}.errCnt && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('UE # %d: PDCCH DCI error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('UE # %d: %d PDCCH DCI detected \n', idx-1, pdcchResults{idx}.totalCnt);
                    end
                end
            end
        end
    end
    if testAlloc.pdsch
        pdschResults = SimCtrl.results.pdsch;
        nResults = length(pdschResults);
        for idx = 1:nResults
            if ~isempty(pdschResults{idx})
                if pdschResults{idx}.tbErrorCnt && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('UE # %d: PDSCH TB error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('UE # %d: %d PDSCH TB detected \n', idx-1, pdschResults{idx}.tbCnt);
                    end
                end
            end
        end
    end
end
    
if testAlloc.ul
    if testAlloc.pusch
        puschResults = SimCtrl.results.pusch;
        nResults = length(puschResults);
        for idx = 1:nResults
            if ~isempty(puschResults{idx})
                if puschResults{idx}.tbErrorCnt && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('UE # %d: PUSCH TB error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('UE # %d: PUSCH TB detected \n', idx-1);
                    end
                end
            end
        end
    end
    
    if testAlloc.prach
        prachResults = SimCtrl.results.prach;
        nResults = length(prachResults);
        for idx = 1:nResults
            if ~isempty(prachResults{idx})
                if (prachResults{idx}.falseCnt || prachResults{idx}.missCnt) && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('RO # %d: PRACH error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('RO # %d: %d Prmb detected \n', idx-1, prachResults{idx}.prmbCnt );
                    end
                end
            end
        end
    end
    
    if testAlloc.pucch
        pucchResults = SimCtrl.results.pucch;
        nResults = length(pucchResults);
        for idx = 1:nResults
            if ~isempty(pucchResults{idx})
                if pucchResults{idx}.errorCnt && ~negativeTC_flag
                    detErr = detErr + 1;
                    fprintf('UE # %d: PUCCH UCI error !!! \n', idx-1);
                else
                    if printFlag
                        fprintf('UE # %d: PUCCH UCI detected \n', idx-1);
                    end
                end
            end
        end
    end
end
