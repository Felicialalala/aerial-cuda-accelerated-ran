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

function outputResult(SimCtrl)
% function outputResult(SimCtrl, gNB)
%
% This function output performance analysis result at the end of run 
%

results = SimCtrl.results;

if isfield(results, 'pusch')
    pusch = results.pusch;
    nPusch = length(pusch);
    for idxUE = 1:nPusch
        fprintf("CH = PUSCH, UE# = %1d, TBcnt = %2d, TBerr = %2d, TBER = %4.2e\n", ...
            pusch{idxUE}.idxUE, pusch{idxUE}.totalCnt, ...
            pusch{idxUE}.errorCnt, pusch{idxUE}.errorCnt/pusch{idxUE}.totalCnt);
    end
end

if isfield(results, 'pucch')
    pucch = results.pucch;
    nPucch = length(pucch);
    for idxUE = 1:nPucch
        fprintf("CH = PUCCH, UE# = %1d, TBcnt = %2d, TBerr = %2d, TBER = %4.2e\n", ...
            pucch{idxUE}.idxUE, pucch{idxUE}.totalCnt, ...
            pucch{idxUE}.errorCnt, pucch{idxUE}.errorCnt/pucch{idxUE}.totalCnt);
    end
end

if isfield(results, 'srs')
    srs = results.srs;
    nSrs = length(srs);
    for idxUE = 1:nSrs
        fprintf("CH = SRS,   UE# = %1d\n", srs{idxUE}.idxUE);
    end
end

if isfield(results, 'prach')
    prach = results.prach;
    nPrach = length(prach);
    for idxUE = 1:nPrach
        fprintf("CH = PRACH, UE# = %1d, TBcnt = %2d, TBerr = %2d, TBER = %4.2e\n", ...
            prach{idxUE}.idxUE, prach{idxUE}.totalCnt, ...
            prach{idxUE}.errorCnt, prach{idxUE}.errorCnt/prach{idxUE}.totalCnt);
    end
end

% 
% N_alloc = length(alloc);
% for idxAlloc = 1:N_alloc        % for each allocation
%     thisAlloc = alloc{idxAlloc};
%     allocType = thisAlloc.type;
%     switch allocType            % process based on allocation type
%         case {'prach'}
%             prach = thisAlloc;
%             if SimCtrl.prach.printSummary
%                 fprintf('---------------------------------');
%                 fprintf('-----------------------------------');
%                 fprintf('\n%6s %12s %10s %10s %10s %16s', 'N_tx', 'N_detected',...
%                     'N_miss', 'N_false', 'N_wrong', 'N_timingErr');
%                 fprintf('\n%6d %12d %10d %10d %10d %16d\n\n', ...
%                     prach.txRaCounter, prach.detectCounter,...
%                     prach.missCounter, prach.falseCounter, ...
%                     prach.wrongCounter, prach.timingErrCounter);
%             end
%         case 'ssb'
%             % TBD
%         case 'pdcch'
%             % TBD
%         case 'pucch'
%             % TBD
%         otherwise
%             error('alloc type is not supported ... \n');
%     end
%     alloc{idxAlloc} = thisAlloc;
% end
% SimCtrl.gNB.alloc = alloc;
