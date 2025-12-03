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

function [symIdxInFrame] = findSsbSymIdx(fc, duplex, caseType, ssb)

if ssb.cfgTx
    symIdxInFrame = ssb.symIdxInFrame;
else        
    switch caseType
        case 'case_A'
            symIdx = [2, 8];
            base = 14;
            if fc <= 3
                nVec = [0, 1];
            elseif fc <= 6
                nVec = [0, 1, 2, 3];
            end
        case 'case_B'
            symIdx = [4, 8, 16, 20];
            base = 28;
            if fc <= 3
                nVec = [0];
            elseif fc <= 6
                nVec = [0, 1];
            end
        case 'case_C'
            symIdx = [2, 8];
            base = 14;
            if duplex == 0 % FDD
                if fc <= 3
                    nVec = [0, 1];
                elseif fc <= 6
                    nVec = [0, 1, 2, 3];
                end
            else
                if fc <= 2.4
                    nVec = [0, 1];
                elseif fc <= 6
                    nVec = [0, 1, 2, 3];
                end
            end
        case 'case_D'
            % TBD
        case 'case_E'
            % TBD
        otherwise
            fprintf('error: SSB case type is not supported\n');
    end
    symIdxInFrame = [];
    for idx = 1:length(nVec)
        symIdxInFrame = [symIdxInFrame, symIdx + nVec(idx)*base];
    end        
end

      


      

        
