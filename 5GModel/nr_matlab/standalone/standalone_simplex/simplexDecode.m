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

function [out, confLevel] = simplexDecode(inLLRs, K, E, Qm)
%
% This function implements Simplex decoding with de-rate-matching given an array of descrambled LLRs,
%
% Input:    inLLRs:  the array of descrambled LLRs
%           K:       the number of information bits
%           E:       the rate-matched LLR sequence length
%           Qm:      modulation order, should be in {1, 2, 4, 6, 8}
%           
%
% Output:   out: the decoded information bit(s), 
%
%

%% validate inputs
if length(inLLRs) ~= E
    error('Invalid input: the length of input LLR sequence does not match with the provided sequence length E');
end

if Qm ~= 1 && Qm ~= 2 && Qm ~= 4 && Qm ~= 6 && Qm ~= 8
    error('Invalid input: Modulation order Qm must be in {1, 2, 4, 6, 8}');
end

if K ~= 1 && K ~= 2
    error('Invalid input: Simplex code only applies to 1-bit or 2-bit information');
end

if K == 1
    if E < Qm
        error('Invalid input: For 1-bit information, the length of rate-matched sequence E must be no less than Qm');
    end
end

%% decode
out = zeros(1, K);
totalPwr = 0;
counter = 0;

if K == 1
    if Qm == 1 || Qm == 2
        sumLLR = 0;
        for bitIdx = 1:E
            sumLLR = sumLLR + inLLRs(bitIdx);
            totalPwr = totalPwr + inLLRs(bitIdx)^2;
            counter = counter + 1;
        end
        if sumLLR < 0
            out(1) = 1;
        end
    else
        sumLLR = 0;
        for bitIdx = 1:E
            temp = mod(bitIdx, Qm);
            if temp == 1 || temp ==2
                sumLLR = sumLLR + inLLRs(bitIdx);
                totalPwr = totalPwr + inLLRs(bitIdx)^2;
                counter = counter + 1;
            end
        end
        if sumLLR < 0
            out(1) = 1;
        end
    end
    
    confLevel = (abs(sumLLR)/counter)/sqrt(totalPwr/counter);

else % K == 2
    
    c0Sum = 0; % sum of LLRs for c0
    c1Sum = 0; % sum of LLRs for c1
    c2Sum = 0; % sum of LLRs for c2
    if Qm == 1
        for bitIdx = 1:E
            temp = mod(bitIdx, 3);
            if temp == 1
                c0Sum = c0Sum + inLLRs(bitIdx);
                totalPwr = totalPwr + inLLRs(bitIdx)^2;
                counter = counter + 1;
            elseif temp == 2
                c1Sum = c1Sum + inLLRs(bitIdx);
                totalPwr = totalPwr + inLLRs(bitIdx)^2;
                counter = counter + 1;
            else % temp == 0
                c2Sum = c2Sum + inLLRs(bitIdx);
                totalPwr = totalPwr + inLLRs(bitIdx)^2;
                counter = counter + 1;
            end
        end
    else
        for rmQamIdx = 0 : (E / Qm - 1)
            cwQamIdx = mod(rmQamIdx, 3);
            
            switch cwQamIdx
                case 0
                    c0Sum = c0Sum + inLLRs(rmQamIdx*Qm + 1);
                    c1Sum = c1Sum + inLLRs(rmQamIdx*Qm + 2);
                case 1
                    c2Sum = c2Sum + inLLRs(rmQamIdx*Qm + 1);
                    c0Sum = c0Sum + inLLRs(rmQamIdx*Qm + 2);
                case 2
                    c1Sum = c1Sum + inLLRs(rmQamIdx*Qm + 1);
                    c2Sum = c2Sum + inLLRs(rmQamIdx*Qm + 2);
            end
            totalPwr = totalPwr + abs(inLLRs(rmQamIdx*Qm + 1))^2 + abs(inLLRs(rmQamIdx*Qm + 2))^2;
            counter  = counter + 2;
        end
    end
    
    %% matrix of all possible codeword [c0 c1 c2; c0 c1 c2; c0 c1 c2; c0 c1 c2], c2 = mod(c0 + c1, 2)
    cw = [0 0 0; 0 1 1; 1 0 1; 1 1 0];
    
    %% array of costs
    costs = zeros(4, 1);
    
    for cwIdx = 1:4
        costs(cwIdx) = cw(cwIdx, 1)*c0Sum + cw(cwIdx, 2)*c1Sum + cw(cwIdx, 3)*c2Sum;
    end
    
    %% find the cw that minimizes the cost function
    [minCost, minCwIdx] = min(costs);
    
    minCost = minCost - max(costs);
    
    confLevel = (-minCost/counter)/sqrt(totalPwr/counter)*3/2;
    
    out = cw(minCwIdx, 1:2);
end

end
