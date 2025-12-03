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

function [Xtf_remap, Xtf_remap_trsnzp] = gen_csirs_remap(csirs_list, Xtf_remap, Xtf_remap_trsnzp, csirsTable)

nCfg = length(csirs_list);

for idxCfg = 1:nCfg
    
    Row = csirs_list{idxCfg}.Row;
    CSIType = csirs_list{idxCfg}.CSIType;
    CDMType = csirs_list{idxCfg}.CDMType;
    FreqDensity = csirs_list{idxCfg}.FreqDensity;
    StartRB = csirs_list{idxCfg}.StartRB;
    NrOfRBs = csirs_list{idxCfg}.NrOfRBs;
    SymbL0 = csirs_list{idxCfg}.SymbL0;
    SymbL1 = csirs_list{idxCfg}.SymbL1;
    FreqDomain_bin = dec2bin(csirs_list{idxCfg}.FreqDomain, 12) - '0';
    
    SymbL = [SymbL0, SymbL1];
    
    % read table 7.4.1.5.3-1
    X = csirsTable.csirsLocTable.Ports{Row};
    KBarLBar = csirsTable.csirsLocTable.KBarLBar{Row};
    CDMGroupIndices = csirsTable.csirsLocTable.CDMGroupIndices{Row};
    KPrime = csirsTable.csirsLocTable.KPrime{Row};
    LPrime = csirsTable.csirsLocTable.LPrime{Row};
    
    switch FreqDensity
        case 0
            rho = 0.5;
            genEvenRB = 1;
        case 1
            rho = 0.5;
            genEvenRB = 0;
        case 2
            rho = 1;
        case 3
            rho = 3;
    end
    
    if X == 1
        alpha = rho;
    else
        alpha = 2*rho;
    end
    
    % FreqDomain_bin = dec2bin(FreqDomain, 12) - '0';
    FreqDomain_flip = fliplr(FreqDomain_bin);
    switch Row
        case 1
            idxOne = find(FreqDomain_flip(1:4));
            ki =idxOne - 1;
        case 2
            idxOne = find(FreqDomain_flip(1:12));
            ki = idxOne - 1;
        case 4
            idxOne = find(FreqDomain_flip(1:3));
            ki = 4*(idxOne - 1);
        otherwise
            idxOne = find(FreqDomain_flip(1:6));
            ki = 2*(idxOne - 1);
    end
    
    lenKBarLBar = length(KBarLBar);
    lenLPrime = length(LPrime);
    lenKPrime = length(KPrime);
    
    hasTwoSyms = ismember(Row, [13 14 16 17]);
    
    for idxRB = StartRB:StartRB + NrOfRBs - 1
        isEvenRB = (mod(idxRB, 2) == 0);
        if (rho == 0.5)
            if (genEvenRB && ~isEvenRB) || (~ genEvenRB && isEvenRB)
                continue;
            end
        end
        
        for idxKBarLBar = 1:lenKBarLBar
            kl_BarPair = KBarLBar{idxKBarLBar};
            if Row == 1 || Row == 4
                k_bar = ki(1) + kl_BarPair(1);
            else
                k_bar = ki(kl_BarPair(1)+1);
            end
            if hasTwoSyms && idxKBarLBar > lenKBarLBar/2
                l_bar = SymbL(2) + kl_BarPair(2);
            else
                l_bar = SymbL(1) + kl_BarPair(2);
            end
            for idxLPrime = 1:lenLPrime
                for idxKPrime = 1:lenKPrime
                    k_prime = KPrime(idxKPrime);
                    k = k_bar + k_prime + idxRB*12;
                    l_prime = LPrime(idxLPrime);
                    ll = l_bar + l_prime;
                    m_prime = floor(idxRB*alpha) + k_prime + ...
                        floor(k_bar*rho/12);
                    Xtf_remap(k+1, ll+1) = 1;
                    if CSIType < 2
                        Xtf_remap_trsnzp(k+1, ll+1) = 1;
                    end
                end
            end
        end
    end
end

return
