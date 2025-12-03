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

function [payloadEst, crcErrorFlag] = polar_decoder(L, K, N, nCrcBits, LLR_cw,cwTreeTypes)

%function applies CA-SCL decoder (CRC aided successive canclenation list
%decoder). Linked list storage of stage variables.

%inputs:
% L           --> number of paths
% K           --> number of payload + CRC bits
% N           --> codeblock length
% nCrcBits    --> number of crc bits
% LLR_cb      --> LLRs of codeword bits. Dim: N x 1 
% cwTreeTypes --> Indicates cw type in cwTree. 
                    % 0 --> rate 0 cw
                    % 1 --> rate 1 cw
                    % 3 --> neither

%outputs:
% payloadEst     --> CA-SCL estimate of payload bits. Dim: (K - nCrcBits) x 1
% crcErrorFlag   --> 0 if CRC passes. 1 if CRC fails.

%%
%SETUP

% number of stages:
n = log2(N) + 1;

% load depth:
load('depth.mat');


%%
% INTITALIZE

% initalize cwTrees:
cwTree         = cwTree_class(N,L);
cwTree.cwTypes = cwTreeTypes;

% intialize LLRs of final stage codeword:
cwTree = set_cwLLR(cwTree,0,n-1,LLR_cw);

% intialize location in cwTree (0'th subCodeTree of last stage):
s = n - 1; subTreeIdx = 0; 

% propogate input LLRs to first leaf codeword:
while (s > -1) 
    
    s = s - 1;
    cwTree = F(cwTree,0,s);
    
    if (getType(cwTree,subTreeIdx,s) ~= 3)
        break;
    end
end


% initialize path metrics:
pm = zeros(L,1);
nPaths = 1;

% intialize linked list pointers:
P = zeros(L,n-1);

% initalize location of last decoded bit:
bitIdx = -1;


%%
%MAIN LOOP


while bitIdx < (N-1)
      
    type = getType(cwTree,subTreeIdx,s);    
    if type == 0
        % decode rate zero codeword:
        [c0, P_prime, pm] = R0_decoder(nPaths, cwTree, L, pm, s);
    
    elseif(((type == 1) || (type == 2)) && (s == 0))
        % decode leaf node:
        [c0, P_prime, pm, nPaths] = R1_S0_decoder(nPaths, cwTree, L, pm);
        
    elseif((type == 1) && (s ~= 0))
        % decode rate one codeword:
        [c0, P_prime, pm, nPaths] = R1_decoder(nPaths, cwTree, L, pm, s);
        
    end
    
    % update bit index:
    bitIdx = bitIdx + 2^s;
    if bitIdx == (N-1)
        break;
    end
   
    % use "H" to combine codeword estimates until stage "d":
    d              = depth(bitIdx+1);    
    cs             = zeros(2^d,nPaths);
    cs(1 : 2^s, :) = c0;
    
    while (s < d)
        for p = 0 : (nPaths-1)
            cs(1:2^(s+1),p+1) = H(cwTree, s+1, cs(1:2^s,p+1), P_prime(p+1));
        end
        
        s = s + 1;
        subTreeIdx = floor(subTreeIdx / 2);
        
        P_prime = P(P_prime + 1, s);
    end
   
    % use "G" and new stage "d" codeword estimate to update LLRs of the "lower" stage "d" codeword:
    subTreeIdx = subTreeIdx + 1;
    for p = 0 : (nPaths-1)
        cwTree = G(cwTree,p,s,cs(:,p+1),P_prime(p+1));
    end
        
    % use "F" to propogate updated LLRs to next leaf codeword
    while (s > -1)
        
        if(getType(cwTree,subTreeIdx,s) ~= 3)
            break;
        end
        
        s = s - 1;
        subTreeIdx = 2*subTreeIdx;
        for p = 0 : (nPaths-1)
            cwTree = F(cwTree,p,s);
        end
    end
    
    % store stage "d" codeword estimate:
    for p = 0 : (nPaths-1)
        cwTree = set_cwEst(cwTree,p,d,cs(:,p+1));
    end
    
    % store pointer:
    P(1:nPaths,d+1) = P_prime;
end
  

%%
%FINALIZE

% use XOR butterfly structure to convert stage estiamtes to data estimates:
bitEsts = zeros(N,L);

idx = (N - 2^s) : (N-1);
for p = 0 : (L-1)
    bitEsts(idx + 1, p + 1) = xor_butterfly(c0(:,p+1),s);
end

t=s;
for s = t : (n - 2)
    idx = (N - 2^(s+1)) : (N - 2^(s+1) + 2^s - 1);
    
    for p = 0 : (L-1)
        cwEst                   = get_cwEst(cwTree,P_prime(p + 1),s); 
        bitEsts(idx + 1, p + 1) = xor_butterfly(cwEst,s);
    end
    
    P_prime = P(P_prime + 1, s + 1);
end

% extract information bits:
dataEst      = zeros(K,L);

count = 0;
for bitIdx = 0 : (N - 1)
    if (getType(cwTree,bitIdx,0) == 1)
       dataEst(count + 1, :) =  bitEsts(bitIdx + 1, :);
       count = count + 1;
    end
end
        
% perform CRC checks:
CRC_error = zeros(L,1);
for p = 0 : (L-1)
    [~,CRC_error(p+1)] = CRC_decode_polar(dataEst(:,p+1),num2str(nCrcBits));
end

% Penalize CRC failures:
pm(CRC_error == 1) = Inf;

% return candidate with smallest path metric:
[~,idx] = min(pm);
payloadEst     = dataEst(1 : end - nCrcBits,idx);
crcErrorFlag   = CRC_error(idx);



    
    
    
    
    
    
    
    
    




    