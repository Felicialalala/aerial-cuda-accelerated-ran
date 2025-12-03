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

% Class holds estimates and LLRs for all codewords in the tree
% Class has "F", "G", and "H" functions to update codeword LLRs and estiamtes

classdef cwTree_class
    
   properties 
      N;              % codeblock size
      n;              % number of stages
      L;              % number of decoding lists
      Est;            % codeblock estimates, one per stage (except final). Dim: L x (N-1)
      LLR;            % codeblock LLRs, one per stage. Dim: L x (2N-1)
      cwTypes;        % code word types. 0: rate-zero, 1: rate-one, -1: neither. Dim: (2N-2) x 1
   end 
   
   methods
       
       %%
       %CONSTRUCTOR
       % intalizes cwTree object
       
      function cwTree = cwTree_class(N,L)
          
         % extract constants:
         cwTree.N = N;           % codeblock size
         cwTree.n = log2(N) + 1; % number of stages, n = log2(N) + 1.
         cwTree.L = L;           % number of decoding lists
         
         % initalize memory:
         cwTree.Est = zeros(L,N - 1);   % init codeword estimates to zero
         cwTree.LLR = zeros(L,2*N - 1); % init codeword LLRs to zero
         
         % initalize codeword types:
         cwTree.cwTypes = -ones(2*N-2,1);
         
      end
      
      %%
      %SET/GET FUNCTIONS
        
          
      % function sets LLRs for the j'th stage "s" codeword:
      function cwTree = set_cwLLR(cwTree,j,s,cwLLR)          
          cwIdx = get_cwIdx(cwTree,s);
          cwTree.LLR(j+1,cwIdx+1) = cwLLR.';   
      end 
      
      % function sets estimate for the j'th stage "s" codeword:
      function cwTree = set_cwEst(cwTree,j,s,cwEst)
          cwIdx = get_cwIdx(cwTree,s);
          cwTree.Est(j+1,cwIdx+1) = cwEst.';   
      end
      
      % function gets LLRs for the j'th stage "s" codeword:
      function cwLLR = get_cwLLR(cwTree,j,s)
          cwIdx = get_cwIdx(cwTree,s);
          cwLLR = cwTree.LLR(j+1,cwIdx+1).';   
      end 
      
      % function gets estimate for the j'th stage "s" codeword:
      function cwEst = get_cwEst(cwTree,j,s) 
          cwIdx = get_cwIdx(cwTree,s);
          cwEst = cwTree.Est(j+1,cwIdx+1).';   
      end
      
      
      % function gets indicies for the stage "s" codeword
      function cwIdx = get_cwIdx(cwTree,s)          
          cwIdx = (2^s - 1) : 2*(2^s - 1);  
      end 
      
      %%
      %H UPDATE
      
      % "H" function updates the stage "s" codeword estimate. Done by
      % combining two stage "s-1" codeword estimates
      
      %inputs:
      % s      --> update stage
      % j      --> index of "top" stage "s-1" codeword estimate
      % cwEst1 --> "bottom" stage "s-1" codeword estimate
      
      %outputs:
      % cs     --> updated stage "s" codeword estiamte
      
      function cs = H(cwTree,s,cwEst1,j)
          
          cwEst0 = get_cwEst(cwTree,j,s-1);
          
          cs = zeros(2^s,1);
          cs(1 : 2^(s-1)) = xor(cwEst0,cwEst1);
          cs(2^(s-1) + 1 : 2^s) = cwEst1;
          
      end
      
      %%
      %F UPDATE
      
      % "F" function updates the j'th stage "s" codeword LLRs. Done using
      % box-plus operator on the j'th stage "s+1" codeword LLRs.
      
      function cwTree = F(cwTree,j,s)
          
          cwLLR = get_cwLLR(cwTree,j,s+1);

          cwLLR0 = cwLLR(1 : 2^s);
          cwLLR1 = cwLLR(2^s+1 : 2^(s+1));
          
          cwLLR_prime = sign(cwLLR0).*sign(cwLLR1).*min(abs(cwLLR0),abs(cwLLR1));
          cwTree = set_cwLLR(cwTree,j,s,cwLLR_prime);
          
      end
      
      %%
      %G UPDATE
      
      % "G" function updates the j'th "bottom" stage "s" codeword LLRs. Done using
      % the "top" stage "s" codeword estimate, and the k'th stage "s+1" codeword LLRs.
      
      %inputs:
      % j      --> index of updated stage "s" codeword
      % s      --> update stage
      % cwEst0 --> "top" stage "s" codeword estimate
      % k      --> index of input stage "s+1" codeword
      
      function cwTree = G(cwTree,j,s,cwEst0,k)
    
          
          cwLLR = get_cwLLR(cwTree,k,s+1);
          cwLLR0 = cwLLR(1 : 2^s);
          cwLLR1 = cwLLR(2^s + 1 : 2^(s+1));
          
          cwLLR_prime = (1 - 2*cwEst0).*cwLLR0 + cwLLR1;
          
          cwTree = set_cwLLR(cwTree,j,s,cwLLR_prime);
          
      end
      
       %%
       %COMPUTE CODEWORD TYPES
       % takes as input infoFlag and computes types of all codewords
       
      function cwTree = computeCwTypes(cwTree, infoFlag)
          
         %initialize "0" stage using infoFlag:
         s = 0;
         for subTreeIdx = 0 : (cwTree.N - 1)
             cwTree = setType(cwTree, infoFlag(subTreeIdx+1), subTreeIdx, s);
         end
         
         %propogte using buttefly structure:
         for s = 1 : (cwTree.n - 2)
             for subTreeIdx = 0 : (2^(cwTree.n - 1 - s) - 1)
                 
                 type0 = getType(cwTree, 2*subTreeIdx  , s-1);
                 type1 = getType(cwTree, 2*subTreeIdx+1, s-1);
                 
                 if (type0 == 0) &&  (type1 == 0)
                     cwTree = setType(cwTree,0,subTreeIdx,s);
%                  
                 elseif (type0 == 1) &&  (type1 == 1)
                     cwTree = setType(cwTree,1,subTreeIdx,s);
                     
                 else
                     cwTree = setType(cwTree,-1,subTreeIdx,s);
                 end
                 
             end
         end
      end
      
      %%
      %EXTRACT CODEWORD TYPE
      
      function type = getType(cwTree, subTreeIdx, idxStage)          
          e = (cwTree.n - 2 - idxStage);
          type = cwTree.cwTypes(2^(e+1) - 2 + subTreeIdx + 1);
      end
      
      %%
      %SET TYPE
      % Set the coding type for a subtree
      
      function cwTree = setType(cwTree, type, subTreeIdx, idxStage)           
         e = (cwTree.n - 2 - idxStage);
         cwTree.cwTypes(2^(e+1) - 2 + subTreeIdx + 1) = type;
      end  
   end
end