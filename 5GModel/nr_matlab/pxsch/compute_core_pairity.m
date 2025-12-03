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

function CorePairity = compute_core_pairity(CodedCb,TannerPar,BGN, Zc, i_LS)

%Function computes the core pairty bits. Done by solving a small linear
%equation in mod2. 

%inputs:
% CodedCb --> coded codeblock, currently has values for systematic ...
%             data bits fixed, but not yet the pairty bits. Dim: Zc x nV


%outputs:
%CorePairity --> core pairty bits. Dim: Zc x 4

%PARAMATERS

%coding paramaters:
% BGN = coding.BGN;   %1 or 2. Indicates which base graph used
% Zc = coding.Zc;     %lifting size
% i_LS = coding.i_LS; %lifting set index

%SETUP

%compute value of first four check nodes:
c_init = zeros(Zc,4);

for i = 1 : 4
    c_init(:,i) = compute_check(i,Zc,CodedCb,TannerPar);
end

%BG1

if BGN == 1
    
    if i_LS == 7
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % 105  0    0    *
        % *    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        p = mod(105,Zc);
        CorePairity(:,1) = circshift(CorePairity(:,1),p);
        
        CorePairity(:,2) = c_init(:,1) + CorePairity(:,1);
        CorePairity(:,4) = c_init(:,4) + CorePairity(:,1);
        CorePairity(:,3) = c_init(:,3) + CorePairity(:,4);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % 0    0    0    *
        % *    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,2) = c_init(:,1) + circshift(CorePairity(:,1),-1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,1) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,3) + CorePairity(:,3);
        
    end

end


%BG2


if BGN == 2
    
    if (i_LS == 4) || (i_LS == 8)
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % *    0    0    *
        % 0    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,2) = c_init(:,1) + circshift(CorePairity(:,1),-1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,3) + CorePairity(:,1) + CorePairity(:,3);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % *    0    0    *
        % 1    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = sum(c_init,2);
        CorePairity(:,1) = circshift(CorePairity(:,1),1);
        CorePairity(:,2) = c_init(:,1) + CorePairity(:,1);
        CorePairity(:,3) = c_init(:,2) + CorePairity(:,2);
        CorePairity(:,4) = c_init(:,4) + CorePairity(:,1);
        
    end
    
end

return





