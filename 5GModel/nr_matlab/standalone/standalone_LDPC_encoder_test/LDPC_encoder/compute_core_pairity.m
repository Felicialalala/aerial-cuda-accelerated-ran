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

function CorePairity = compute_core_pairity(coreChecks,BGN,i_LS,Zc)

%Function computes the core pairty bits. Done by solving a 4Zc x 4Zc
%equation via back-substitution. There are four fixed ways to do this
%depending on BGN and lifting index.

%inputs:
%coreChecks --> core check equations. Dim: Zc x 4
%BGN        --> base graph number
%i_LS       --> lifting set index

%outputs:
%CorePairity --> core pairty bits. Dim: Zc x 4

%%
%BG1

if BGN == 1
    
    if i_LS == 7
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % 105  0    0    *
        % *    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = mod(sum(coreChecks,2),2);
        p = mod(105,Zc);
        CorePairity(:,1) = circshift(CorePairity(:,1),p);
        
        CorePairity(:,2) = mod(coreChecks(:,1) + CorePairity(:,1),2);
        CorePairity(:,4) = mod(coreChecks(:,4) + CorePairity(:,1),2);
        CorePairity(:,3) = mod(coreChecks(:,3) + CorePairity(:,4),2);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % 0    0    0    *
        % *    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = mod(sum(coreChecks,2),2);
        CorePairity(:,2) = mod(coreChecks(:,1) + circshift(CorePairity(:,1),-1),2);
        CorePairity(:,3) = mod(coreChecks(:,2) + CorePairity(:,1) + CorePairity(:,2),2);
        CorePairity(:,4) = mod(coreChecks(:,3) + CorePairity(:,3),2);
        
    end

end


%%
%BG2


if BGN == 2
    
    if (i_LS == 4) || (i_LS == 8)
        
        % Tanner matrix for core pairty bits:
        % 1    0    *    *
        % *    0    0    *
        % 0    *    0    0
        % 1    *    *    0
        
        CorePairity(:,1) = mod(sum(coreChecks,2),2);
        CorePairity(:,2) = mod(coreChecks(:,1) + circshift(CorePairity(:,1),-1),2);
        CorePairity(:,3) = mod(coreChecks(:,2) + CorePairity(:,2),2);
        CorePairity(:,4) = mod(coreChecks(:,3) + CorePairity(:,1) + CorePairity(:,3),2);
        
    else
        
        % Tanner matrix for core pairty bits:
        % 0    0    *    *
        % *    0    0    *
        % 1    *    0    0
        % 0    *    *    0
        
        CorePairity(:,1) = mod(sum(coreChecks,2),2);
        CorePairity(:,1) = mod(circshift(CorePairity(:,1),1),2);
        CorePairity(:,2) = mod(coreChecks(:,1) + CorePairity(:,1),2);
        CorePairity(:,3) = mod(coreChecks(:,2) + CorePairity(:,2),2);
        CorePairity(:,4) = mod(coreChecks(:,4) + CorePairity(:,1),2);
        
    end
    
end


        
        
        
        
        
        
        
        
        
        
        
        
        




        
        
        
        
        
        
        
        










