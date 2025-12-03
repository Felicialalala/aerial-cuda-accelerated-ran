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

function [totNumPolCbs, polarCbPrms_cell, polarUciSegPrms_cell] = derive_polarCbPrms(nPolUciSegs, polarUciSegPrms_cell)

% Functions uses polarUciSegPrms to populate polarCbPrms and links them
% together.

%inputs:
% nPolUciSegs          --> number of polar uci segments
% polarUciSegPrms_cell --> cell containing uci seg prms. Dim: nPolUciSegs x 1

% outputs:
% totNumPolCbs         --> total number of polar codeblocks
% polarCbPrms_cell     --> polar cb prms. Dim: totNumPolCbs x 1
% polarUciSegPrms_cell --> updated uci seg prms. Now points to their children.

%%
%START

maxNumCbs        = 10000;
polarCbPrms_cell = cell(maxNumCbs,1);
totNumPolCbs     = 0;

for uciIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms_cell{uciIdx + 1}.childCbIdxs = zeros(2,1);
    
    for i = 1 : polarUciSegPrms_cell{uciIdx + 1}.nCbs      
        % copy paramaters:
        polarCbPrms_cell{totNumPolCbs + 1}.K_cw     = polarUciSegPrms_cell{uciIdx + 1}.K_cw;
        polarCbPrms_cell{totNumPolCbs + 1}.E_cw     = polarUciSegPrms_cell{uciIdx + 1}.E_cw;
        polarCbPrms_cell{totNumPolCbs + 1}.N_cw     = polarUciSegPrms_cell{uciIdx + 1}.N_cw;
        polarCbPrms_cell{totNumPolCbs + 1}.n_cw     = polarUciSegPrms_cell{uciIdx + 1}.n_cw;
        polarCbPrms_cell{totNumPolCbs + 1}.nCrcBits = polarUciSegPrms_cell{uciIdx + 1}.nCrcBits;
        
        % link:
        polarUciSegPrms_cell{uciIdx + 1}.childCbIdxs(i) = totNumPolCbs;
        
        % update cb counter:
        totNumPolCbs = totNumPolCbs + 1;
    end
end


        
        
    
    

