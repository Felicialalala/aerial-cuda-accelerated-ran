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

function [errCnt, nTV] = genCfgTV_perf_prach(caseSet)

if nargin == 0
    caseSet = ["F08", "F14", "F09", "V15"]; % all cases
end


for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F08"
            [~, errCnt, nTV] = testCompGenTV_prach(5801);  % F08, full cell
            copyfile('./GPU_test_input/TVnr_5801_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F08-RA-01.h5');
                
            [~, errCnt, nTV] = testCompGenTV_prach(5802);  % F08, ave cell
            copyfile('./GPU_test_input/TVnr_5802_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F08-RA-02.h5');
            
        case "F14"
            [~, errCnt, nTV] = testCompGenTV_prach(5803);  % F14, full cell
            copyfile('./GPU_test_input/TVnr_5803_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F14-RA-01.h5');        
            
            [~, errCnt, nTV] = testCompGenTV_prach(5804);  % F14, ave cell
            copyfile('./GPU_test_input/TVnr_5804_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F14-RA-02.h5');
            
            [~, errCnt, nTV] = testCompGenTV_prach(5807);  % F14, full cell
            copyfile('./GPU_test_input/TVnr_5807_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F14-RA-40.h5');
         
        case "F09"
            [~, errCnt, nTV] = testCompGenTV_prach(5805);  % F14, ave cell
            copyfile('./GPU_test_input/TVnr_5805_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_F09-RA-01.h5');
        
        case "V15"
            [~, errCnt, nTV] = testCompGenTV_prach(5806);  % F14, ave cell
            copyfile('./GPU_test_input/TVnr_5806_PRACH_gNB_CUPHY_s1p0.h5','./GPU_test_input/TV_cuphy_V15-RA-01.h5');
            
    end
end

end