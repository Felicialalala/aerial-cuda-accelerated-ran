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

function [errCnt, nTV] = genCfgTV_perf_pdcch(caseSet)

if nargin == 0
    caseSet = ["F08", "F14", "F09"]; % all cases
end


for caseIdx = 1 : length(caseSet)
    switch caseSet(caseIdx)
        case "F14"
            [~, errCnt, nTV] = testCompGenTV_pdcch(2801);  % F14, full cell
            copyfile('./GPU_test_input/TVnr_2801_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-DC-01_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2802);  % F14, ave cell
            copyfile('./GPU_test_input/TVnr_2802_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-DC-02_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2810);  % F14, 16 DCI
            copyfile('./GPU_test_input/TVnr_2810_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-DC-40_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2811);  % F14, 16 DCI, 20 MHz, two symbols
            copyfile('./GPU_test_input/TVnr_2811_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F14-DC-42_PRB273.h5');
            
        case "F08"
            [~, errCnt, nTV] = testCompGenTV_pdcch(2803);  % F08, full cell
            copyfile('./GPU_test_input/TVnr_2803_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-DC-01_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2804);  % F08, ave cell
            copyfile('./GPU_test_input/TVnr_2804_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-DC-02_PRB273.h5'); 
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2807);  % F08, half traffic
            copyfile('./GPU_test_input/TVnr_2807_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-DC-40_PRB273.h5'); 
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2808);  % F08, reduced complexity
            copyfile('./GPU_test_input/TVnr_2808_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F08-DC-RC_PRB273.h5'); 
            
          case "V15"
            [~, errCnt, nTV] = testCompGenTV_pdcch(2809);  % F08, full cell
            copyfile('./GPU_test_input/TVnr_2809_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_V15-DC-01_PRB106.h5');            
            
          case "F09"
            [~, errCnt, nTV] = testCompGenTV_pdcch(2805);  % F09, full cell
            copyfile('./GPU_test_input/TVnr_2805_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F09-DC-01_PRB273.h5');
            
            [~, errCnt, nTV] = testCompGenTV_pdcch(2805);  % F09, ave cell (same as peak but give a different name)
            copyfile('./GPU_test_input/TVnr_2805_PDCCH_gNB_CUPHY_s0p0.h5','./GPU_test_input/TV_cuphy_F09-DC-02_PRB136.h5'); 
    end    
end



end