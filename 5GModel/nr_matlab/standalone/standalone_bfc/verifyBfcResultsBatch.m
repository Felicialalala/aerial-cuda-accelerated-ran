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

close all; clear all;

enableTvGen = 1;
enablePlots = 0;
pruneZeros = 0;

%%
tv.fName = {
    'BfcCoef_MIMO16x64_NumCoef52',
    'BfcCoef_MIMO16x64_NumCoef76',
    'BfcCoef_MIMO16x64_NumCoef84',
    'BfcCoef_MIMO16x64_NumCoef136'
    };
tv.resultFName = {
    'GPU_OUT_3_GPU_TV_BFC_COEF_MIMO16x64_N_COEF52.h5',
    'GPU_OUT_1_GPU_TV_BFC_COEF_MIMO16x64_N_COEF76.h5',
    'GPU_OUT_2_GPU_TV_BFC_COEF_MIMO16x64_N_COEF84.h5',
    'GPU_OUT_0_GPU_TV_BFC_COEF_MIMO16x64_N_COEF136.h5'
    }; 
%tvFName = 'BfcCoef_MIMO16x64_NumCoef1600';
%tvFName = 'BfcCoef_MIMO16x64_NumCoef2400';
%tvFName = 'BfcCoef_MIMO16x64_NumCoef3200';

tvDirName = [pwd filesep 'GPU_test_input'];
resultsDirName = [pwd filesep 'GPU_results'];

%tvCoef = getfield(hdf5_load_nv([tvDirName filesep 'Gpu_tv_' tvFName '.h5']), 'Coef');

if enableTvGen
    for i = 1:length(tv.fName)
        load([tvDirName filesep tv.fName{i} '.mat']);
        genBfcCoefTv(bfc,tvH,tvCoef);
    end
    keyboard;
end

%%

for i = 1:length(tv.resultFName)    
    load([tvDirName filesep tv.fName{i} '.mat']);
    
    fprintf('==============================================================\n');
    fprintf('Comparing: %s\n', tv.resultFName{i});
    
    if 0
        gpuH = getfield(hdf5_load_nv([resultsDirName filesep tv.resultFName{i}]), 'Dbg');
        compare(permute(tvH, [2 3 1]), gpuH, 'Channel Coefficients', enablePlots, pruneZeros);
    end
    
    if 0
        gpuA0 = getfield(hdf5_load_nv([resultsDirName filesep tv.resultFName{i}]), 'Dbg');
        compare(tvA0, gpuA0, 'A0 Coefficients', enablePlots, pruneZeros);
    end
    
    if 0
        gpuA1 = getfield(hdf5_load_nv([resultsDirName filesep tv.resultFName{i}]), 'Dbg');
        compare(tvA1, gpuA1, 'A1 Coefficients', enablePlots, pruneZeros);
    end
    
    gpuCoef = getfield(hdf5_load_nv([resultsDirName filesep tv.resultFName{i}]), 'Coef');
    compare(tvCoef, gpuCoef, 'BFC Coefficients', enablePlots, pruneZeros);
end
