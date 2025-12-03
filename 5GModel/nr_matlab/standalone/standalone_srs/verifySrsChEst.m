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

clear all; 
close all;

%% Config
skipTvGen = 1;
enablePlots = 0;
pruneZeros = 1;

%% Paths
wrkspaceDir = pwd;
tvDirName = 'GPU_TV'; [status,msg] = mkdir(tvDirName);
resultsDirName = 'GPU_results'; [status,msg] = mkdir(resultsDirName);

tvFName = 'GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2_FP16.h5';
%tvFName = 'GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2.h5';

tv = hdf5_load_nv([tvDirName filesep tvFName]);
tvHEst = tv.H; %getfield(hdf5_load_nv([tvDirName filesep tvFName]), 'H');
%load([tvDirName filesep tvFName '.mat']);
%load([tvFName '.mat']);

%% Parameters


%% Save Channel estimation Test Vector
if(~skipTvGen)
    genSrsChEstTv(sp,Y,WFreq,s_grid,s,H_est);
    keyboard;
end

%resultFName = 'gpu_out_002_GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2_FP16.h5';% interp in
%resultFName = 'gpu_out_003_GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2_FP16.h5';% interp out
resultFName = 'gpu_out_208_GPU_TV_SRS_CH_EST_MIMO32x64_PRB272_SRS_SYM2_FP16.h5';

% shift sequence
% gpuShiftSeqPhase = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
% compare(tv.s_in_phase, real(gpuShiftSeqPhase), 'ShiftSequence', enablePlots, pruneZeros, 'MATLAB', 'GPU');

% gpuShiftSeq = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
% compare(tv.s_in, gpuShiftSeq, 'ShiftSequence', enablePlots, pruneZeros, 'MATLAB', 'GPU');

% gpuInterpIn = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
% compare(tv.yk_perm, gpuInterpIn, 'InterpInp', enablePlots, pruneZeros, 'MATLAB', 'GPU');

%compare(tv.yk_perm(:,40:44,4,3), gpuInterpIn(:,40:44,4,3), 'InterpInp', enablePlots, pruneZeros, 'MATLAB', 'GPU');

%gpuHestInterpOut = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
%compare(tv.Hest_preUnshift, gpuHestInterpOut, 'Hest_preUnshift', enablePlots, pruneZeros, 'MATLAB', 'GPU');
% 
% compare(tv.Hest_preUnshift(:,41:44,4,3), gpuHestInterpOut(:,41:44,4,3), 'Hest_preUnshift', enablePlots, pruneZeros, 'MATLAB', 'GPU');

% gpuUnShiftSeq = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
% compare(tv.s_out, gpuUnShiftSeq, 'UnShiftSequence', enablePlots, pruneZeros, 'MATLAB', 'GPU');

% gpuHestPostUnshift = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'Dbg');
% compare(tv.Hest_postUnshift, gpuHestPostUnshift, 'HEst', enablePlots, pruneZeros, 'MATLAB', 'GPU');

%compare(tv.Hest_postUnshift(:,40:44,4,3), gpuHestPostUnshift(:,40:44,4,3), 'HEst', enablePlots, pruneZeros, 'MATLAB', 'GPU');

if 0
gpuHEst = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'HEst');
tvHEstPerm = permute(tvHEst, [1,3,2,4]);
compare(tvHEstPerm(81:88,18,4,7), gpuHEst(81:88,18,4,7), 'Estimated channel', enablePlots, pruneZeros, 'MATLAB', 'GPU');
end

if 1
    % equalizer input compatible order
    gpuHEst = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'HEst'); % Output layout is [nBSAnts nLayers nTotalDataPRB*12]   
    %compare(permute(tvHEst, [3,1,2,4]), gpuHEst, 'Estimated channel', enablePlots, pruneZeros, 'MATLAB', 'GPU');
    compare(permute(tvHEst, [1,3,2,4]), gpuHEst, 'Estimated channel', enablePlots, pruneZeros, 'MATLAB', 'GPU');
else
    % natural order
    gpuHEst = getfield(hdf5_load_nv([resultsDirName filesep resultFName]), 'HEst'); % Output layout is [nTotalDataPRB*12 nLayers nBSAnts Nh]
    gpuHEst = reshape(gpuHEst(:), [Nf, nBSAnts, nLayers]); % Output layout is [nTotalDataPRB*12 nLayers nBSAnts Nh]    
    compare(permute(H_est,[3 1 2]), gpuHEst, 'Estimated channel', enablePlots, pruneZeros, 'MATLAB', 'GPU');
end
