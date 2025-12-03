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

function cwTreeTypes_cell = uci_polar_rx(varargin)

%inputs:
% A_seg_array  --> number of info bits per segment. Dim: nPolUciSegs x 1
% E_seg_array  --> number of tx bits per segment.   Dim: nPolUciSegs x 1
% genTvFlag    --> indicates if Tv should be generated
% tcNumber     --> test-case number, used to name TV H5 file

%outputs
% mismatchFlag --> indicates if mismatch detected between cuPHY and reference model

%%
%PARSE INPUTS

DefaultInputs = {0,0,0,0};
DefaultInputs(1:nargin) = varargin;

A_seg        = DefaultInputs{1};
E_seg        = DefaultInputs{2};
genTvFlag    = DefaultInputs{3};
tcNumber     = DefaultInputs{4};
nPolUciSegs  = length(A_seg);


%%
% DERIVE UCI PARAMATERS

polarUciSegPrms_cell = cell(nPolUciSegs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms_cell{segIdx + 1} = derive_polarUciSegPrms(A_seg(segIdx + 1), E_seg(segIdx + 1));
end

%%
% RUN 

cwTreeTypes_cell = cell(nPolUciSegs,1);
for segIdx = 0 : (nPolUciSegs - 1)
    [~,cwTreeTypes_cell{segIdx + 1}] = compBitTypesKernel(polarUciSegPrms_cell{segIdx + 1});
end

%%
% SAVE TEST VECTOR


if(genTvFlag)
    % create h5 file
    tvDirName = 'GPU_test_input';
    tvName    = strcat('TVnr_',num2str(tcNumber),'_uciPolar');
    [status,msg] = mkdir(tvDirName); 

    h5File = H5F.create([tvDirName filesep tvName '.h5'], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');


    % save polarUciSegPrms 
    polarUciSegPrms = [];
    for segIdx = 0 : (nPolUciSegs - 1)
        polarUciSegPrms(segIdx + 1).nCbs            = uint8(polarUciSegPrms_cell{segIdx + 1}.nCbs);
        polarUciSegPrms(segIdx + 1).K_cw            = uint16(polarUciSegPrms_cell{segIdx + 1}.K_cw);   
        polarUciSegPrms(segIdx + 1).E_cw            = uint16(polarUciSegPrms_cell{segIdx + 1}.E_cw);
        polarUciSegPrms(segIdx + 1).zeroInsertFlag  = uint8(polarUciSegPrms_cell{segIdx + 1}.zeroInsertFlag);
        polarUciSegPrms(segIdx + 1).n_cw            = uint8(polarUciSegPrms_cell{segIdx + 1}.n_cw);
        polarUciSegPrms(segIdx + 1).N_cw            = uint16(polarUciSegPrms_cell{segIdx + 1}.N_cw);
    end
    hdf5_write_nv_exp(h5File, 'polarUciSegPrms', polarUciSegPrms);

    % save cwTreeTypes
    for segIdx = 0 : (nPolUciSegs - 1)
        nameStr = strcat('cwTreeTypes',num2str(segIdx));
        hdf5_write_nv(h5File, nameStr, uint8(cwTreeTypes_cell{segIdx + 1}));
    end

    % sizes:
    sizes = [];
    sizes.nPolUciSegs = uint16(nPolUciSegs);
    hdf5_write_nv_exp(h5File, 'sizes', sizes);

    H5F.close(h5File);
end

