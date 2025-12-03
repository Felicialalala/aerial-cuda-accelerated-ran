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

function save_TbCodedCbs(TbCodedCbs, TbCbs, PuschCfg)

% Create H5 file

% Save data

% Close H5 file


C = PuschCfg.coding.C;
BG = PuschCfg.coding.BGN;
rv = 0;
K = PuschCfg.coding.K;
R = PuschCfg.coding.codeRate;
Z = PuschCfg.coding.Zc;
nV_parity = PuschCfg.coding.nV_parity;

if BG==1
    %len = 3*Z*22+2*Z;
    %punclen = size(TbCodedCbs,1);
    %TbCodedCbs = [TbCodedCbs ; zeros(len - punclen, C)];
    %TbCodedCbs = [zeros(2*Z, C); TbCodedCbs ; zeros(len - punclen, C)];
    %TbCodedCbs = [zeros(2*Z, C); TbCodedCbs ; zeros(len - punclen - 2*Z, C)];
end

if BG==2
    len = 5*Z*10;
    punclen = size(TbCodedCbs,1);
    TbCodedCbs = [TbCodedCbs ; zeros(len - punclen, C)];
    
end


if exist('GPU_test_input') == 0
    mkdir GPU_test_input
end

if exist('GPU_test_input', 'dir')    
    cd GPU_test_input
    if BG==1
    fname = sprintf('ldpc_BG-%i_Zc-%3.0f_C-%i_R-%1.2f.h5', BG, K/22, C, R);
    elseif BG==2
        fname = sprintf('ldpc_BG-%i_Zc-%3.0f_C-%i_R-%1.2f.h5', BG, K/10, C, R);
    end
    
    fprintf('Creating file: %s\n', fname);
    h5File  = H5F.create(fname, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    
    % write parameters
    hdf5_write_nv(h5File, 'C',   uint32(C));
    hdf5_write_nv(h5File, 'BG',   uint8(BG));
    hdf5_write_nv(h5File, 'K',   uint32(K));
    hdf5_write_nv(h5File, 'K_b',   uint32(PuschCfg.coding.K_b));
    hdf5_write_nv(h5File, 'rv',   uint32(rv));
    hdf5_write_nv(h5File, 'Z',   uint32(Z));
    hdf5_write_nv(h5File, 'R',   R);
    hdf5_write_nv(h5File, 'nV_parity',   uint8(nV_parity));
    PuschCfg.coding.qamstr = 0;
    PuschCfg.coding.CRC = 0;
    hdf5_write_nv(h5File, 'PuschCfgCoding',   PuschCfg.coding);
    hdf5_write_nv(h5File, 'PuschCfgAlloc',   PuschCfg.alloc);
    hdf5_write_nv(h5File, 'PuschCfgMimo',   PuschCfg.mimo);

    % write data
    %tmp = rem(size(TbCbs, 1), 32);
    %TbCbs = [TbCbs; zeros(32 - tmp, size(TbCbs,2))];
    hdf5_write_nv(h5File, 'TbCbsUncoded',   uint8(TbCbs));
    
%     tmp = rem(size(TbCodedCbs, 1), 32);
%     TbCodedCbs = [TbCodedCbs; zeros(32 - tmp, size(TbCodedCbs,2))];
    hdf5_write_nv(h5File, 'TbCbsCoded',   uint8(TbCodedCbs(2*Z+1:end,:)));
     
    H5F.close(h5File);
    fprintf('HDF5 file generation success.\n');
    cd ..
end


