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

function genBfcCoefTv(bfc,H,Coef, tvName)

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);

[nTxBBPorts,nTxLayers,nCoefs] = size(Coef);
if((nTxBBPorts ~= bfc.L_gNB) || (nTxLayers ~= bfc.nLayers) || (nCoefs ~= bfc.nWeights))
    error('Mismtach in coefficient dimension (nTxBBPorts %d, nLayers %d, nCoefs %d) and BFC config (nTxBBPorts %d, nLayers %d, nCoefs %d)', nTxBBPorts, nTxLayers, nCoefs, bfc.L_gNB, bfc.nLayers, bfc.nWeights);
end
bfcCfg.nTxLayers = uint32(bfc.nLayers);
bfcCfg.nTxBBPorts = uint32(bfc.L_gNB);
bfcCfg.nCoefs = uint32(bfc.nWeights);

h5File = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'bfcCfg', bfcCfg);
hdf5_write_nv(h5File, 'Lambda', single(bfc.lambda.*ones(nTxLayers,bfc.nWeights)));
hdf5_write_nv(h5File, 'H', single(permute(H, [1 3 2]))); % [nCoefs nTxLayers nTxBBports] to [nCoefs nTxBBports nTxLayers]
hdf5_write_nv(h5File, 'Coef', single(Coef));
H5F.close(h5File);

end
