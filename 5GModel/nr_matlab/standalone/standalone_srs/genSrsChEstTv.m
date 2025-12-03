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

function genSrsChEstTv(srsPrms, freqInterpCoef, dataRx, HEst, dbg, fp16)

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);

nBSAnts = srsPrms.Lgnb;
nCyclicShifts = 4; % UE Antenna ports
nCombs = 4;
nSrsPrb = srsPrms.nPrb;
nSrsSyms = srsPrms.nSym;

srsSymPosBmsk = uint16(0);
for i = 1:length(srsPrms.symIdx)
    srsSymPosBmsk = bitor(srsSymPosBmsk, bitset(0, srsPrms.symIdx(i), 'uint16'));
end
%dec2bin(srsSymBmsk)

nLayers = nCombs*nCyclicShifts*nSrsSyms; % total layer count

nZc = srsPrms.N_zc;
zcSeqNum = srsPrms.q;
delaySpread = srsPrms.delaySpread;

% layerPeel - cyclicShifts, comb, srsSym
srsChEstCfg.nRxAnts = uint16(nBSAnts);
srsChEstCfg.nLayers = uint8(nLayers);
srsChEstCfg.nPrb = uint16(nSrsPrb);
srsChEstCfg.scsKHz = uint16(srsPrms.scs/10^3);
srsChEstCfg.nCyclicShifts = uint8(nCyclicShifts);
srsChEstCfg.nCombs = uint8(nCombs);
srsChEstCfg.nZc = uint16(nZc);
srsChEstCfg.zcSeqNum = uint8(zcSeqNum);
srsChEstCfg.srsSymPosBmsk = uint16(srsSymPosBmsk);
srsChEstCfg.delaySpreadSecs = single(delaySpread);

% Test vector in HDF5 file format
if fp16
    tvName = sprintf('GPU_TV_SRS_CH_EST_MIMO%dx%d_PRB%d_SRS_SYM%d_FP16.h5',nLayers, nBSAnts, nSrsPrb, nSrsSyms);
else
    tvName = sprintf('GPU_TV_SRS_CH_EST_MIMO%dx%d_PRB%d_SRS_SYM%d.h5',nLayers, nBSAnts, nSrsPrb, nSrsSyms);
end
h5File = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
hdf5_write_nv(h5File, 'srsChEstCfg', srsChEstCfg);
if fp16
    hdf5_write_nv(h5File, 'DataRx', single(dataRx), 'fp16');
else
    hdf5_write_nv(h5File, 'DataRx', single(dataRx));
end
hdf5_write_nv(h5File, 'FreqInterpCoefs', single(real(freqInterpCoef))); % [nSrsChEstOut, nSrsTonesIn, nComb]
hdf5_write_nv(h5File, 'H', single(HEst));
hdf5_write_nv(h5File, 's_in', single(dbg.s_in));
hdf5_write_nv(h5File, 's_in_phase', single(dbg.s_in_phase));
hdf5_write_nv(h5File, 'yk_perm', single(dbg.yk_perm));
hdf5_write_nv(h5File, 'Hest_preUnshift', single(dbg.Hest_preUnshift));
hdf5_write_nv(h5File, 's_out', single(dbg.s_out));
hdf5_write_nv(h5File, 'Hest_postUnshift', single(dbg.Hest_postUnshift));
H5F.close(h5File);

fprintf('Generated TV: %s\n', tvName);

end
