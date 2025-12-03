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

function SysPar = updateAlloc(SysPar)
% function SysPar = updateSysPar(SysPar)
%
% This function derives all the other parameters based on the pre-set
% configurations
%
% Input:    SysPar: structure with all simulation configurations
%
% Output:   SysPar: structure with all simulation configurations
%

testAlloc = SysPar.testAlloc;
N_UE = SysPar.SimCtrl.N_UE;

% config for gNB

SysPar.SimCtrl.gNB.Nant = SysPar.carrier.Nant_gNB;                % number of antennas

SysPar.SimCtrl.gNB.tx.alloc = [];
allocIdx_gNB_tx = 1;

if isfield(testAlloc, 'ssb') && testAlloc.ssb
    SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.type = 'ssb';     % alloc to be tested
    SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.idx = 1;
    allocIdx_gNB_tx = allocIdx_gNB_tx + 1;
end
if isfield(testAlloc, 'pdcch')
    for idx = 1:testAlloc.pdcch
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.type = 'pdcch';
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.idx = idx;
        allocIdx_gNB_tx = allocIdx_gNB_tx + 1;
    end
end
if isfield(testAlloc, 'csirs')
    for idx = 1:testAlloc.csirs
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.type = 'csirs';
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.idx = idx;
        allocIdx_gNB_tx = allocIdx_gNB_tx + 1;
    end
end
if isfield(testAlloc, 'pdsch')
    for idx = 1:testAlloc.pdsch
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.type = 'pdsch';
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.idx = idx;
        allocIdx_gNB_tx = allocIdx_gNB_tx + 1;
    end
end
if isfield(testAlloc, 'bfw')
    for idx = 1:testAlloc.bfw
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.type = 'bfw';
        SysPar.SimCtrl.gNB.tx.alloc{allocIdx_gNB_tx}.idx = idx;
        allocIdx_gNB_tx = allocIdx_gNB_tx + 1;
    end
end


SysPar.SimCtrl.gNB.rx.alloc = [];
allocIdx_gNB_rx = 1;

SysPar.SimCtrl.results = [];

if isfield(testAlloc, 'prach')
    for idx = 1:testAlloc.prach
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.type = 'prach';   % alloc to be tested
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.idx = idx;
        SysPar.SimCtrl.results.prach{idx}.totalCnt = 0;
        SysPar.SimCtrl.results.prach{idx}.falseCnt = 0;
        SysPar.SimCtrl.results.prach{idx}.missCnt = 0;
        SysPar.SimCtrl.results.prach{idx}.prmbCnt = 0;
        allocIdx_gNB_rx = allocIdx_gNB_rx + 1;
    end
end
if isfield(testAlloc, 'pucch')
    for idx = 1:testAlloc.pucch
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.type = 'pucch';
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.idx = idx;
        SysPar.SimCtrl.results.pucch{idx}.totalCnt = 0;
        SysPar.SimCtrl.results.pucch{idx}.totalUciCnt = 0;
        SysPar.SimCtrl.results.pucch{idx}.misssr = 0;
        SysPar.SimCtrl.results.pucch{idx}.falsesr = 0;
        SysPar.SimCtrl.results.pucch{idx}.missack = 0;
        SysPar.SimCtrl.results.pucch{idx}.nack2ack = 0;
        SysPar.SimCtrl.results.pucch{idx}.falseCnt = 0;
        SysPar.SimCtrl.results.pucch{idx}.errorCnt = 0;
        SysPar.SimCtrl.results.pucch{idx}.snrdB = [];
        SysPar.SimCtrl.results.pucch{idx}.taEstMicroSec = [];
        allocIdx_gNB_rx = allocIdx_gNB_rx + 1;
    end
end
if isfield(testAlloc, 'srs')
    for idx = 1:testAlloc.srs
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.type = 'srs';
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.idx = idx;
        SysPar.SimCtrl.results.srs{idx}.totalCnt = 0;
        SysPar.SimCtrl.results.srs{idx}.snrErr = 0;
        SysPar.SimCtrl.results.srs{idx}.toErr = 0;
        SysPar.SimCtrl.results.srs{idx}.hestErr = 0;
        allocIdx_gNB_rx = allocIdx_gNB_rx + 1;
    end
end
if isfield(testAlloc, 'pusch')
    for idx = 1:testAlloc.pusch
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.type = 'pusch';
        SysPar.SimCtrl.gNB.rx.alloc{allocIdx_gNB_rx}.idx = idx;
        SysPar.SimCtrl.results.pusch{idx}.cbCnt = 0;
        SysPar.SimCtrl.results.pusch{idx}.cbErrorCnt = 0;
        SysPar.SimCtrl.results.pusch{idx}.tbCnt = 0;
        SysPar.SimCtrl.results.pusch{idx}.tbErrorCnt = 0; 
        SysPar.SimCtrl.results.pusch{idx}.harqErrCnt = 0; 
        SysPar.SimCtrl.results.pusch{idx}.csi1ErrCnt = 0; 
        SysPar.SimCtrl.results.pusch{idx}.csi2ErrCnt = 0; 
        SysPar.SimCtrl.results.pusch{idx}.cfoEstHz = []; 
        SysPar.SimCtrl.results.pusch{idx}.toEstMicroSec = []; 
        SysPar.SimCtrl.results.pusch{idx}.sinrdB = []; 
        SysPar.SimCtrl.results.pusch{idx}.postEqSinrdB = []; 
        SysPar.SimCtrl.results.pusch{idx}.LDPC_numItr = [];
        SysPar.SimCtrl.results.pusch{idx}.LDPC_badItrCnt = [];
        SysPar.SimCtrl.results.pusch{idx}.cbErr = [];
        SysPar.SimCtrl.results.pusch{idx}.derateCbs = [];
        SysPar.SimCtrl.results.pusch{idx}.derateCbs_percentiles = [];
        SysPar.SimCtrl.results.pusch{idx}.derateCbs_centralMoments = [];
        SysPar.SimCtrl.results.pusch{idx}.Hest = [];
        SysPar.SimCtrl.results.pusch{idx}.Hgenie = [];
        SysPar.SimCtrl.results.pusch{idx}.interf_Hgenie = [];
        SysPar.SimCtrl.results.pusch{idx}.genie_nCov = [];
        SysPar.SimCtrl.results.pusch{idx}.genie_CQI = [];
        allocIdx_gNB_rx = allocIdx_gNB_rx + 1;
    end
end

% config for UE

for idxUE = 1:N_UE
    SysPar.SimCtrl.UE{idxUE}.Nant = SysPar.carrier.Nant_UE;                  % number of antennas
    SysPar.SimCtrl.UE{idxUE}.rx.alloc = [];
    allocIdx_UE_rx(idxUE) = 1;
end

if isfield(testAlloc, 'ssb') && testAlloc.ssb
    ssb = SysPar.ssb;
    for idxUE = 1:N_UE
        if 1 % ismember(idxUE-1, ssb.idxUE)
            SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.type = 'ssb';   % alloc to be tested
            SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.idx = 1;   % alloc to be tested
            SysPar.SimCtrl.results.ssb{idxUE}.totalCnt = 0;
            SysPar.SimCtrl.results.ssb{idxUE}.errCnt = 0;
            allocIdx_UE_rx(idxUE) = allocIdx_UE_rx(idxUE) + 1;
        end
    end
end
if isfield(testAlloc, 'pdcch')
    pdcch = SysPar.pdcch;
    for idx = 1:testAlloc.pdcch
        for idxUE = 1:N_UE
            if ismember(idxUE-1, pdcch{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.type = 'pdcch';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.idx = idx;   % alloc to be tested
                SysPar.SimCtrl.results.pdcch{idxUE}.totalCnt = 0;
                SysPar.SimCtrl.results.pdcch{idxUE}.errCnt = 0;
                allocIdx_UE_rx(idxUE) = allocIdx_UE_rx(idxUE) + 1;
            end
        end
    end
end
if isfield(testAlloc, 'csirs')
    csirs = SysPar.csirs;
    for idx = 1:testAlloc.csirs
        for idxUE = 1:N_UE
            if 1 % ismember(idxUE-1, csirs{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.type = 'csirs';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.idx = idx;   % alloc to be tested
                SysPar.SimCtrl.results.csirs{idxUE}.totalCnt = 0;
                SysPar.SimCtrl.results.csirs{idxUE}.errCnt = 0;
                allocIdx_UE_rx(idxUE) = allocIdx_UE_rx(idxUE) + 1;
            end
        end
    end
end
if isfield(testAlloc, 'pdsch')
    pdsch = SysPar.pdsch;
    for idx = 1:testAlloc.pdsch
        for idxUE = 1:N_UE
            if ismember(idxUE-1, pdsch{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.type = 'pdsch';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.rx.alloc{allocIdx_UE_rx(idxUE)}.idx = idx;   % alloc to be tested
                SysPar.SimCtrl.results.pdsch{idxUE}.cbCnt = 0;
                SysPar.SimCtrl.results.pdsch{idxUE}.cbErrorCnt = 0;
                SysPar.SimCtrl.results.pdsch{idxUE}.tbCnt = 0;
                SysPar.SimCtrl.results.pdsch{idxUE}.tbErrorCnt = 0;
                SysPar.SimCtrl.results.pdsch{idxUE}.evm = [];
                allocIdx_UE_rx(idxUE) = allocIdx_UE_rx(idxUE) + 1;
            end
        end
    end
end



for idxUE = 1:N_UE
    SysPar.SimCtrl.UE{idxUE}.tx.alloc = [];
    allocIdx_UE_tx(idxUE) = 1;
end

if isfield(testAlloc, 'prach')
    prach = SysPar.prach;
    for idx = 1:testAlloc.prach
        for idxUE = 1:N_UE
            if ismember(idxUE-1, prach{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.type = 'prach';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.idx = idx;   % alloc to be tested
                allocIdx_UE_tx(idxUE) = allocIdx_UE_tx(idxUE) + 1;
            end
        end
    end
end
if isfield(testAlloc, 'pucch')
    pucch = SysPar.pucch;
    for idx = 1:testAlloc.pucch
        for idxUE = 1:N_UE
            if ismember(idxUE-1, pucch{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.type = 'pucch';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.idx = idx;   % alloc to be tested
                allocIdx_UE_tx(idxUE) = allocIdx_UE_tx(idxUE) + 1;
            end
        end
    end
end
if isfield(testAlloc, 'srs')
    srs = SysPar.srs;
    for idx = 1:testAlloc.srs
        for idxUE = 1:N_UE
            if ismember(idxUE-1, srs{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.type = 'srs';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.idx = idx;   % alloc to be tested
                allocIdx_UE_tx(idxUE) = allocIdx_UE_tx(idxUE) + 1;
            end
        end
    end
end
if isfield(testAlloc, 'pusch')
    pusch = SysPar.pusch;
    for idx = 1:testAlloc.pusch
        for idxUE = 1:N_UE
            if ismember(idxUE-1, pusch{idx}.idxUE)
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.type = 'pusch';   % alloc to be tested
                SysPar.SimCtrl.UE{idxUE}.tx.alloc{allocIdx_UE_tx(idxUE)}.idx = idx;   % alloc to be tested
                allocIdx_UE_tx(idxUE) = allocIdx_UE_tx(idxUE) + 1;
            end
        end
    end
end

return
