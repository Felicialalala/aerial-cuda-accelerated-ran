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

harqTcExel   = 'harqTcs.xlsx';
maxUesPerTc  = 3;
maxNumHarqPr = 16;
maxNumTbs    = 100;


%%
% HARQ PARAMTERS

 puschSlotIdxsInFrame = [3 9 14 19];

      % TC#   mcsTable  mcs  nl  rb0  Nrb  sym0   Nsym  SCID  BWP0   nBWP   RNTI  rvIdx dataScId  dmrs0  maxLen addPos dmrsScId nCdm port0 nAnt slotIdx
CFG = {7346,  1,        1,   1,  0,     8,  0,     4,     0,    0,    273, 20001,    0,    41,     2,      1,     0,     41,      2,   0,   4,     0};

harqTvPrm_cell = cell(4,1);

harqTvPrm_1Tx      = [];
harqTvPrm_1Tx.gain = 1;
harqTvPrm_1Tx.rv   = 0;
harqTvPrm_cell{1}  = harqTvPrm_1Tx;

harqTvPrm_2Tx      = [];
harqTvPrm_2Tx.gain = sqrt(10^(-49/10));
harqTvPrm_2Tx.rv   = [0 2];
harqTvPrm_cell{2}  = harqTvPrm_2Tx;

harqTvPrm_3Tx      = [];
harqTvPrm_3Tx.gain = sqrt(10^(-50/10));
harqTvPrm_3Tx.rv   = [0 2 3];
harqTvPrm_cell{3}  = harqTvPrm_3Tx;

harqTvPrm_4Tx      = [];
harqTvPrm_4Tx.gain = sqrt(10^(-51.5/10));
harqTvPrm_4Tx.rv   = [0 2 3 1];
harqTvPrm_cell{4}  = harqTvPrm_4Tx;

%%
% PARSE EXCEL FILE

T      = readcell(harqTcExel);
nSlots = size(T, 2) - 1;
nTc    = floor(size(T,1) / 4);

for tcIdx = 0 : (nTc - 1)
% for tcIdx = 17:20
    nTxsPerPrTbUe          = zeros(maxNumHarqPr, maxNumTbs, maxUesPerTc);
    puschSlotRequests_cell = cell(nSlots , 1);
    
    for slotIdx = 0 : (nSlots - 1)
        slotRequest = [];
        slotRequest.ueIdxs     = [];
        slotRequest.harqPrIdxs = [];
        slotRequest.tbIdxs     = [];
        slotRequest.nUeTxs     = 0;
        slotRequest.txIdxs     = [];

    
        for ueIdx = 0 : (maxUesPerTc - 1)
            excelRowIdx         = tcIdx * (maxUesPerTc + 1) + ueIdx;
            ueSlotRequestString = T{excelRowIdx + 1, slotIdx + 2};
            
            if(length(ueSlotRequestString) > 1)
                slotRequest.ueIdxs = [slotRequest.ueIdxs ueIdx];
                
                tbIdx = str2num(ueSlotRequestString(3)) * 10 + str2num(ueSlotRequestString(4));
                if(tbIdx > maxNumTbs)
                    error('Error: tb idx exceds mas number of Tbs');
                end
                slotRequest.tbIdxs = [slotRequest.tbIdxs tbIdx];

                harqPrIdx = str2num(ueSlotRequestString(8)) * 10 + str2num(ueSlotRequestString(9));
                if(harqPrIdx > maxNumHarqPr)
                    error('Error: Harq pr idx exceds mas number of Harq processes (16)');
                end
                slotRequest.harqPrIdxs = [slotRequest.harqPrIdxs harqPrIdx];
                
                slotRequest.txIdxs                                 = [slotRequest.txIdxs nTxsPerPrTbUe(harqPrIdx + 1, tbIdx + 1, ueIdx + 1)];
                nTxsPerPrTbUe(harqPrIdx + 1, tbIdx + 1, ueIdx + 1) = nTxsPerPrTbUe(harqPrIdx + 1, tbIdx + 1, ueIdx + 1) + 1;
                slotRequest.nUeTxs                                 = slotRequest.nUeTxs + 1;
            end
        end
        
        puschSlotRequests_cell{slotIdx + 1} = slotRequest;
    end
    
    if(max(nTxsPerPrTbUe(:)) > 4)
        error('Error: Tcs currently only support 4 Harq Txs');
    end
    
    generatePuschHarqTvs(puschSlotRequests_cell, nTxsPerPrTbUe, harqTvPrm_cell, CFG, puschSlotIdxsInFrame, maxNumHarqPr, maxNumTbs, tcIdx);
    
end


%%
%



function generatePuschHarqTvs(puschSlotRequests_cell, nTxsPerPrTbUe, harqTvPrm_cell, CFG, puschSlotIdxsInFrame, maxNumHarqPr, maxNumTbs, tcIdx)

    % slots paramaters
    nPuschSlots         = length(puschSlotRequests_cell);
    nPuschSlotsPerFrame = length(puschSlotIdxsInFrame);

    % common system paramaters:
    idxSet = 1;
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc.pusch = 1;
    SysPar = initSysPar(testAlloc);
    SysPar.pusch{1}.mcsTable = CFG{idxSet, 2};
    SysPar.pusch{1}.mcsIndex =  CFG{idxSet, 3};
    SysPar.pusch{1}.nrOfLayers = CFG{idxSet, 4};
    SysPar.pusch{1}.rbStart = CFG{idxSet, 5};
    SysPar.pusch{1}.rbSize = CFG{idxSet, 6};
    SysPar.pusch{1}.StartSymbolIndex = CFG{idxSet, 7};
    SysPar.pusch{1}.NrOfSymbols = CFG{idxSet, 8};
    SysPar.pusch{1}.SCID =  CFG{idxSet, 9};
    SysPar.pusch{1}.BWPStart =  CFG{idxSet, 10};
    SysPar.pusch{1}.BWPSize =  CFG{idxSet, 11};
    SysPar.pusch{1}.RNTI =  CFG{idxSet, 12};
    SysPar.pusch{1}.rvIndex =  CFG{idxSet, 13};
    SysPar.pusch{1}.dataScramblingId =  CFG{idxSet, 14};
    sym0 = SysPar.pusch{1}.StartSymbolIndex;
    nSym = SysPar.pusch{1}.NrOfSymbols;
    dmrs0 = CFG{idxSet, 15};
    SysPar.carrier.dmrsTypeAPos = dmrs0;
    maxLen = CFG{idxSet, 16};
    addPos = CFG{idxSet, 17};
    DmrsSymbPos = findDmrsSymbPos(sym0, nSym, dmrs0, maxLen, addPos, 'UL', 'typeA');
    SysPar.pusch{1}.DmrsSymbPos = DmrsSymbPos;
    SysPar.pusch{1}.DmrsScramblingId =  CFG{idxSet, 18};
    SysPar.pusch{1}.numDmrsCdmGrpsNoData =  CFG{idxSet, 19};
    SysPar.pusch{1}.portIdx = CFG{idxSet, 20} + [0:SysPar.pusch{1}.nrOfLayers-1];
    SysPar.carrier.Nant_gNB =  CFG{idxSet, 21};
    SysPar.carrier.N_ID_CELL = 41;
    SysPar.SimCtrl.timeDomainSim = 1;
     SysPar.SimCtrl.genTV.enable = 1;
    SysPar.SimCtrl.alg.enableCfoEstimation = 0;
    SysPar.SimCtrl.alg.enableCfoCorrection = 0;
    SysPar.SimCtrl.alg.enableToEstimation = 0;
    SysPar.SimCtrl.alg.enableToCorrection = 0;
    SysPar.SimCtrl.alg.TdiMode = 0;
    SysPar.SimCtrl.genTV.fakeSlotNumber = 0;

    %start:
    for puschSlotIdx = 0 : (nPuschSlots - 1)
        slotIdx  = puschSlotIdxsInFrame(mod(puschSlotIdx, nPuschSlotsPerFrame) + 1);
        frameIdx = floor(puschSlotIdx / nPuschSlotsPerFrame);

        slotRequest = puschSlotRequests_cell{puschSlotIdx + 1};
        nUeTxs      = slotRequest.nUeTxs;

        if(nUeTxs > 0)
            for i = 0 : (nUeTxs - 1)
                ueIdx     = slotRequest.ueIdxs(i + 1);
                tbIdx     = slotRequest.tbIdxs(i + 1);
                txIdx     = slotRequest.txIdxs(i + 1);
                harqPrIdx = slotRequest.harqPrIdxs(i + 1);

                % test-vector paramaters:
                totalNumTxs                  = nTxsPerPrTbUe(harqPrIdx + 1, tbIdx + 1, ueIdx + 1);
                harqTvPrm                    = harqTvPrm_cell{totalNumTxs};
                SysPar.Chan{i + 1}.gain      = harqTvPrm.gain;

                % user specific paramaters:
                SysPar.pusch{i + 1}               = SysPar.pusch{1};
                SysPar.pusch{i + 1}.startPrb      = ueIdx * SysPar.pusch{1}.rbSize;
                SysPar.pusch{i + 1}.RNTI          = 20000 + ueIdx;
                SysPar.pusch{i + 1}.seed          = harqPrIdx + maxNumHarqPr * tbIdx + maxNumHarqPr * maxNumTbs * ueIdx;
                SysPar.pusch{i + 1}.rv            = harqTvPrm.rv(txIdx + 1);
                SysPar.pusch{i + 1}.harqProcessID = harqPrIdx;
                if(txIdx > 1)
                   SysPar.pusch{i + 1}.ndi = 0;
                else
                   SysPar.pusch{i + 1}.ndi = 1;
                end
                
                % generate test-vectors:
                SysPar.SimCtrl.genTV.TVname = sprintf('harqTV_%04d_frame%02d_slot%02d', tcIdx, frameIdx, slotIdx);
                [SysPar, UE, gNB]           = nrSimulator(SysPar);
            end
        end
    end
end
        
        
        
        
        
        
        
        






