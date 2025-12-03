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

function UE = updateUE(SysPar, UE, idxUE)
% function UE = initUE(SysPar)
%
% This function initializes an UE. It prepares paramaters, load tables,
% pre-generate some variables for each allocation based on the config
% from SysPar.
%
% Input:    SysPar: structure with all configurations
%
% Output:   UE: structure for a single UE
%

UE.Mac = UeMacUpdateConfig(SysPar, UE.Mac, idxUE);
UE.FAPIconfig = UeMacUpdateConfigToPhy(UE.Mac, UE.FAPIconfig);
UE.Phy = UePhyUpdateConfig(UE.FAPIconfig, UE.Phy);

% UE.idxUE = idxUE;
% UE.Mac = Mac;
% UE.FAPIconfig = FAPIconfig;
% UE.Phy = Phy;

return

function Mac = UeMacUpdateConfig(SysPar, Mac, idxUE)

% Mac.Config.carrier = SysPar.carrier;
% Mac.Config.table = loadTable;

alloc = SysPar.SimCtrl.UE{idxUE}.tx.alloc;
Mac.tx.alloc = alloc;

idxPrach = 1;
idxPucch = 1;
idxPusch = 1;
idxSrs = 1;
pucch = {};
pusch = {};
srs = {};

if isfield(SysPar, 'prach')
    prach{1} = SysPar.prach{1};
    prach{1} = findPreambleTfLoc(prach{1}, ...
        Mac.Config.carrier, Mac.Config.table);
end

for idxAlloc = 1:length(alloc)
    allocType = alloc{idxAlloc}.type;
    idx = alloc{idxAlloc}.idx;
    switch allocType
        case 'prach'
            prach{idxPrach} = SysPar.prach{idx};
            prach{idxPrach} = findPreambleTfLoc(prach{idxPrach}, ...
                    Mac.Config.carrier, Mac.Config.table);
            prmbIdx =  prach{idxPrach}.prmbIdx;
            prmbIdx = mod(prmbIdx + (idxUE-1)*3, 64);
            prach{idxPrach}.prmbIdx = prmbIdx;
            idxPrach = idxPrach + 1;
        case 'pucch'
            pucch{idxPucch} = SysPar.pucch{idx};
            idxPucch = idxPucch + 1;
        case 'pusch'
            pusch{idxPusch} = SysPar.pusch{idx};
            pusch{idxPusch}.gNB_idxPusch = idx;
            idxPusch = idxPusch + 1;
        case 'srs'
            srs{idxSrs} = SysPar.srs{idx};
            idxSrs = idxSrs + 1;
        otherwise
            error('allocType is not supported...\n');
    end
end

if idxPrach > 1
    Mac.Config.prach = prach;
end

if idxPucch > 1
    Mac.Config.pucch = pucch;
end

if idxPusch > 1
    for idx = 1:(idxPusch-1)
        if ~isempty(Mac.Config.pusch)
            pusch{idx}.harqState = Mac.Config.pusch{idx}.harqState;
        else
            pusch{idx}.harqState = struct;
        end
    end
    Mac.Config.pusch = pusch;
end

if idxSrs > 1
    Mac.Config.srs = srs;
end

alloc = SysPar.SimCtrl.UE{idxUE}.rx.alloc;
Mac.rx.alloc = alloc;

idxSsb = 1;
idxPdcch = 1;
idxPdsch = 1;
idxCsirs = 1;
ssb = {};
pdcch = {};
pdsch = {};
csirs = {};

if isfield(SysPar, 'ssb')
    ssb = SysPar.ssb;
    caseType = ssb.caseType;
    fc = Mac.Config.carrier.carrierFreq;
    duplex = Mac.Config.carrier.duplex;
    [symIdxInFrame] = findSsbSymIdx(fc, duplex, caseType, ssb);
    ssb.L_max = length(symIdxInFrame);
    ssb.block_idx = 0;
    global SimCtrl
    if SimCtrl.genTV.forceSlotIdxFlag
        block_idx = find(ssb.ssbBitMap, 1)-1;
        ssb.block_idx = floor(block_idx/2)*2;
    end
    Mac.Config.ssb = ssb;
end

for idxAlloc = 1:length(alloc)
    allocType = alloc{idxAlloc}.type;
    idx = alloc{idxAlloc}.idx;
    switch allocType
        case 'ssb'
            %             
        case 'pdcch'
            pdcch{idxPdcch} = SysPar.pdcch{idx};
            idxPdcch = idxPdcch + 1;
        case 'pdsch'
            pdsch{idxPusch} = SysPar.pdsch{idx};
            pusch{idxPusch}.UE_idxPdsch = idx;
            idxPdsch = idxPdsch + 1;
        case 'csirs'
            csirs{idxCsirs} = SysPar.csirs{idx};
            idxCsirs = idxCsirs + 1;
        otherwise
            error('allocType is not supported...\n');
    end
end

if idxPdcch > 1
    Mac.Config.pdcch = pdcch;
end

if idxPdsch > 1
    Mac.Config.pdsch = pdsch;
end

if idxCsirs > 1
    Mac.Config.csirs = csirs;
end

% Mac.Config.carrier.idxFrame = Mac.Config.carrier.SFN_start;
% Mac.Config.carrier.idxSubframe = 0;
% Mac.Config.carrier.idxSlot = 0;
% Mac.Config.carrier.idxSlotInFrame = 0;


return;


function FAPIconfig = UeMacUpdateConfigToPhy(Mac, FAPIconfig)

MacConfig = Mac.Config;

% FAPIconfig.CarrierConfig = UeGenMac2PhyConfig(MacConfig, 'carrier');
% FAPIconfig.CellConfig = UeGenMac2PhyConfig(MacConfig, 'cell');
% FAPIconfig.TddConfig = UeGenMac2PhyConfig(MacConfig, 'tdd');
% 
% FAPIconfig.SsbConfig = UeGenMac2PhyConfig(MacConfig, 'ssb');
% FAPIconfig.PdcchConfig = UeGenMac2PhyConfig(MacConfig, 'pdcch');
% FAPIconfig.PdschConfig = UeGenMac2PhyConfig(MacConfig, 'pdsch');
% FAPIconfig.CsirsConfig = UeGenMac2PhyConfig(MacConfig, 'csirs');

FAPIconfig.PrachConfig = UeGenMac2PhyConfig(MacConfig, 'prach');
% FAPIconfig.PucchConfig = UeGenMac2PhyConfig(MacConfig, 'pucch');
% FAPIconfig.PuschConfig = UeGenMac2PhyConfig(MacConfig, 'pusch');
% FAPIconfig.SrsConfig = UeGenMac2PhyConfig(MacConfig, 'srs');

return;


function config = UeGenMac2PhyConfig(MacConfig,configType)

switch configType
    case 'carrier'
        % Table 3-21
        carrier = MacConfig.carrier;
        config.dlBandwidth          = carrier.BW;
        config.dlFrequency          = carrier.carrierFreq*1e6;
        config.dlk0                 = carrier.k0_mu;
        config.dlGridSize           = carrier.N_grid_size_mu;
        config.numTxAnt             = carrier.Nant_UE;
        config.uplinkBandwidth      = carrier.BW;
        config.uplinkFrequency      = carrier.carrierFreq*1e6;
        config.ulk0                 = carrier.k0_mu;
        config.ulGridSize           = carrier.N_grid_size_mu;
        config.numRxAnt             = carrier.Nant_UE;
        config.FrequencyShift7p5KHz = carrier.freqShift7p5KHz;
        config.mu                   = carrier.mu;         % added
        config.dmrsTypeAPos         = carrier.dmrsTypeAPos;        
    case 'cell'
        % Table 3-22
        carrier = MacConfig.carrier;
        config.phyCellId            = carrier.N_ID_CELL;
        config.FrameDuplexType      = carrier.duplex;
        config.SFN_start            = carrier.SFN_start; % added
    case 'tdd'
        % Table 3-26
        carrier = MacConfig.carrier;
        [~, idx] = ismember(carrier.TddPeriod, [0.5, 0.625, 1, 1.25, 2, 2.5, 5, 10]);
        if idx
            config.TddPeriod = idx-1;
        else
            error('TddPeriod is not supported...');
        end
        config.SlotConfig = carrier.SlotConfig;
    case 'ssb'
        % Table 3-23
        ssb = MacConfig.ssb;
        carrier = MacConfig.carrier;
        config.betaPss = ssb.betaPss;        % FAPI Pss to Sss scaler: 0=0 dB, 1=3 dB
        config.BchPayload = 1;               % hardcode 1: PHY generates the timing PBCH bits
        config.ScsCommon = carrier.mu;       % numerology for SSB and RA
        % Table 3-24
        config.SsbOffsetPointA = ssb.SsbOffsetPointA;
        [~, idx] = ismember(ssb.periodFrame*10, [5, 10, 20, 40, 80, 160]);
        if idx
            config.SsbPeriod = idx-1;
        else
            error('SsbPeriod is not supported...');
        end
        config.ssbSubcarrierOffset = ssb.ssbSubcarrierOffset;
        config.MIB = 0;         % not used
        config.SsbMask(1) = 0;  % not used;
        config.SsbMask(2) = 0;  % not used
        config.ssPbchMultipleCarriersInABand = 0; % not used
        config.multipleCellsSsPbchInACarrier = 0; % not used
        config.L_max = ssb.L_max; % added
        config.n_hf = ssb.n_hf;   % added
    case 'pdcch'
        config = [];
    case 'pdsch'
        config = [];
    case 'csirs'
        config = [];        
    case 'prach'
        prach = MacConfig.prach{1};
        carrier = MacConfig.carrier;
        switch prach.preambleFormat
            case {'0', '1', '2', '3'}
                config.prachSequenceLength = 0;
            otherwise
                config.prachSequenceLength = 1;
        end
        config.prachSubCSpacing = carrier.mu;
        config.restrictedSetConfig = prach.restrictedSet;
        config.numPrachFdOccasions = prach.msg1_FDM;
        config.prachConfigIndex = prach.configurationIndex;
        config.prachRootSequenceIndex = prach.rootSequenceIndex;
        config.numRootSequences = 0; % not used
        config.k1 = prach.k1;
        config.prachZeroCorrConf = prach.zeroCorrelationZone;
        config.numUnusedRootSequences = 0; % not used
        config.SsbPerRach = 0; % not used
        config.prachMultipleCarriersInABand = 0; % not used
        config.n_RA_start = prach.n_RA_start; % added
        config.ssbIdx = prach.ssbIdx; % added
    case 'pucch'
        config = [];
    case 'pusch'
        config = [];
    case 'srs'
        config = [];
    otherwise

end

return


function Phy = UePhyUpdateConfig(FAPIconfig, Phy)

Phy.Config.table = loadTable;

CarrierConfig = FAPIconfig.CarrierConfig;
CellConfig = FAPIconfig.CellConfig;
TddConfig = FAPIconfig.TddConfig;
SsbConfig = FAPIconfig.SsbConfig;
PdcchConfig = FAPIconfig.PdcchConfig;
PdschConfig = FAPIconfig.PdschConfig;
CsirsConfig = FAPIconfig.CsirsConfig;
PrachConfig = FAPIconfig.PrachConfig;

carrier.T_c = 1/(480e3*4096);       % NR basic sampling interval
carrier.T_s = 1/(15e3*2048);        % LTE 20M sampling interval
carrier.k_const = 64;               % T_s/T_c
carrier.N_sc_RB = 12;               % number of subcarrier per RB
carrier.N_subframe = 10;            % number of subframe per frame
carrier.T_subframe = 1e-3;          % duration of a subframe

carrier.carrierFreq = CarrierConfig.dlFrequency/1e6;          % carrier freq (GHz)
if carrier.carrierFreq <= 6
    carrier.FR = 1;                     % FR = 1(sub-6G), 2(mmW)
else
    carrier.FR = 2;
end

carrier.duplex = CellConfig.FrameDuplexType;             % 'TDD', 'FDD'
carrier.CpType = 0;                 % hardcode 'normal'
carrier.mu = CarrierConfig.mu;      % 0, 1, 2, 3, 4
carrier.N_grid_start_mu = 0;        % hardcode grid start in PRB
carrier.N_grid_size_mu = CarrierConfig.dlGridSize;       % grid size in PRB
carrier.BW = CarrierConfig.dlBandwidth;                    % BW in MHz
carrier.N_BWP_start = 0;            % hardcode inital UL BW part in PRB
carrier.N_ID_CELL = CellConfig.phyCellId;             % Cell ID
carrier.SFN_start = CellConfig.SFN_start;
carrier.numTxAnt = CarrierConfig.numTxAnt;
carrier.numRxAnt = CarrierConfig.numRxAnt;
carrier.dmrsTypeAPos = CarrierConfig.dmrsTypeAPos;

TddPeriodVector = [0.5, 0.625, 1, 1.25, 2, 2.5, 5, 10];
idx = TddConfig.TddPeriod + 1;
carrier.TddPeriod = TddPeriodVector(idx);
carrier.SlotConfig =TddConfig.SlotConfig;
carrier = updateCarrier(carrier);

prach.configurationIndex = PrachConfig.prachConfigIndex;
prach.msg1_FDM = PrachConfig.numPrachFdOccasions;
prach.n_RA_start = PrachConfig.n_RA_start;
prach.ssbIdx = PrachConfig.ssbIdx;
prach.zeroCorrelationZone = PrachConfig.prachZeroCorrConf;
prach.restrictedSet = PrachConfig.restrictedSetConfig;

prach = findPreambleTfLoc(prach, carrier, Phy.Config.table);
prach.prachRootSequenceIndex = PrachConfig.prachRootSequenceIndex;

% carrier.idxFrame = carrier.SFN_start;
% carrier.idxSubframe = 0;
% carrier.idxSlot = 0;
% carrier.idxSlotInFrame = 0;

periodFrameVector = [5, 10, 20, 40, 80, 160];
idx = SsbConfig.SsbPeriod+1;
ssb.periodFrame = periodFrameVector(idx);
ssb.betaPss = SsbConfig.betaPss;
ssb.ssbSubcarrierOffset = SsbConfig.ssbSubcarrierOffset;
ssb.SsbOffsetPointA = SsbConfig.SsbOffsetPointA;
ssb.L_max = SsbConfig.L_max;
ssb.n_hf = SsbConfig.n_hf;

pdcch = [];
pdsch = [];
csirs = [];
pucch = [];
pusch = [];
srs = [];

% pusch.harqProcess = cell(16,1);

% Phy.Config.carrier = carrier;
% Phy.Config.ssb = ssb;
% Phy.Config.pdcch = pdcch;
% Phy.Config.pdsch = pdsch;
% Phy.Config.csirs = csirs;
Phy.Config.prach = prach;
% Phy.Config.pucch = pucch;
% Phy.Config.pusch = pusch;
% Phy.Config.srs = srs;

% Phy.Config.preambleBufferValid = 0;

return;
