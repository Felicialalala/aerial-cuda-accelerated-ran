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

function gNB = updategNB(SysPar, gNB)
% function UE = initgNB(SysPar)
%
% This function initializes an gNB. It prepares paramaters, load tables,
% pre-generate some variables for each allocation based on the config
% from SysPar.
%
% Input:    SysPar: structure with all configurations
%
% Output:   gNB: structure for a single UE
%

Mac = gNbMacUpdateConfig(SysPar, gNB.Mac);
% FAPIconfig = gNbMacSendConfigToPhy(Mac);
% Phy = gNbPhyInitConfig(FAPIconfig, SysPar);

gNB.Mac = Mac;
% gNB.FAPIconfig = FAPIconfig;
% gNB.Phy = Phy;
% gNB.Phy.Chan_DL = SysPar.Chan_DL;
% gNB.Phy.Chan_UL = SysPar.Chan_UL;

return;


function Mac = gNbMacUpdateConfig(SysPar, Mac)

% Mac.Config.carrier = SysPar.carrier;
% Mac.Config.table = loadTable;

testAlloc = SysPar.testAlloc;

if testAlloc.ssb
    Mac.Config.ssb = SysPar.ssb;
    ssb = Mac.Config.ssb;
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

if testAlloc.pdcch
    Mac.Config.pdcch = SysPar.pdcch;
end

if testAlloc.pdsch
    Mac.Config.pdsch = SysPar.pdsch;
end

if testAlloc.csirs
    Mac.Config.csirs = SysPar.csirs;
end

if testAlloc.prach
    Mac.Config.prach = [];
    prach = SysPar.prach;
    for idxPrach = 1:length(prach)
        Mac.Config.prach{idxPrach} = findPreambleTfLoc(prach{idxPrach}, ...
            Mac.Config.carrier, Mac.Config.table);
    end
end

if testAlloc.pucch
    Mac.Config.pucch = SysPar.pucch;
end

if testAlloc.pusch
    Mac.Config.pusch = SysPar.pusch;
end

if testAlloc.srs
    Mac.Config.srs = SysPar.srs;
end

if testAlloc.bfw
    Mac.Config.bfw   = SysPar.bfw;
end

Mac.tx.alloc = SysPar.SimCtrl.gNB.tx.alloc;
Mac.rx.alloc = SysPar.SimCtrl.gNB.rx.alloc;

% Mac.Config.carrier.idxFrame = Mac.Config.carrier.SFN_start;
% Mac.Config.carrier.idxSubframe = 0;
% Mac.Config.carrier.idxSlot = 0;
% Mac.Config.carrier.idxSlotInFrame = 0;
% 

% 
% N_UE = SysPar.SimCtrl.N_UE;

return;


function FAPIconfig = gNbMacSendConfigToPhy(Mac)

MacConfig = Mac.Config;

FAPIconfig.CarrierConfig = genMac2PhyConfig(MacConfig, 'carrier');
FAPIconfig.CellConfig = genMac2PhyConfig(MacConfig, 'cell');
FAPIconfig.TddConfig = genMac2PhyConfig(MacConfig, 'tdd');

FAPIconfig.SsbConfig = genMac2PhyConfig(MacConfig, 'ssb');
FAPIconfig.PdcchConfig = genMac2PhyConfig(MacConfig, 'pdcch');
FAPIconfig.PdschConfig = genMac2PhyConfig(MacConfig, 'pdsch');
FAPIconfig.CsirsConfig = genMac2PhyConfig(MacConfig, 'csirs');

FAPIconfig.PrachConfig = genMac2PhyConfig(MacConfig, 'prach');
FAPIconfig.PucchConfig = genMac2PhyConfig(MacConfig, 'pucch');
FAPIconfig.PuschConfig = genMac2PhyConfig(MacConfig, 'pusch');
FAPIconfig.SrsConfig = genMac2PhyConfig(MacConfig, 'srs');

return;


function config = genMac2PhyConfig(MacConfig,configType)

switch configType
    case 'carrier'
        % Table 3-21
        carrier = MacConfig.carrier;
        config.dlBandwidth          = carrier.BW;
        config.dlFrequency          = carrier.carrierFreq*1e6;
        config.dlk0                 = carrier.k0_mu;
        config.dlGridSize           = carrier.N_grid_size_mu;
        config.numTxAnt             = carrier.Nant_gNB;
        config.uplinkBandwidth      = carrier.BW;
        config.uplinkFrequency      = carrier.carrierFreq*1e6;
        config.ulk0                 = carrier.k0_mu;
        config.ulGridSize           = carrier.N_grid_size_mu;
        config.numRxAnt             = carrier.Nant_gNB;
        config.FrequencyShift7p5KHz = carrier.freqShift7p5KHz;
        config.dmrsTypeAPos         = carrier.dmrsTypeAPos;
        config.numTxPort            = carrier.N_FhPort_DL;
        config.numRxPort            = carrier.N_FhPort_UL;
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
        nPrach = length(MacConfig.prach);
        for idxPrach = 1:nPrach
            prach = MacConfig.prach{idxPrach};
            carrier = MacConfig.carrier;
            switch prach.preambleFormat
                case {'0', '1', '2', '3'}
                    config(idxPrach).prachSequenceLength = 0;
                otherwise
                    config(idxPrach).prachSequenceLength = 1;
            end
            config(idxPrach).prachSubCSpacing = carrier.mu;
            config(idxPrach).restrictedSetConfig = prach.restrictedSet;
            config(idxPrach).numPrachFdOccasions = prach.msg1_FDM;
            config(idxPrach).prachConfigIndex = prach.configurationIndex;
            config(idxPrach).prachRootSequenceIndex = prach.rootSequenceIndex;
            config(idxPrach).numRootSequences = 0; % not used
            config(idxPrach).k1 = prach.k1;
            config(idxPrach).prachZeroCorrConf = prach.zeroCorrelationZone;
            config(idxPrach).numUnusedRootSequences = 0; % not used
            config(idxPrach).SsbPerRach = 0; % not used
            config(idxPrach).prachMultipleCarriersInABand = 0; % not used
            config(idxPrach).n_RA_start = prach.n_RA_start; % added
            config(idxPrach).ssbIdx = prach.ssbIdx; % added
            config(idxPrach).force_thr0 = prach.force_thr0; % added
        end
    case 'pucch'
        config = [];
    case 'pusch'
        config = [];
    case 'srs'
        config = [];
    case 'bfw'
        config = [];
    otherwise

end

return


function Phy = gNbPhyInitConfig(FAPIconfig, SysPar)

global SimCtrl;

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
carrier.CpType = 0;           % hardcode 'normal'
carrier.mu = SsbConfig.ScsCommon;   % 0, 1, 2, 3, 4
carrier.N_grid_start_mu = 0;        % hardcode grid start in PRB
carrier.N_grid_size_mu = CarrierConfig.dlGridSize;       % grid size in PRB
carrier.BW = CarrierConfig.dlBandwidth;                    % BW in MHz
carrier.N_BWP_start = 0;            % hardcode inital UL BW part in PRB
carrier.N_ID_CELL = CellConfig.phyCellId;             % Cell ID
carrier.SFN_start = CellConfig.SFN_start;
carrier.numTxAnt = CarrierConfig.numTxAnt;
carrier.numRxAnt = CarrierConfig.numRxAnt;
carrier.numTxPort = CarrierConfig.numTxPort;
carrier.numRxPort = CarrierConfig.numRxPort;
carrier.dmrsTypeAPos = CarrierConfig.dmrsTypeAPos;

TddPeriodVector = [0.5, 0.625, 1, 1.25, 2, 2.5, 5, 10];
idx = TddConfig.TddPeriod + 1;
carrier.TddPeriod = TddPeriodVector(idx);
carrier.SlotConfig =TddConfig.SlotConfig;
carrier = updateCarrier(carrier);

carrier.idxFrame = carrier.SFN_start;
carrier.idxSubframe = 0;
carrier.idxSlot = 0;
carrier.idxSlotInFrame = 0;

periodFrameVector = [5, 10, 20, 40, 80, 160];
idx = SsbConfig.SsbPeriod+1;
ssb.periodFrame = periodFrameVector(idx);
ssb.betaPss = SsbConfig.betaPss;
ssb.ssbSubcarrierOffset = SsbConfig.ssbSubcarrierOffset;
ssb.SsbOffsetPointA = SsbConfig.SsbOffsetPointA;
ssb.L_max = SsbConfig.L_max;
ssb.n_hf = SsbConfig.n_hf;

nPrach = length(PrachConfig);
for idxPrach = 1:nPrach
    thisPrach.configurationIndex = PrachConfig(idxPrach).prachConfigIndex;
    thisPrach.msg1_FDM = PrachConfig(idxPrach).numPrachFdOccasions;
    thisPrach.n_RA_start = PrachConfig(idxPrach).n_RA_start;
    thisPrach.ssbIdx = PrachConfig(idxPrach).ssbIdx;
    thisPrach.zeroCorrelationZone = PrachConfig(idxPrach).prachZeroCorrConf;
    thisPrach.restrictedSet = PrachConfig(idxPrach).restrictedSetConfig;    
    thisPrach = findPreambleTfLoc(thisPrach, carrier, Phy.Config.table); 
    thisPrach.prachRootSequenceIndex = PrachConfig(idxPrach).prachRootSequenceIndex;
    thisPrach.force_thr0 = PrachConfig(idxPrach).force_thr0;
    prach(idxPrach) = thisPrach;
end

pdcch = [];
pdsch = [];
csirs = [];
pucch = [];
pusch = [];
srs = [];
bfw = [];

% compute MMSE ChEst filters
Phy.Config.table.W_lower  = derive_legacy_lower(10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W_middle = derive_legacy_middle(10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W_upper  = derive_legacy_upper(10^(-3),15e3*2^(carrier.mu));

Phy.Config.table.W4_upper  = derive_upper_filter(10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W4_middle = derive_middle_filter(10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W4_lower  = derive_lower_filter(10^(-3),15e3*2^(carrier.mu));

Phy.Config.table.W3 = derive_small_filter(3,10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W2 = derive_small_filter(2,10^(-3),15e3*2^(carrier.mu));
Phy.Config.table.W1 = derive_small_filter(1,10^(-3),15e3*2^(carrier.mu));

% compute ChEst shift sequences
tau    = (2.0*10^(-6) - 2*10^(-6)/10) / 2; % delay shift

% compute SRS ChEst filters
Phy.Config.table.W_comb2_nPorts1_wide = build_SRS_filter(2, 1, 2.0*10^(-6), 0.4);
Phy.Config.table.W_comb2_nPorts2_wide = build_SRS_filter(2, 2, 2.0*10^(-6), 0.4);
Phy.Config.table.W_comb2_nPorts4_wide = build_SRS_filter(2, 4, 2.0*10^(-6), 0.4);
Phy.Config.table.W_comb4_nPorts1_wide = build_SRS_filter(4, 1, 2.0*10^(-6), 0.4);
Phy.Config.table.W_comb4_nPorts2_wide = build_SRS_filter(4, 2, 2.0*10^(-6), 0.4);
Phy.Config.table.W_comb4_nPorts4_wide = build_SRS_filter(4, 4, 1.5*10^(-6), 0.4);

[W,noisEstDebias] = build_SRS_filter(2, 1, 1.0*10^(-6), 0);
Phy.Config.table.W_comb2_nPorts1_narrow    = W;
Phy.Config.table.noisEstDebias_comb2_nPorts1 = noisEstDebias;

[W,noisEstDebias] = build_SRS_filter(2, 2, 1.0*10^(-6), 0);
Phy.Config.table.W_comb2_nPorts2_narrow    = W;
Phy.Config.table.noisEstDebias_comb2_nPorts2 = noisEstDebias;

[W,noisEstDebias] = build_SRS_filter(2, 4, 1.0*10^(-6), 0);
Phy.Config.table.W_comb2_nPorts4_narrow    = W;
Phy.Config.table.noisEstDebias_comb2_nPorts4 = noisEstDebias;

[W,noisEstDebias] = build_SRS_filter(4, 1, 1.0*10^(-6), 0);
Phy.Config.table.W_comb4_nPorts1_narrow    = W;
Phy.Config.table.noisEstDebias_comb4_nPorts1 = noisEstDebias;

[W,noisEstDebias] = build_SRS_filter(4, 2, 1.0*10^(-6), 0);
Phy.Config.table.W_comb4_nPorts2_narrow    = W;
Phy.Config.table.noisEstDebias_comb4_nPorts2 = noisEstDebias;

[W,noisEstDebias] = build_SRS_filter(4, 4, 0.25*10^(-6), 0);
Phy.Config.table.W_comb4_nPorts4_narrow    = W;
Phy.Config.table.noisEstDebias_comb4_nPorts4 = noisEstDebias;

global SimCtrl
if SimCtrl.delaySpread > 0
    tau = (SimCtrl.delaySpread-SimCtrl.delaySpread/8)/2;
end

df     = 15e3*2^(carrier.mu);

f_dmrs = 0 : 2 : (8*12 - 1);
f_dmrs = df * f_dmrs';

f_data = -1 : (8*12 - 1);
f_data = df * f_data';

shiftSeq  = exp(2*pi*1i*tau*f_dmrs);
shiftSeq  = fp16nv(real(shiftSeq), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(shiftSeq), SimCtrl.fp16AlgoSel);
shiftSeq4 = shiftSeq(1 : 4*6);

unShiftSeq  = exp(-2*pi*1i*tau*f_data);
unShiftSeq  = fp16nv(real(unShiftSeq), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(unShiftSeq), SimCtrl.fp16AlgoSel);
unShiftSeq4 = unShiftSeq(1 : (12*4 + 1));

Phy.Config.table.shiftSeq    = shiftSeq;
Phy.Config.table.shiftSeq4   = shiftSeq4;
Phy.Config.table.unShiftSeq  = unShiftSeq;
Phy.Config.table.unShiftSeq4 = unShiftSeq4;

% Add harq buffers
pusch.harqState = struct;

Phy.Config.carrier = carrier;
Phy.Config.ssb = ssb;
Phy.Config.pdcch = pdcch;
Phy.Config.pdsch = pdsch;
Phy.Config.csirs = csirs;
Phy.Config.prach = prach;
Phy.Config.pucch = pucch;
Phy.Config.pusch = pusch;
Phy.Config.srs = srs;
Phy.Config.bfw = bfw;

Phy.Config.preambleBufferValid = zeros(1, nPrach);

% Init srs chEst buffer
chan_BF          = SysPar.chan_BF;
N_SRS_CHEST_BUFF = SysPar.SimCtrl.bf.N_SRS_CHEST_BUFF;
srsChEstBuff = [];
startPrbGrps = [];
for idxSrsBuff = 0 : (N_SRS_CHEST_BUFF - 1)
    srsChEstBuffPrms              = SysPar.srsChEstBuff{idxSrsBuff + 1};
    srsChEstBuff{idxSrsBuff + 1}  = initSrsBuff(chan_BF{idxSrsBuff + 1}, srsChEstBuffPrms);
    startPrbGrps(idxSrsBuff + 1)  = srsChEstBuffPrms.startPrbGrp;
end
srsChEstDatabase = [];
srsChEstDatabase.srsChEstBuff = srsChEstBuff;
srsChEstDatabase.startPrbGrps = startPrbGrps;
Phy.srsChEstDatabase          = srsChEstDatabase;

return;
