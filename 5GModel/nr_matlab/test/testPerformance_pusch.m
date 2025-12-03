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

function [nPerf, nFail, CBER, TBER, CFOERR, TOERR, SNR] = testPerformance_pusch(caseSet, SNRoffset, Nframe, puschTestMode, batchsimCfg, relNum)

if nargin == 0
    caseSet = 'full';
    SNRoffset = 0;
    Nframe = 1;
    puschTestMode = 0;
    batchsimCfg.batchsimMode = 0; % 0: disable batchsimMode, 1: phase2 perf study, 2: performance match test for 5GModel and cuPHY
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 1
    SNRoffset = 0;
    Nframe = 1;
    puschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 2
    Nframe = 1;
    puschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 3
    puschTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 4
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 5
    relNum = 10000;
end

selected_TC = [7001:7800];
compact_TC = [7001:7800];
full_TC = [7001:7800];

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch puschTestMode
        case 0
            TcToTest = [7001:7030]; % PUSCH data
        case 1
            TcToTest = [7051:7054]; % PUSCH UCI
        case 2
            TcToTest = [7055:7060]; % PUSCH transform precoding
        case 3
            TcToTest = [7101]; % 0.001% BLER
        case 7
            TcToTest = [7701:7748]; % mMIMO SRS/BF/PUSCH
        case 8
            TcToTest = [7801:7899]; % Channel model
        case 9
            TcToTest = [7901:7907]; % PUSCH self-defined
        case 10
            TcToTest = [7911:7918]; % UL measurement
        case 11
            TcToTest = [7950:7983]; % AWGN, BLER for all MCS 
        case 12
            TcToTest = [7984:7991]; % fading, BLER for all MCS
        otherwise
            error('puschTestMode is not supported...\n');
    end
end
if isfield(batchsimCfg, 'caseSet')
    if ~isempty(batchsimCfg.caseSet)
        TcToTest_superset_from_batchsimCfg = batchsimCfg.caseSet;
    else
        TcToTest_superset_from_batchsimCfg = 1:1e5;
    end
else
    TcToTest_superset_from_batchsimCfg = 1:1e5;
end

if isfield(batchsimCfg, 'test_version')
    test_version = batchsimCfg.test_version; % '38.141-1.v15.14'
else
    test_version = '38.104.v16.4';
end

if strcmp(test_version, '38.104.v16.4')

    CFG = {...
    % TS38-104 V16.4 Table 8.2.1.2-7 100MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7001,  'G-FR1-A3-14', 'TDLB100-400-Low',   2,  -2.8, 200,   0e-6;
      7002,  'G-FR1-A4-14', 'TDLC300-100-Low',   2,  10.2, 200,   0e-6;
      7003,  'G-FR1-A5-14',   'TDLA30-10-Low',   2,  13.0, 200,   0e-6;
      7004,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -5.8, 200,   0e-6;
      7005,  'G-FR1-A4-14', 'TDLC300-100-Low',   4,   6.5, 200,   0e-6;
      7006,  'G-FR1-A5-14',   'TDLA30-10-Low',   4,   9.0, 200,   0e-6;
      7007,  'G-FR1-A3-14', 'TDLB100-400-Low',   8,  -8.7, 200,   0e-6;
      7008,  'G-FR1-A4-14', 'TDLC300-100-Low',   8,   3.2, 200,   0e-6;
      7009,  'G-FR1-A5-14',   'TDLA30-10-Low',   8,   5.8, 200,   0e-6;
      7010,  'G-FR1-A3-28', 'TDLB100-400-Low',   2,   1.4, 200,   0e-6;
      7011,  'G-FR1-A4-28', 'TDLC300-100-Low',   2,  19.2, 200,   0e-6;
      7012,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -2.2, 200,   0e-6;
      7013,  'G-FR1-A4-28', 'TDLC300-100-Low',   4,  11.6, 200,   0e-6;
      7014,  'G-FR1-A3-28', 'TDLB100-400-Low',   8,  -5.2, 200,   0e-6;
      7015,  'G-FR1-A4-28', 'TDLC300-100-Low',   8,   7.1, 200,   0e-6;
    % TS38-104 Table 8.2.1.2-3 20MHz, 15kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7016,  'G-FR1-A3-10', 'TDLB100-400-Low',   2,  -2.1, 200,   0e-6;
      7017,  'G-FR1-A4-10', 'TDLC300-100-Low',   2,  10.0, 200,   0e-6;
      7018,  'G-FR1-A5-10',   'TDLA30-10-Low',   2,  12.4, 200,   0e-6;
      7019,  'G-FR1-A3-10', 'TDLB100-400-Low',   4,  -5.5, 200,   0e-6;
      7020,  'G-FR1-A4-10', 'TDLC300-100-Low',   4,   6.2, 200,   0e-6;
      7021,  'G-FR1-A5-10',   'TDLA30-10-Low',   4,   8.6, 200,   0e-6;
      7022,  'G-FR1-A3-10', 'TDLB100-400-Low',   8,  -8.5, 200,   0e-6;
      7023,  'G-FR1-A4-10', 'TDLC300-100-Low',   8,   3.0, 200,   0e-6;
      7024,  'G-FR1-A5-10',   'TDLA30-10-Low',   8,   5.5, 200,   0e-6;
      7025,  'G-FR1-A3-24', 'TDLB100-400-Low',   2,   2.1, 200,   0e-6;
      7026,  'G-FR1-A4-24', 'TDLC300-100-Low',   2,  18.3, 200,   0e-6;
      7027,  'G-FR1-A3-24', 'TDLB100-400-Low',   4,  -1.8, 200,   0e-6;
      7028,  'G-FR1-A4-24', 'TDLC300-100-Low',   4,  11.1, 200,   0e-6;
      7029,  'G-FR1-A3-24', 'TDLB100-400-Low',   8,  -5.3, 200,   0e-6;
      7030,  'G-FR1-A4-24', 'TDLC300-100-Low',   8,   6.9, 200,   0e-6;
    % TS38-104 Table 8.2.3.2-1 TypeA, CSI-P1, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7051,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   5.4, 200,   0e-6; % case1: 5+2
      7052,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   4.3, 200,   0e-6; % case2: 20+20
    % TS38-104 Table 8.2.3.2-3 TypeA, CSI-P2, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7053,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,  -0.2, 200,   0e-6; % case1: 5+2
      7054,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   2.4, 200,   0e-6; % case2: 20+20
    % TS38-104 Table 8.2.2.2-2 TypeA, 10MHz, 30kHz SCS, transform precoding
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7055,  'G-FR1-A3-32', 'TDLB100-400-Low',   2,  -2.5, 200,   0e-6;
      7056,  'G-FR1-A3-32', 'TDLB100-400-Low',   4,  -5.7, 200,   0e-6;
      7057,  'G-FR1-A3-32', 'TDLB100-400-Low',   8,  -8.4, 200,   0e-6;
    % TS38-104 Table 8.2.2.2-1 TypeA, 5MHz, 15kHz SCS, transform precoding
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7058,  'G-FR1-A3-31', 'TDLB100-400-Low',   2,  -2.4, 200,   0e-6;
      7059,  'G-FR1-A3-31', 'TDLB100-400-Low',   4,  -5.7, 200,   0e-6;
      7060,  'G-FR1-A3-31', 'TDLB100-400-Low',   8,  -8.5, 200,   0e-6;
    % TS38-104 (V16.7) Table 8.2.6.2-3 0.001% BLER, Type A, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7101,  'G-FR1-A3A-3',            'AWGN',   2,  -5.4, 200,   0e-6; %
    % Self defined test cases
    % TC for mMIMO SRS/BFW/PUSCH 
    % SU-MIMO/nl = 1 x 2
    % nSlotSrsUpdate = 20
      7701,  'G-FR1-A4-28',            'AWGN',   4,  -4.6,   0,   0e-6; % known AWGN channel
      7702,  'G-FR1-A4-28',            'AWGN',   4,  -3.2,   0,   0e-6; % estimated AWGN channel
      7703,  'G-FR1-A4-28',       'CDLA30-10',   4,  -6.4,   0,   0e-6; % known CDLA channel
      7704,  'G-FR1-A4-28',       'CDLA30-10',   4,  -4.8,   0,   0e-6; % estimated CDLA channel
      7705,  'G-FR1-A4-28',     'CDLB100-400',   4,   1.2,   0,   0e-6; % known CDLB channel
      7706,  'G-FR1-A4-28',     'CDLB100-400',   4,   1.6,   0,   0e-6; % estimated CDLB channel
      7707,  'G-FR1-A4-28',     'CDLC300-100',   4,  -6.0,   0,   0e-6; % known CDLC channel
      7708,  'G-FR1-A4-28',     'CDLC300-100',   4,  -5.0,   0,   0e-6; % estimated CDLC channel

    % nSlotSrsUpdate = 1
      7709,  'G-FR1-A4-28',            'AWGN',   4,  -4.6,   0,   0e-6; % known AWGN channel
      7710,  'G-FR1-A4-28',            'AWGN',   4,  -3.0,   0,   0e-6; % estimated AWGN channel
      7711,  'G-FR1-A4-28',       'CDLA30-10',   4,  -6.4,   0,   0e-6; % known CDLA channel
      7712,  'G-FR1-A4-28',       'CDLA30-10',   4,  -4.8,   0,   0e-6; % estimated CDLA channel
      7713,  'G-FR1-A4-28',     'CDLB100-400',   4,  -6.4,   0,   0e-6; % known CDLB channel
      7714,  'G-FR1-A4-28',     'CDLB100-400',   4,  -4.8,   0,   0e-6; % estimated CDLB channel
      7715,  'G-FR1-A4-28',     'CDLC300-100',   4,  -7.8,   0,   0e-6; % known CDLC channel
      7716,  'G-FR1-A4-28',     'CDLC300-100',   4,  -6.2,   0,   0e-6; % estimated CDLC channel
    % MU-MIMO/nl = 2 x 1
    % nSlotSrsUpdate = 20
      7717,  'G-FR1-A4-14',            'AWGN',   4,  -7.6,   0,   0e-6; % known AWGN channel
      7718,  'G-FR1-A4-14',            'AWGN',   4,  -5.4,   0,   0e-6; % estimated AWGN channel
      7719,  'G-FR1-A4-14',       'CDLA30-10',   4,  -9.4,   0,   0e-6; % known CDLA channel
      7720,  'G-FR1-A4-14',       'CDLA30-10',   4,  -6.6,   0,   0e-6; % estimated CDLA channel
      7721,  'G-FR1-A4-14',     'CDLB100-400',   4,  -3.0,   0,   0e-6; % known CDLB channel
      7722,  'G-FR1-A4-14',     'CDLB100-400',   4,  -1.6,   0,   0e-6; % estimated CDLB channel
      7723,  'G-FR1-A4-14',     'CDLC300-100',   4, -10.0,   0,   0e-6; % known CDLC channel
      7724,  'G-FR1-A4-14',     'CDLC300-100',   4,  -7.8,   0,   0e-6; % estimated CDLC channel
    % nSlotSrsUpdate = 1
      7725,  'G-FR1-A4-14',            'AWGN',   4,  -7.6,   0,   0e-6; % known AWGN channel
      7726,  'G-FR1-A4-14',            'AWGN',   4,  -5.4,   0,   0e-6; % estimated AWGN channel
      7727,  'G-FR1-A4-14',       'CDLA30-10',   4,  -9.4,   0,   0e-6; % known CDLA channel
      7728,  'G-FR1-A4-14',       'CDLA30-10',   4,  -6.6,   0,   0e-6; % estimated CDLA channel
      7729,  'G-FR1-A4-14',     'CDLB100-400',   4,  -9.8,   0,   0e-6; % known CDLB channel
      7730,  'G-FR1-A4-14',     'CDLB100-400',   4,  -7.4,   0,   0e-6; % estimated CDLB channel
      7731,  'G-FR1-A4-14',     'CDLC300-100',   4, -11.2,   0,   0e-6; % known CDLC channel
      7732,  'G-FR1-A4-14',     'CDLC300-100',   4,  -9.0,   0,   0e-6; % estimated CDLC channel
    % MU-MIMO/nl = 8 x 1
    % nSlotSrsUpdate = 20
      7733,  'G-FR1-B4-14',            'AWGN',   4,  -5.8,   0,   0e-6; % known AWGN channel
      7734,  'G-FR1-B4-14',            'AWGN',   4,  -4.0,   0,   0e-6; % estimated AWGN channel
      7735,  'G-FR1-B4-14',       'CDLA30-10',   4,  -5.8,   0,   0e-6; % known CDLA channel
      7736,  'G-FR1-B4-14',       'CDLA30-10',   4,  -3.4,   0,   0e-6; % estimated CDLA channel
      7737,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % known CDLB channel
      7738,  'G-FR1-B4-14',     'CDLB100-400',   4,    40,   0,   0e-6; % estimated CDLB channel
      7739,  'G-FR1-B4-14',     'CDLC300-100',   4,  -4.0,   0,   0e-6; % known CDLC channel
      7740,  'G-FR1-B4-14',     'CDLC300-100',   4,  -3.0,   0,   0e-6; % estimated CDLC channel
    % nSlotSrsUpdate = 1
      7741,  'G-FR1-B4-14',            'AWGN',   4,  -5.8,   0,   0e-6; % known AWGN channel
      7742,  'G-FR1-B4-14',            'AWGN',   4,  -4.0,   0,   0e-6; % estimated AWGN channel
      7743,  'G-FR1-B4-14',       'CDLA30-10',   4,  -5.4,   0,   0e-6; % known CDLA channel
      7744,  'G-FR1-B4-14',       'CDLA30-10',   4,  -3.6,   0,   0e-6; % estimated CDLA channel
      7745,  'G-FR1-B4-14',     'CDLB100-400',   4,  -5.8,   0,   0e-6; % known CDLB channel
      7746,  'G-FR1-B4-14',     'CDLB100-400',   4,  -3.4,   0,   0e-6; % estimated CDLB channel
      7747,  'G-FR1-B4-14',     'CDLC300-100',   4,  -8.4,   0,   0e-6; % known CDLC channel
      7748,  'G-FR1-B4-14',     'CDLC300-100',   4,  -5.6,   0,   0e-6; % estimated CDLC channel
      
    % TC for CDL/TDL channel model     
      7801,  'G-FR1-A3-14',             'CDL',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, isotropic
      7802,  'G-FR1-A3-14',             'CDL',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, isotropic
      7803,  'G-FR1-A3-14',             'CDL',   4,  -8.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, directional
      7804,  'G-FR1-A3-14',             'CDL',   4,  -8.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, directional
      7805,  'G-FR1-A3-28',             'CDL',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, isotropic
      7806,  'G-FR1-A3-28',             'CDL',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, isotropic    
      7807,  'G-FR1-A3-28',             'CDL',   4,  -4.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, directional
      7808,  'G-FR1-A3-28',             'CDL',   4,  -4.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, directional   
      7821,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, TDL
      7822,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, TDL
      7823,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, TDL
      7824,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, TDL
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7901,  'L1T1M27B273',            'AWGN',   1,    27, 100,   1e-6;   % 1 layer, 1 RX antenna
      7902,  'L1T1M27B273',   'TDLA30-10-Low',   2,    27, 100,   1e-6;   % 1 layer, 2 RX antennas
      7903,  'L1T1M27B273',   'TDLA30-10-Low',   4,    27, 100,   1e-6;   % 1 layer, 4 RX antennas
      7904,  'L1T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 1 layer, 8 RX antennas
      7905,  'L2T1M27B273',   'TDLA30-10-Low',   4,    27, 100,   1e-6;   % 2 layers, 4 RX antennas
      7906,  'L2T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 2 layers, 8 RX antennas
      7907,  'L4T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 4 layers, 8 RX antennas
    % UL measurement
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7911,  'L1T1M00B273',            'AWGN',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7912,  'L1T1M00B273',   'TDLA30-10-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7913,  'L1T1M00B273', 'TDLB100-400-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7914,  'L1T1M00B273', 'TDLC300-100-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7915,  'L2T1M00B273',            'AWGN',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7916,  'L2T1M00B273',   'TDLA30-10-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7917,  'L2T1M00B273', 'TDLB100-400-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7918,  'L2T1M00B273', 'TDLC300-100-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
    % BLER for all MCS levels TS 38.214 Table 5.1.3.1-2
    % BLER  30% with 0.25 dB resolution
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7950,  'L1T1M00B024',            'AWGN',   1, -4.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7951,  'L1T1M01B024',            'AWGN',   1,    -3,   0,   0e-6;   % 1 layer, 1 RX antenna
      7952,  'L1T1M02B024',            'AWGN',   1, -0.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7953,  'L1T1M03B024',            'AWGN',   1,  1.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7954,  'L1T1M04B024',            'AWGN',   1,     3,   0,   0e-6;   % 1 layer, 1 RX antenna
      7955,  'L1T1M05B024',            'AWGN',   1,     5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7956,  'L1T1M06B024',            'AWGN',   1,     6,   0,   0e-6;   % 1 layer, 1 RX antenna
      7957,  'L1T1M07B024',            'AWGN',   1,  6.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7958,  'L1T1M08B024',            'AWGN',   1,  7.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7959,  'L1T1M09B024',            'AWGN',   1,   8.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7960,  'L1T1M10B024',            'AWGN',   1,   9.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7961,  'L1T1M11B024',            'AWGN',   1, 10.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7962,  'L1T1M12B024',            'AWGN',   1, 11.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7963,  'L1T1M13B024',            'AWGN',   1,  12.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7964,  'L1T1M14B024',            'AWGN',   1,  13.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7965,  'L1T1M15B024',            'AWGN',   1,  14.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7966,  'L1T1M16B024',            'AWGN',   1,  15.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7967,  'L1T1M17B024',            'AWGN',   1, 16.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7968,  'L1T1M18B024',            'AWGN',   1,    17,   0,   0e-6;   % 1 layer, 1 RX antenna
      7969,  'L1T1M19B024',            'AWGN',   1,    18,   0,   0e-6;   % 1 layer, 1 RX antenna
      7970,  'L1T1M20B024',            'AWGN',   1,  19.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7971,  'L1T1M21B024',            'AWGN',   1, 19.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7972,  'L1T1M22B024',            'AWGN',   1,    21,   0,   0e-6;   % 1 layer, 1 RX antenna
      7973,  'L1T1M23B024',            'AWGN',   1,    22,   0,   0e-6;   % 1 layer, 1 RX antenna
      7974,  'L1T1M24B024',            'AWGN',   1,  22.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7975,  'L1T1M25B024',            'AWGN',   1, 24.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7976,  'L1T1M26B024',            'AWGN',   1, 24.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7977,  'L1T1M27B024',            'AWGN',   1, 26.25,   0,   0e-6;   % 1 layer, 1 RX antenna
    % BLER for all MCS levels TS 38.214 Table 6.1.4.1-2
      7978,  'L1T4M00B024',            'AWGN',   1,    -8,   0,   0e-6;   % 1 layer, 1 RX antenna
      7979,  'L1T4M01B024',            'AWGN',   1,  -7.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7980,  'L1T4M02B024',            'AWGN',   1,    -7,   0,   0e-6;   % 1 layer, 1 RX antenna
      7981,  'L1T4M03B024',            'AWGN',   1,  -6.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7982,  'L1T4M04B024',            'AWGN',   1,    -6,   0,   0e-6;   % 1 layer, 1 RX antenna
      7983,  'L1T4M05B024',            'AWGN',   1,  -5.5,   0,   0e-6;   % 1 layer, 1 RX antenna
    % BLER for selected MCS levels TS 38.214 Table 5.1.3.1-2 with fading channels
    % BLER 30% with 0.5 dB resolution
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7984,  'L1T1M00B024', 'TDLC300-100-Low',   4,  -8.5,   0,   0e-6;   % 1 layer, 4 RX antenna
      7985,  'L1T1M09B024', 'TDLC300-100-Low',   4,   3.5,   0,   0e-6;   % 1 layer, 4 RX antenna
      7986,  'L1T1M18B024', 'TDLC300-100-Low',   4,    12,   0,   0e-6;   % 1 layer, 4 RX antenna
      7987,  'L1T1M27B024', 'TDLC300-100-Low',   4, 22.25,   0,   0e-6;   % 1 layer, 4 RX antenna
      7988,  'L2T1M00B024', 'TDLC300-100-Low',   4,  -4.5,   0,   0e-6;   % 2 layer, 4 RX antenna
      7989,  'L2T1M09B024', 'TDLC300-100-Low',   4,     8,   0,   0e-6;   % 2 layer, 4 RX antenna
      7990,  'L2T1M18B024', 'TDLC300-100-Low',   4,    18,   0,   0e-6;   % 2 layer, 4 RX antenna
      7991,  'L2T1M27B024', 'TDLC300-100-Low',   4,    34,   0,   0e-6;   % 2 layer, 4 RX antenna
    % PUSCH with perfect channel estimation
      7992,  'L1T1M00B024', 'TDLC300-100-Low',   4,  -8.5,   0,   0e-6;   % 1 layer, 4 RX antenna
    % TC for additional FRC 
      7993,  'G-FR1-A1-5',             'AWGN',   1,     0,   0,   0e-6;   % 1 layer, 1 RX antenna
      7994,  'G-FR1-A2-5',             'AWGN',   1,     0,   0,   0e-6;   % 1 layer, 1 RX antenna
    };
elseif strcmp(test_version, '38.141-1') % SNR requirement based on 38.141-1
    CFG = {...
    % TS38-141 Table 8.2.1.5-7 100MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7001,  'G-FR1-A3-14', 'TDLB100-400-Low',   2,  -2.2,   0,   0e-6;
      7002,  'G-FR1-A4-14', 'TDLC300-100-Low',   2,  10.8,   0,   0e-6;
      7003,  'G-FR1-A5-14',   'TDLA30-10-Low',   2,  13.6,   0,   0e-6;
      7004,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -5.2,   0,   0e-6;
      7005,  'G-FR1-A4-14', 'TDLC300-100-Low',   4,   7.1,   0,   0e-6;
      7006,  'G-FR1-A5-14',   'TDLA30-10-Low',   4,   9.6,   0,   0e-6;
      7007,  'G-FR1-A3-14', 'TDLB100-400-Low',   8,  -8.1,   0,   0e-6;
      7008,  'G-FR1-A4-14', 'TDLC300-100-Low',   8,   3.8,   0,   0e-6;
      7009,  'G-FR1-A5-14',   'TDLA30-10-Low',   8,   6.4,   0,   0e-6;
      7010,  'G-FR1-A3-28', 'TDLB100-400-Low',   2,   2.2,   0,   0e-6;
      7011,  'G-FR1-A4-28', 'TDLC300-100-Low',   2,  20.0,   0,   0e-6;
      7012,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -1.4,   0,   0e-6;
      7013,  'G-FR1-A4-28', 'TDLC300-100-Low',   4,  12.4,   0,   0e-6;
      7014,  'G-FR1-A3-28', 'TDLB100-400-Low',   8,  -4.4,   0,   0e-6;
      7015,  'G-FR1-A4-28', 'TDLC300-100-Low',   8,   7.9,   0,   0e-6;
    % TS38-141 Table 8.2.1.5-3 20MHz, 15kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7016,  'G-FR1-A3-10', 'TDLB100-400-Low',   2,  -1.5,   0,   0e-6;
      7017,  'G-FR1-A4-10', 'TDLC300-100-Low',   2,  10.6,   0,   0e-6;
      7018,  'G-FR1-A5-10',   'TDLA30-10-Low',   2,  13.0,   0,   0e-6;
      7019,  'G-FR1-A3-10', 'TDLB100-400-Low',   4,  -4.9,   0,   0e-6;
      7020,  'G-FR1-A4-10', 'TDLC300-100-Low',   4,   6.8,   0,   0e-6;
      7021,  'G-FR1-A5-10',   'TDLA30-10-Low',   4,   9.2,   0,   0e-6;
      7022,  'G-FR1-A3-10', 'TDLB100-400-Low',   8,  -7.9,   0,   0e-6;
      7023,  'G-FR1-A4-10', 'TDLC300-100-Low',   8,   3.6,   0,   0e-6;
      7024,  'G-FR1-A5-10',   'TDLA30-10-Low',   8,   6.1,   0,   0e-6;
      7025,  'G-FR1-A3-24', 'TDLB100-400-Low',   2,   2.9,   0,   0e-6;
      7026,  'G-FR1-A4-24', 'TDLC300-100-Low',   2,  19.1,   0,   0e-6;
      7027,  'G-FR1-A3-24', 'TDLB100-400-Low',   4,  -1.0,   0,   0e-6;
      7028,  'G-FR1-A4-24', 'TDLC300-100-Low',   4,  11.9,   0,   0e-6;
      7029,  'G-FR1-A3-24', 'TDLB100-400-Low',   8,  -4.5,   0,   0e-6;
      7030,  'G-FR1-A4-24', 'TDLC300-100-Low',   8,   7.7,   0,   0e-6;
    % TS38-141 Table 8.2.3.5-1 TypeA, CSI-P1, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7051,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,     6,   0,   0e-6; % case1: 5+2
      7052,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   4.9,   0,   0e-6; % case2: 20+20
    % TS38-141 Table 8.2.3.5-3 TypeA, CSI-P2, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7053,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   0.4,   0,   0e-6; % case1: 5+2
      7054,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   3.0,   0,   0e-6; % case2: 20+20
    % TS38-141 Table 8.2.2.5-2 TypeA, 10MHz, 30kHz SCS, transform precoding
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7055,  'G-FR1-A3-32', 'TDLB100-400-Low',   2,  -1.9,   0,   0e-6;
      7056,  'G-FR1-A3-32', 'TDLB100-400-Low',   4,  -5.1,   0,   0e-6;
      7057,  'G-FR1-A3-32', 'TDLB100-400-Low',   8,  -7.8,   0,   0e-6;
    % TS38-141 Table 8.2.2.5-1 TypeA, 5MHz, 15kHz SCS, transform precoding
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7058,  'G-FR1-A3-31', 'TDLB100-400-Low',   2,  -1.8,   0,   0e-6;
      7059,  'G-FR1-A3-31', 'TDLB100-400-Low',   4,  -5.1,   0,   0e-6;
      7060,  'G-FR1-A3-31', 'TDLB100-400-Low',   8,  -7.9,   0,   0e-6;
    % TS38-141 (rel.16) Table 8.2.6.5-3 0.001% BLER, Type A, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7101,  'G-FR1-A3A-3',            'AWGN',   2,  -4.1,   0,   0e-6; %
    % Self defined test cases
    % TC for TDL/CDL channel model     
      7801,  'G-FR1-A3-14',             'CDL',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, isotropic
      7802,  'G-FR1-A3-14',             'CDL',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, isotropic
      7803,  'G-FR1-A3-14',             'CDL',   4,  -8.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, directional
      7804,  'G-FR1-A3-14',             'CDL',   4,  -8.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, directional
      7805,  'G-FR1-A3-28',             'CDL',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, isotropic
      7806,  'G-FR1-A3-28',             'CDL',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, isotropic    
      7807,  'G-FR1-A3-28',             'CDL',   4,  -4.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, directional
      7808,  'G-FR1-A3-28',             'CDL',   4,  -4.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, directional   
      7821,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, TDL
      7822,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -6.8, 200,   0e-6;   % 1 layer, 4 RX antenna 7004, TDL
      7823,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, TDL
      7824,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -3.2, 200,   0e-6;   % 2 layer, 4 RX antenna 7012, TDL
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7901,  'L1T1M27B273',            'AWGN',   1,    27, 100,   1e-6;   % 1 layer, 1 RX antenna
      7902,  'L1T1M27B273',   'TDLA30-10-Low',   2,    27, 100,   1e-6;   % 1 layer, 2 RX antennas
      7903,  'L1T1M27B273',   'TDLA30-10-Low',   4,    27, 100,   1e-6;   % 1 layer, 4 RX antennas
      7904,  'L1T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 1 layer, 8 RX antennas
      7905,  'L2T1M27B273',   'TDLA30-10-Low',   4,    27, 100,   1e-6;   % 2 layers, 4 RX antennas
      7906,  'L2T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 2 layers, 8 RX antennas
      7907,  'L4T1M27B273',   'TDLA30-10-Low',   8,    27, 100,   1e-6;   % 4 layers, 8 RX antennas
    % UL measurement
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7911,  'L1T1M00B273',            'AWGN',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7912,  'L1T1M00B273',   'TDLA30-10-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7913,  'L1T1M00B273', 'TDLB100-400-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7914,  'L1T1M00B273', 'TDLC300-100-Low',   4,     0, 100,   1e-6;   % 1 layer, 4 RX antenna
      7915,  'L2T1M00B273',            'AWGN',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7916,  'L2T1M00B273',   'TDLA30-10-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7917,  'L2T1M00B273', 'TDLB100-400-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
      7918,  'L2T1M00B273', 'TDLC300-100-Low',   4,     0, 100,   1e-6;   % 2 layer, 4 RX antenna
    % BLER for all MCS levels TS 38.214 Table 5.1.3.1-2
    % BLER  30% with 0.25 dB resolution
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7950,  'L1T1M00B024',            'AWGN',   1, -4.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7951,  'L1T1M01B024',            'AWGN',   1,    -3,   0,   0e-6;   % 1 layer, 1 RX antenna
      7952,  'L1T1M02B024',            'AWGN',   1, -0.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7953,  'L1T1M03B024',            'AWGN',   1,  1.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7954,  'L1T1M04B024',            'AWGN',   1,     3,   0,   0e-6;   % 1 layer, 1 RX antenna
      7955,  'L1T1M05B024',            'AWGN',   1,     5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7956,  'L1T1M06B024',            'AWGN',   1,     6,   0,   0e-6;   % 1 layer, 1 RX antenna
      7957,  'L1T1M07B024',            'AWGN',   1,  6.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7958,  'L1T1M08B024',            'AWGN',   1,  7.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7959,  'L1T1M09B024',            'AWGN',   1,   8.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7960,  'L1T1M10B024',            'AWGN',   1,   9.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7961,  'L1T1M11B024',            'AWGN',   1, 10.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7962,  'L1T1M12B024',            'AWGN',   1, 11.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7963,  'L1T1M13B024',            'AWGN',   1,  12.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7964,  'L1T1M14B024',            'AWGN',   1,  13.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7965,  'L1T1M15B024',            'AWGN',   1,  14.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7966,  'L1T1M16B024',            'AWGN',   1,  15.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7967,  'L1T1M17B024',            'AWGN',   1, 16.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7968,  'L1T1M18B024',            'AWGN',   1,    17,   0,   0e-6;   % 1 layer, 1 RX antenna
      7969,  'L1T1M19B024',            'AWGN',   1,    18,   0,   0e-6;   % 1 layer, 1 RX antenna
      7970,  'L1T1M20B024',            'AWGN',   1,  19.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7971,  'L1T1M21B024',            'AWGN',   1, 19.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7972,  'L1T1M22B024',            'AWGN',   1,    21,   0,   0e-6;   % 1 layer, 1 RX antenna
      7973,  'L1T1M23B024',            'AWGN',   1,    22,   0,   0e-6;   % 1 layer, 1 RX antenna
      7974,  'L1T1M24B024',            'AWGN',   1,  22.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7975,  'L1T1M25B024',            'AWGN',   1, 24.25,   0,   0e-6;   % 1 layer, 1 RX antenna
      7976,  'L1T1M26B024',            'AWGN',   1, 24.75,   0,   0e-6;   % 1 layer, 1 RX antenna
      7977,  'L1T1M27B024',            'AWGN',   1, 26.25,   0,   0e-6;   % 1 layer, 1 RX antenna
    % BLER for all MCS levels TS 38.214 Table 6.1.4.1-2
      7978,  'L1T4M00B024',            'AWGN',   1,    -8,   0,   0e-6;   % 1 layer, 1 RX antenna
      7979,  'L1T4M01B024',            'AWGN',   1,  -7.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7980,  'L1T4M02B024',            'AWGN',   1,    -7,   0,   0e-6;   % 1 layer, 1 RX antenna
      7981,  'L1T4M03B024',            'AWGN',   1,  -6.5,   0,   0e-6;   % 1 layer, 1 RX antenna
      7982,  'L1T4M04B024',            'AWGN',   1,    -6,   0,   0e-6;   % 1 layer, 1 RX antenna
      7983,  'L1T4M05B024',            'AWGN',   1,  -5.5,   0,   0e-6;   % 1 layer, 1 RX antenna
    % BLER for selected MCS levels TS 38.214 Table 5.1.3.1-2 with fading channels
    % BLER 30% with 0.5 dB resolution
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7984,  'L1T1M00B024', 'TDLC300-100-Low',   4,  -8.5,   0,   0e-6;   % 1 layer, 4 RX antenna
      7985,  'L1T1M09B024', 'TDLC300-100-Low',   4,   3.5,   0,   0e-6;   % 1 layer, 4 RX antenna
      7986,  'L1T1M18B024', 'TDLC300-100-Low',   4,    12,   0,   0e-6;   % 1 layer, 4 RX antenna
      7987,  'L1T1M27B024', 'TDLC300-100-Low',   4, 22.25,   0,   0e-6;   % 1 layer, 4 RX antenna
      7988,  'L2T1M00B024', 'TDLC300-100-Low',   4,  -4.5,   0,   0e-6;   % 2 layer, 4 RX antenna
      7989,  'L2T1M09B024', 'TDLC300-100-Low',   4,     8,   0,   0e-6;   % 2 layer, 4 RX antenna
      7990,  'L2T1M18B024', 'TDLC300-100-Low',   4,    18,   0,   0e-6;   % 2 layer, 4 RX antenna
      7991,  'L2T1M27B024', 'TDLC300-100-Low',   4,    34,   0,   0e-6;   % 2 layer, 4 RX antenna
    % PUSCH with perfect channel estimation
      7992,  'L1T1M00B024', 'TDLC300-100-Low',   4,  -8.5,   0,   0e-6;   % 1 layer, 4 RX antenna
    % TC for additional FRC
      7993,  'G-FR1-A1-5',             'AWGN',   1,     0,   0,   0e-6;   % 1 layer, 1 RX antenna
      7994,  'G-FR1-A2-5',             'AWGN',   1,     0,   0,   0e-6;   % 1 layer, 1 RX antenna
    };

elseif strcmp(test_version, '38.141-1.v15.14') % SNR requirement based on 38.141-1 for conformance tests
    CFG = {...
    % TS38-104 V16.4 Table 8.2.1.5-7 100MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
    % PUSCH data w/o tf precoding
      7004,  'G-FR1-A3-14', 'TDLB100-400-Low',   4,  -5.2, 200,   0e-6;
      7005,  'G-FR1-A4-14', 'TDLC300-100-Low',   4,   7.1, 200,   0e-6;
      7006,  'G-FR1-A5-14',   'TDLA30-10-Low',   4,   9.6, 200,   0e-6;
      7012,  'G-FR1-A3-28', 'TDLB100-400-Low',   4,  -1.4, 200,   0e-6;
      7013,  'G-FR1-A4-28', 'TDLC300-100-Low',   4,  12.4, 200,   0e-6;
    % TypeA, CSI-P1, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7051,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   6.0, 200,   0e-6; % case1: 5+2
      7052,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   4.9, 200,   0e-6; % case2: 20+20
    % TypeA, CSI-P2, 10MHz, 30kHz SCS
    % TC#           FRC            Chan        rxAnt  SNR   CFO   delay
      7053,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   0.4, 200,   0e-6; % case1: 5+2
      7054,  'G-FR1-A4-11', 'TDLC300-100-Low',   2,   3.0, 200,   0e-6; % case2: 20+20
    % PUSCH data w/ tf precoding
      7056,  'G-FR1-A3-32', 'TDLB100-400-Low',   4,  -5.1, 200,   0e-6;
      };
end

SRS_CFG = {
   % TC#  rnti  Nap nSym  Nrep sym0 cfgIdx seqId bwIdx cmbSz cmbOffset cs  fPos fShift frqH grpH resType Tsrs  Toffset idxSlot
    7701   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7702   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7703   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7704   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7705   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7706   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7707   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7708   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7709   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7710   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7711   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7712   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7713   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7714   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7715   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7716   0     2    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0    
    7717   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7718   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7719   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7720   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7721   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7722   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7723   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7724   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7725   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7726   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7727   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7728   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7729   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7730   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7731   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7732   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7733   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7734   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7735   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7736   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7737   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7738   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7739   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7740   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7741   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7742   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7743   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7744   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7745   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7746   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7747   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    7748   0     1    1     1   13    63     3    0     4        1      3    2    1      3    1     0      1      0       0
    };


% update CFG if necessary
enable_update_FRC_PRB_number = 0;
new_prb_number = 273;
if enable_update_FRC_PRB_number    
    for idx_case_number = 7950:7983  
        for idx_row = 1:size(CFG,1)
            if CFG{idx_row}==idx_case_number
                tmp = CFG{idx_row, 2};
                CFG{idx_row, 2} = strrep(tmp,'024',num2str(new_prb_number));
            end
        end
    end   
end

% export CFG into csv file
if isfield(batchsimCfg, 'export_cfg')
    if batchsimCfg.export_cfg
        if isfield(batchsimCfg,'export_fileName')
            cfg_table = cell2table(CFG);
            cfg_table.Properties.VariableNames = {'TC', 'FRC', 'Chan', 'rxAnt', 'SNR', 'CFO', 'delay'};
            %filter table and just keep valid TCs
            idx_row = ismember(cfg_table.TC,caseSet);
            writetable(cfg_table(idx_row,:),batchsimCfg.export_fileName);
        else
            error('Please specify the exporting path!')
        end
        nPerf = nan;
        nFail = nan;
        CBER  = nan; 
        TBER  = nan;
        CFOERR= nan; 
        TOERR = nan;
        SNR   = nan;
        return;
    end
end

[NallTest, ~] = size(CFG);
N_SNR = length(SNRoffset);
SNR = zeros(NallTest, N_SNR);
CBER = zeros(NallTest, N_SNR);
TBER = zeros(NallTest, N_SNR);
CFOERR = zeros(NallTest, N_SNR);
TOERR = zeros(NallTest, N_SNR);

switch puschTestMode
    case 0
        fprintf('Test PUSCH data detection performance:\n');
    case 1
        fprintf('Test PUSCH UCI detection performance:\n');
    case 2
        fprintf('Test PUSCH transform precoding detection performance:\n');
    case 3
        fprintf('Test PUSCH 1e-5 BLER detection performance:\n');
    case 7
        fprintf('Test mMIMO SRS/BFW/PUSCH performance:\n');
    case 8
        fprintf('Test PUSCH BLER with TDL/CDL channel model:\n');         
    case 10
        fprintf('Test PUSCH UL measurement:\n');
    case 11
        fprintf('Test PUSCH BLER with AWGN channel:\n');
    case 12
        fprintf('Test PUSCH BLER with fading channel:\n');       
    case 99
        fprintf('Save test configuration:\n');
    otherwise
        fprintf('Test PUSCH other performance:\n');
end

fprintf('\nTC#    Nframe      FRC            Chan       rxAnt  SNR  SNRoffset  CFO   delay   nCB  errCB   CBer   nTB  errTB   TBer Csi1Ber Csi2Ber cfoErr toErr sinrErr postSinrErr\n');
fprintf('------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n');

% initialize a txt file to save info used for cuPHY test bench
output_infor_for_cuphy = cell(1,1); 

load('CDLparam.mat', 'CDLparam');

num_generated_subscenarios = 0;
parfor idxSet = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc.pusch = 1;
    testAlloc_ssb = [];
    testAlloc_sib1 = [];
    testAlloc_pdcch = [];
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest) && ismember(caseNum, TcToTest_superset_from_batchsimCfg)
        snr = [];
        cber = [];
        tber = [];
        csi1ber = [];
        csi2ber = [];
        cfoErrHz = [];
        toErrMicroSec = [];
        sinrdB = [];
        postEqSinrdB = [];
        for idxSNR = 1:N_SNR
            if ismember(caseNum, [7701:7799])
                rng(7701,'threefry');
            else
                rng(caseNum,'threefry'); % clearly indicate the rng generator type although it is 'threefry' by default in parallel mode. 
            end
            SysPar = initSysPar(testAlloc);
            SysPar.SimCtrl.relNum = relNum;
            SysPar.SimCtrl.N_frame = Nframe;
            SysPar.SimCtrl.N_slot_run = 0;
            SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 1; % automatically set newDataIndication and rvIndex
            SysPar.SimCtrl.puschHARQ.MaxTransmissions = 4; % PUSCH number of transmissions per TB for HARQ performance testing
            SysPar.SimCtrl.timeDomainSim = 1;
            SysPar.SimCtrl.normalize_pusch_tx_power_over_layers = 1;
            SysPar.SimCtrl.alg.enableCfoEstimation = 1;
            SysPar.SimCtrl.alg.enableCfoCorrection = 1;
            SysPar.SimCtrl.alg.enableToEstimation = 1;
            SysPar.SimCtrl.alg.enableToCorrection = 0;
            SysPar.SimCtrl.alg.TdiMode = 1;
            SysPar.SimCtrl.alg.enableIrc = 1;
            SysPar.SimCtrl.alg.enableNoiseEstForZf = 0;
            SysPar.SimCtrl.alg.listLength = 8;
            SysPar.pusch{1} = loadFRC(SysPar.pusch{1}, CFG{idxSet, 2});
            SysPar.Chan{1}.type =  CFG{idxSet, 3};
            SysPar.carrier.Nant_gNB = CFG{idxSet, 4};
            SysPar.carrier.Nant_UE = SysPar.pusch{1}.nrOfLayers;
            snr_offset = SNRoffset(idxSNR);
            if snr_offset == 1000 % SNR = 1000dB is used for generating a noiseless TV for cuPHY SNR sweeping usage
                SysPar.Chan{1}.SNR =  snr_offset;
            else
                SysPar.Chan{1}.SNR =  CFG{idxSet, 5} + snr_offset;
            end
            SysPar.Chan{1}.CFO = CFG{idxSet, 6};
            SysPar.Chan{1}.delay = CFG{idxSet, 7};
            
            SysPar.SimCtrl.fp_flag_pusch_equalizer = 0;

            if ismember(caseNum, [7016:7030])
                SysPar.carrier.mu = 0;
                SysPar.carrier.N_grid_size_mu = 106;
            elseif ismember(caseNum, [7051:7054])
                SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 0;
                SysPar.SimCtrl.puschHARQ.MaxTransmissions = 1;
                SysPar.pusch{1}.pduBitmap = 2^0 + 2^1 + 2^5;
                SysPar.pusch{1}.harqAckBitLength = 0;   % TBD: needs to be 0
                if ismember(caseNum, [7051, 7053])
                    SysPar.pusch{1}.csiPart1BitLength = 5;
                    SysPar.SimCtrl.forceCsiPart2Length = 2;
                    %if batchsimCfg.batchsimMode==3
                    %    SysPar.pusch{1}.csiPart1Payload= [1 0 1 0 1];
                    %    SysPar.pusch{1}.csiPart2Payload = [1 0];
                    %end
                elseif ismember(caseNum, [7052, 7054])
                    SysPar.pusch{1}.csiPart1BitLength = 20;
                    SysPar.SimCtrl.forceCsiPart2Length = 20;
                end
                SysPar.pusch{1}.alphaScaling = 3;
                SysPar.pusch{1}.betaOffsetHarqAck = 11;
                SysPar.pusch{1}.betaOffsetCsi1 = 13;
                SysPar.pusch{1}.betaOffsetCsi2 = 13;
            elseif ismember(caseNum, [7055:7060])
                if ismember(caseNum, [7055:7057])
                    SysPar.carrier.N_grid_size_mu = 24;
                elseif ismember(caseNum, [7058:7060])
                    SysPar.carrier.N_grid_size_mu = 25;
                    SysPar.carrier.mu = 0;
                end
                SysPar.pusch{1}.TransformPrecoding = 0;
                SysPar.SimCtrl.alg.enableDftSOfdm = 1;
            elseif ismember(caseNum, [7911:7918])
                SysPar.SimCtrl.alg.enableIrc = 1;
                SysPar.SimCtrl.alg.enable_get_genie_meas = 1;
                SysPar.SimCtrl.enable_get_genie_channel_matrix = 1;
            elseif ismember(caseNum, [7101, 7950:7983])
                if enable_update_FRC_PRB_number == 1
                    SysPar.carrier.N_grid_size_mu = new_prb_number;
                else
                    SysPar.carrier.N_grid_size_mu = 24;
                end
                SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 0; % disable HARQ
                SysPar.SimCtrl.puschHARQ.MaxTransmissions = 1; % PUSCH number of transmissions per TB for HARQ performance testing
                if ismember(caseNum, [7978:7983])
                    SysPar.SimCtrl.pusch{1}.TransformPrecoding = 0;
                    SysPar.SimCtrl.pusch{1}.pi2BPSK = 1;
                end
            elseif ismember(caseNum, [7992])     % PUSCH with perfect channel estimation
                SysPar.SimCtrl.enable_get_genie_channel_matrix = 1;
                SysPar.SimCtrl.alg.enable_use_genie_channel_for_equalizer = 1;
                SysPar.SimCtrl.alg.TdiMode = 2;
                SysPar.SimCtrl.alg.enableCfoCorrection = 0;
            elseif ismember(caseNum, [7801, 7802])  % CDL
                SysPar.Chan{1}.UE_AntArraySize = [1 1 1];
                SysPar.carrier.Nant_UE = prod(SysPar.Chan{1}.UE_AntArraySize);
                SysPar.Chan{1}.UE_AntPattern = 'isotropic';
                SysPar.Chan{1}.UE_AntPolarizationAngles = [0 90];
                SysPar.Chan{1}.gNB_AntArraySize = [1 4 1];
                SysPar.Chan{1}.gNB_AntPattern = 'isotropic';
                SysPar.Chan{1}.gNB_AntPolarizationAngles = [0 90];
                if caseNum == 7802
                    SysPar.Chan{1}.model_source='custom';
                end
            elseif ismember(caseNum, [7803, 7804])  % CDL
                SysPar.Chan{1}.UE_AntArraySize = [1 1 1];
                SysPar.carrier.Nant_UE = prod(SysPar.Chan{1}.UE_AntArraySize);
                SysPar.Chan{1}.UE_AntPattern = 'isotropic';
                SysPar.Chan{1}.UE_AntPolarizationAngles = [0 90];
                SysPar.Chan{1}.gNB_AntArraySize = [1 2 2];
                SysPar.Chan{1}.gNB_AntPattern = '38.901';
                SysPar.Chan{1}.gNB_AntPolarizationAngles = [45 -45];
                if caseNum == 7804
                    SysPar.Chan{1}.model_source='custom';
                end                
            elseif ismember(caseNum, [7805, 7806])  % CDL
                SysPar.Chan{1}.UE_AntArraySize = [1 2 1];
                SysPar.carrier.Nant_UE = prod(SysPar.Chan{1}.UE_AntArraySize);
                SysPar.Chan{1}.UE_AntPattern = 'isotropic';
                SysPar.Chan{1}.UE_AntPolarizationAngles = [0 90];
                SysPar.Chan{1}.gNB_AntArraySize = [1 4 1];
                SysPar.Chan{1}.gNB_AntPattern = 'isotropic';
                SysPar.Chan{1}.gNB_AntPolarizationAngles = [0 90];
                if caseNum == 7806
                    SysPar.Chan{1}.model_source='custom';
                end                
            elseif ismember(caseNum, [7807, 7808])  % CDL
                SysPar.Chan{1}.UE_AntArraySize = [1 2 1];
                SysPar.carrier.Nant_UE = prod(SysPar.Chan{1}.UE_AntArraySize);
                SysPar.Chan{1}.UE_AntPattern = 'isotropic';
                SysPar.Chan{1}.UE_AntPolarizationAngles = [0 90];
                SysPar.Chan{1}.gNB_AntArraySize = [1 2 2];
                SysPar.Chan{1}.gNB_AntPattern = '38.901';
                SysPar.Chan{1}.gNB_AntPolarizationAngles = [45 -45];
                if caseNum == 7808
                    SysPar.Chan{1}.model_source='custom';
                end 
            elseif ismember(caseNum, [7822, 7824]) % TDL
                SysPar.Chan{1}.model_source='custom';
            elseif ismember(caseNum, [7701:7716])
                SysPar.Chan{1}.model_source='custom';
                SysPar.SimCtrl.enableUlRxBf = 1;
                SysPar.SimCtrl.enableSrsState = 1;
                SysPar.SimCtrl.bfw.enable_prg_chest = 1;
                SysPar.pusch{1}.prgSize = 2;
                if ismember(caseNum, [7701:2:7716])
                    SysPar.SimCtrl.BfKnownChannel = 1;
                else
                    SysPar.SimCtrl.BfKnownChannel = 0;
                end
                run_CDL = 1;
                if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                    switch SysPar.Chan{1}.type
                        case 'CDLA30-10'
                            SysPar.Chan{1}.DelayProfile = 'CDL-A';
                            SysPar.Chan{1}.DelaySpread = 30e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 10;
                        case 'CDLB100-400'
                            SysPar.Chan{1}.DelayProfile = 'CDL-B';
                            SysPar.Chan{1}.DelaySpread = 100e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 400;
                        case 'CDLC300-100'
                            SysPar.Chan{1}.DelayProfile = 'CDL-C';
                            SysPar.Chan{1}.DelaySpread = 300e-9;
                            SysPar.Chan{1}.MaximumDopplerShift = 100;
                        otherwise
                            warning('Chan.type is not supported ...\n')
                    end
                    SysPar.Chan{1}.type = 'CDL';
                    SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                    SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                end
                if ismember(caseNum, [7701:7708])
                    SysPar.SimCtrl.nSlotSrsUpdate = 20;
                else
                    SysPar.SimCtrl.nSlotSrsUpdate = 1;
                end
                % SysPar.SimCtrl.N_slot_run = 3;
                SysPar.carrier.Nant_gNB = 64;
                SysPar.carrier.Nant_UE = 2;
                SysPar.SimCtrl.CellConfigPorts = 2;
                % SysPar.Chan{1}.SNR = 20;
                SysPar.SimCtrl.checkSrsHestErr = 1;
                testAlloc.srs = 1;
                SysPar.testAlloc = testAlloc;
                SrsCfg = cell2mat(SRS_CFG);
                SrsCfgList = SrsCfg(:, 1);
                SrsCfgIdx = find(caseNum == SrsCfgList);
                SysPar.srs{1}.RNTI = SRS_CFG{SrsCfgIdx, 2};
                SysPar.srs{1}.numAntPorts = SRS_CFG{SrsCfgIdx, 3};
                SysPar.srs{1}.numSymbols = SRS_CFG{SrsCfgIdx, 4};
                SysPar.srs{1}.numRepetitions = SRS_CFG{SrsCfgIdx, 5};
                SysPar.srs{1}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                SysPar.srs{1}.configIndex = SRS_CFG{SrsCfgIdx, 7};
                SysPar.srs{1}.sequenceId = SRS_CFG{SrsCfgIdx, 8};
                SysPar.srs{1}.bandwidthIndex = SRS_CFG{SrsCfgIdx, 9};
                SysPar.srs{1}.combSize = SRS_CFG{SrsCfgIdx, 10};
                SysPar.srs{1}.combOffset = SRS_CFG{SrsCfgIdx, 11};
                SysPar.srs{1}.cyclicShift = SRS_CFG{SrsCfgIdx, 12};
                SysPar.srs{1}.frequencyPosition = SRS_CFG{SrsCfgIdx, 13};
                SysPar.srs{1}.frequencyShift = SRS_CFG{SrsCfgIdx, 14};
                SysPar.srs{1}.frequencyHopping = SRS_CFG{SrsCfgIdx, 15};
                SysPar.srs{1}.groupOrSequenceHopping = SRS_CFG{SrsCfgIdx, 16};
                SysPar.srs{1}.resourceType = SRS_CFG{SrsCfgIdx, 17};
                SysPar.srs{1}.Tsrs = SRS_CFG{SrsCfgIdx, 18};
                SysPar.srs{1}.Toffset = SRS_CFG{SrsCfgIdx, 19};
                SysPar.pusch{1}.RNTI = SysPar.srs{1}.RNTI;
                SysPar.pusch{1}.NrOfSymbols = 13;
                SysPar.pusch{1}.rbSize = 272;
            elseif ismember(caseNum, [7717:7748])
                if ismember(caseNum, [7717:7732])
                    nUe = 2;
                else
                    nUe = 8;
                end
                SysPar.SimCtrl.N_UE = nUe;
                SysPar.Chan{1}.model_source='custom';
                SysPar.SimCtrl.enableUlRxBf = 1;
                SysPar.SimCtrl.enableSrsState = 1;
                SysPar.SimCtrl.bfw.enable_prg_chest = 1;
                SysPar.pusch{1}.prgSize = 2;
                if ismember(caseNum, [7717:2:7748])
                    SysPar.SimCtrl.BfKnownChannel = 1;
                else
                    SysPar.SimCtrl.BfKnownChannel = 0;
                end
                run_CDL = 1;
                if nUe == 2
                    if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                        switch SysPar.Chan{1}.type
                            case 'CDLA30-10'
                                SysPar.Chan{1}.DelayProfile = 'CDL-A';
                                SysPar.Chan{1}.DelaySpread = 30e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 10;
                            case 'CDLB100-400'
                                SysPar.Chan{1}.DelayProfile = 'CDL-B';
                                SysPar.Chan{1}.DelaySpread = 100e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 400;
                            case 'CDLC300-100'
                                SysPar.Chan{1}.DelayProfile = 'CDL-C';
                                SysPar.Chan{1}.DelaySpread = 300e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 100;
                            otherwise
                                warning('Chan.type is not supported ...\n')
                        end
                        SysPar.Chan{1}.type = 'CDL';
                        SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                        SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                    end
                else
                    if run_CDL && ~strcmp(SysPar.Chan{1}.type, 'AWGN')
                        switch SysPar.Chan{1}.type
                            case 'CDLA30-10'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_A;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_A;
                                SysPar.Chan{1}.DelaySpread = 30e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 10;
                            case 'CDLB100-400'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_B;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_B;
                                SysPar.Chan{1}.DelaySpread = 100e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 400;
                            case 'CDLC300-100'
                                % load CDLparam;
                                SysPar.Chan{1}.DelayProfile = 'CDL_customized';
                                SysPar.Chan{1}.CDL_DPA = CDLparam.DPA_C;
                                SysPar.Chan{1}.CDL_PCP = CDLparam.PCP_C;
                                SysPar.Chan{1}.DelaySpread = 300e-9;
                                SysPar.Chan{1}.MaximumDopplerShift = 100;
                            otherwise
                                warning('Chan.type is not supported ...\n')
                        end
                        SysPar.Chan{1}.type = 'CDL';
                        SysPar.Chan{1}.gNB_AntArraySize = [4,8,2];
                        SysPar.Chan{1}.UE_AntArraySize = [1,1,2];
                    end
                end

                for idxUe = 1:nUe
                    SysPar.Chan{idxUe} =SysPar.Chan{1};
                    if strcmp(SysPar.Chan{1}.DelayProfile, 'CDL_customized')
                        angle_ue = (idxUe-0.5)*120/nUe - 60;
                        SysPar.Chan{idxUe}.CDL_DPA(:, 4) = angle_ue;
                    end
                end
                if ismember(caseNum, [7717:7724, 7733:7740])
                    SysPar.SimCtrl.nSlotSrsUpdate = 20;
                else
                    SysPar.SimCtrl.nSlotSrsUpdate = 1;
                end
                % SysPar.SimCtrl.N_slot_run = 3;
                SysPar.carrier.Nant_gNB = 64;
                SysPar.carrier.Nant_UE = 1;
                SysPar.SimCtrl.CellConfigPorts = nUe;
                % SysPar.Chan{1}.SNR = 20;
                SysPar.SimCtrl.checkSrsHestErr = 1;
                testAlloc.srs = nUe;
                testAlloc.pusch = nUe;
                SysPar.testAlloc = testAlloc;
                SrsCfg = cell2mat(SRS_CFG);
                SrsCfgList = SrsCfg(:, 1);
                SrsCfgIdx = find(caseNum == SrsCfgList);
                for idxUe = 1:nUe
                    SysPar.srs{idxUe} = SysPar.srs{1};
                    SysPar.srs{idxUe}.RNTI = SRS_CFG{SrsCfgIdx, 2} + idxUe-1;
                    SysPar.srs{idxUe}.numAntPorts = 1; % SRS_CFG{SrsCfgIdx, 3};
                    SysPar.srs{idxUe}.numSymbols = SRS_CFG{SrsCfgIdx, 4};
                    SysPar.srs{idxUe}.numRepetitions = SRS_CFG{SrsCfgIdx, 5};
                    if nUe > 4
                        if idxUe < 5
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6} - 1;
                        else
                            SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                        end
                    else
                        SysPar.srs{idxUe}.timeStartPosition = SRS_CFG{SrsCfgIdx, 6};
                    end
                    SysPar.srs{idxUe}.configIndex = SRS_CFG{SrsCfgIdx, 7};
                    SysPar.srs{idxUe}.sequenceId = SRS_CFG{SrsCfgIdx, 8};
                    SysPar.srs{idxUe}.bandwidthIndex = SRS_CFG{SrsCfgIdx, 9};
                    SysPar.srs{idxUe}.combSize = SRS_CFG{SrsCfgIdx, 10};
                    SysPar.srs{idxUe}.combOffset = mod(SRS_CFG{SrsCfgIdx, 11} + idxUe-1, 4);
                    SysPar.srs{idxUe}.cyclicShift = SRS_CFG{SrsCfgIdx, 12};
                    SysPar.srs{idxUe}.frequencyPosition = SRS_CFG{SrsCfgIdx, 13};
                    SysPar.srs{idxUe}.frequencyShift = SRS_CFG{SrsCfgIdx, 14};
                    SysPar.srs{idxUe}.frequencyHopping = SRS_CFG{SrsCfgIdx, 15};
                    SysPar.srs{idxUe}.groupOrSequenceHopping = SRS_CFG{SrsCfgIdx, 16};
                    SysPar.srs{idxUe}.resourceType = SRS_CFG{SrsCfgIdx, 17};
                    SysPar.srs{idxUe}.Tsrs = SRS_CFG{SrsCfgIdx, 18};
                    SysPar.srs{idxUe}.Toffset = SRS_CFG{SrsCfgIdx, 19};
                    SysPar.srs{idxUe}.idxUE = idxUe-1;
                    SysPar.pusch{idxUe} = SysPar.pusch{1};
                    SysPar.pusch{idxUe}.RNTI = SysPar.srs{idxUe}.RNTI;
                    SysPar.pusch{idxUe}.portIdx = idxUe - 1;
                    if nUe > 4
                        SysPar.pusch{idxUe}.NrOfSymbols = 12;
                    else
                        SysPar.pusch{idxUe}.NrOfSymbols = 13;
                    end
                    SysPar.pusch{idxUe}.rbSize = 272;
                    SysPar.pusch{idxUe}.idxUE = idxUe-1;
                end
            end

            if isfield(batchsimCfg, 'enable_UL_Rx_RF_impairments') 
                if batchsimCfg.enable_UL_Rx_RF_impairments
                    SysPar.SimCtrl.enable_UL_Rx_RF_impairments = 1;
                    SysPar.RF.UL_Rx_tot_NF_dB = 0;
                    SysPar.RF.UL_Rx_gain_dB = 33.5;
                    SysPar.RF.UL_Rx_IIP3_dBm = -3.5
                    SysPar.RF.UL_Rx_IQ_imblance_gain_dB = 0.005;
                    SysPar.RF.UL_Rx_IQ_imblance_phase_degree = 0.063;
                    SysPar.RF.UL_Rx_PN_level_offset_dB = 0;
                    SysPar.RF.UL_Rx_PN_spectral_mask_freqOffset_Hz = [1e5, 1e6, 1e7];
                    SysPar.RF.UL_Rx_PN_spectral_mask_power_dBcPerHz = [-104.0, -125.0, -149.0];
                    SysPar.RF.UL_Rx_DC_offset_real_volt = 2e-7;
                    SysPar.carrier.N_grid_size_mu = 273; % to make sure the sampling rate is twice larger than max freq. offset of phase noise
                end
            end

            if puschTestMode == 99
                full_cfg_template_yaml_file_name = ['cfg-', num2str(caseNum), '.yaml'];
                WriteYaml(full_cfg_template_yaml_file_name, SysPar);
                continue;
            elseif batchsimCfg.batchsimMode==0
                [SysPar, UE, gNB] = nrSimulator(SysPar);
            else
                for idx_seed = 1:length(batchsimCfg.seed_list)
                    my_seed = batchsimCfg.seed_list(idx_seed);
                    SysPar.SimCtrl.seed = my_seed;
                    SysPar.SimCtrl.batchsim.save_results = 1;
                    SysPar.SimCtrl.batchsim.save_results_short = 1;
                    if ~ismember(caseNum, [7911:7918]) % for PUSCH measurement, we don't need to set MIMO equalizer related params. Just use the default config.
                        SysPar.SimCtrl.alg.enableIrc = batchsimCfg.SimCtrl.alg.enableIrc;
                        SysPar.SimCtrl.alg.enableNoiseEstForZf =  batchsimCfg.SimCtrl.alg.enableNoiseEstForZf;
                        SysPar.SimCtrl.alg.enable_use_genie_nCov =  batchsimCfg.SimCtrl.alg.enable_use_genie_nCov;
                        SysPar.SimCtrl.alg.genie_nCov_method =  batchsimCfg.SimCtrl.alg.genie_nCov_method;
                        SysPar.SimCtrl.alg.enable_nCov_shrinkage =  batchsimCfg.SimCtrl.alg.enable_nCov_shrinkage;
                        SysPar.SimCtrl.alg.nCov_shrinkage_method =  batchsimCfg.SimCtrl.alg.nCov_shrinkage_method;
                        SysPar.SimCtrl.alg.enable_get_genie_meas =  batchsimCfg.SimCtrl.alg.enable_get_genie_meas;        
                        SysPar.SimCtrl.enable_get_genie_channel_matrix =  batchsimCfg.SimCtrl.enable_get_genie_channel_matrix;
                        if batchsimCfg.enable_pusch_use_perfect_channel_est_for_equalizer
                            SysPar.SimCtrl.enable_get_genie_channel_matrix = batchsimCfg.SimCtrl.enable_get_genie_channel_matrix;
                            SysPar.SimCtrl.alg.enable_use_genie_channel_for_equalizer = batchsimCfg.SimCtrl.alg.enable_use_genie_channel_for_equalizer;
                            SysPar.SimCtrl.alg.TdiMode = batchsimCfg.SimCtrl.alg.TdiMode;
                            SysPar.SimCtrl.alg.enableCfoCorrection = batchsimCfg.SimCtrl.alg.enableCfoCorrection;
                        end
                    end 
                    % tdi mode
%                     SysPar.SimCtrl.alg.TdiMode = 2;
                    % config sub-slot processing
                    enable_sub_slot_proc = 0;%1;
                    if enable_sub_slot_proc == 1
                        SysPar.SimCtrl.subslot_proc_option = 2;
                        SysPar.SimCtrl.alg.enable_avg_nCov_prbs_fd = 1;
                        SysPar.SimCtrl.alg.win_size_avg_nCov_prbs_fd = 3;
                        SysPar.SimCtrl.alg.threshold_avg_nCov_prbs_fd_dB = 3;
                        SysPar.SimCtrl.alg.enable_instant_equ_coef_cfo_corr = 1;
                    end
%                     SysPar.SimCtrl.useCuphySoftDemapper = 1;%2;
%                     SysPar.SimCtrl.BFPforCuphy = 9;
                    subscenario_name = sprintf('scenario_TC%d___seed_%d___SNR_%2.2f',caseNum,my_seed,SysPar.Chan{1}.SNR);
                    subscenario_folder_name = fullfile(batchsimCfg.ws_folder_name,subscenario_name);
                    if ~exist(subscenario_folder_name, 'dir')
                       mkdir(subscenario_folder_name)
                    end
                    if batchsimCfg.batchsimMode == 2
                        % freeze Tx and Chan
                        SysPar.SimCtrl.enable_freeze_tx_and_channel = 1;
                    elseif batchsimCfg.batchsimMode == 3
                        % freeze Tx signal (except DMRS symbols which depends on slot idx)
                        SysPar.SimCtrl.enable_freeze_tx = 1;
                    end
                    if batchsimCfg.batchsimMode == 2 || (batchsimCfg.batchsimMode == 3)
                        % enable logging tx Xtf into TV
                        SysPar.SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl = 1;
                        SysPar.SimCtrl.genTV.enable_logging_tx_Xtf = 1;
                        SysPar.SimCtrl.genTV.enable_logging_carrier_and_channel_info = 1;
                        % disable HARQ
                        if batchsimCfg.cfgMode ~= 2 % in conformance TV generation, we need harq for PUSCH data
                            SysPar.SimCtrl.puschHARQ.EnableAutoHARQ = 0;
                            SysPar.SimCtrl.puschHARQ.MaxTransmissions = 1;
                        end
                    end
                    % set the fake SNR for ZF to be a very low value as cuPHY has perf issue when it is high.
%                     SysPar.SimCtrl.alg.fakeSNRdBForZf = -36;
                    %
                    if (SysPar.Chan{1}.SNR == 1000) || (batchsimCfg.cfgMode==2) % cfgMode 2 is used for conformance TV generation
                        SysPar.SimCtrl.N_frame = 1;
                        SysPar.SimCtrl.N_slot_run = 0;
                        SysPar.SimCtrl.genTV.enable = 1;
                        SysPar.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar.SimCtrl.genTV.FAPI = 1;
                        if batchsimCfg.cfgMode==2
                            SysPar.SimCtrl.genTV.TVname = 'e2e_cfm_pusch';
                            SysPar.SimCtrl.genTV.slotIdx = [4, 5, 14, 15];
                            SysPar.SimCtrl.genTV.launchPattern = 1;
                            SysPar.SimCtrl.genTV.FAPIyaml = 1;
                            num_pusch_pdu = length(SysPar.pusch);
                            for idx_pdu = 1:num_pusch_pdu
                                SysPar.pusch{idx_pdu}.DmrsScramblingId = 0; % required by conformance tests.
                            end
                        else
                            SysPar.SimCtrl.genTV.TVname = 'TV';
                            SysPar.SimCtrl.genTV.slotIdx = 0;
                        end
                    end  
                    if snr_offset == 1000
                        SNR_range_for_cuPHY = CFG{idxSet, 5}+SNRoffset(SNRoffset<1000);
                        str_SNR_range_for_cuPHY = sprintf('%2.2f,', SNR_range_for_cuPHY);
                        str_SNR_range_for_cuPHY = str_SNR_range_for_cuPHY(1:end-1); % remove the last comma
                        output_infor_for_cuphy = [output_infor_for_cuphy;{sprintf('TV_subfolder_name=%s, SNR_range=[%s], N_slots=%d;\n',subscenario_name, str_SNR_range_for_cuPHY, Nframe*10*2^(SysPar.carrier.mu))}];
                    end
                    full_cfg_template_yaml_file_name = fullfile(subscenario_folder_name,'cfg_template.yaml');
                    fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name)
                    WriteYaml(full_cfg_template_yaml_file_name, SysPar);
                    num_generated_subscenarios = num_generated_subscenarios + 1;

                    % for E2E conformance tests TVs, we also need to add SSB and SIB1 slots so that gNB and the Keysight devices can sycn up together
                    if batchsimCfg.cfgMode==2
                        % SSB
                        testAlloc_ssb.dl = 1;
                        testAlloc_ssb.ul = 0;
                        testAlloc_ssb.ssb = 1;
                        testAlloc_ssb.pdcch = 0;
                        testAlloc_ssb.pdsch = 0;
                        testAlloc_ssb.csirs = 0;
                        testAlloc_ssb.prach = 0;
                        testAlloc_ssb.pucch = 0;
                        testAlloc_ssb.pusch = 0;
                        testAlloc_ssb.srs = 0;
                        SysPar_ssb = initSysPar(testAlloc_ssb); 
                        SysPar_ssb_yaml = ReadYaml('./test/e2e_cfm_ssb.yaml', 0, 0);
                        SysPar_ssb = updateStruct(SysPar_ssb, SysPar_ssb_yaml);
                        SysPar_ssb.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_ssb.SimCtrl.genTV.TVname = 'e2e_cfm_ssb';
                        SysPar_ssb.SimCtrl.genTV.launchPattern = 1;
                        SysPar_ssb.SimCtrl.genTV.FAPIyaml = 1;
                        full_cfg_template_yaml_file_name_ssb = fullfile(subscenario_folder_name,'cfg_template_ssb.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_ssb)
                        WriteYaml(full_cfg_template_yaml_file_name_ssb, SysPar_ssb);
                        % SIB1
                        testAlloc_sib1.dl = 1;
                        testAlloc_sib1.ul = 0;
                        testAlloc_sib1.ssb = 0;
                        testAlloc_sib1.pdcch = 1;
                        testAlloc_sib1.pdsch = 1;
                        testAlloc_sib1.csirs = 0;
                        testAlloc_sib1.prach = 0;
                        testAlloc_sib1.pucch = 0;
                        testAlloc_sib1.pusch = 0;
                        testAlloc_sib1.srs = 0;
                        SysPar_sib1 = initSysPar(testAlloc_sib1); 
                        SysPar_sib1_yaml = ReadYaml('./test/e2e_cfm_sib1.yaml', 0, 0);
                        SysPar_sib1 = updateStruct(SysPar_sib1, SysPar_sib1_yaml);
                        SysPar_sib1.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_sib1.SimCtrl.genTV.TVname = 'e2e_cfm_sib1';
                        SysPar_sib1.SimCtrl.genTV.launchPattern = 1;
                        SysPar_sib1.SimCtrl.genTV.FAPIyaml = 1;
                        full_cfg_template_yaml_file_name_sib1 = fullfile(subscenario_folder_name,'cfg_template_sib1.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_sib1)
                        WriteYaml(full_cfg_template_yaml_file_name_sib1, SysPar_sib1);
                        % PDCCH for PUSCH
                        testAlloc_pdcch.dl = 1;
                        testAlloc_pdcch.ul = 0;
                        testAlloc_pdcch.ssb = 0;
                        testAlloc_pdcch.pdcch = 1;
                        testAlloc_pdcch.pdsch = 0;
                        testAlloc_pdcch.csirs = 0;
                        testAlloc_pdcch.prach = 0;
                        testAlloc_pdcch.pucch = 0;
                        testAlloc_pdcch.pusch = 0;
                        testAlloc_pdcch.srs = 0;
                        SysPar_pdcch = initSysPar(testAlloc_pdcch); 
                        SysPar_pdcch_yaml = ReadYaml('./test/e2e_cfm_ul_pdcch.yaml', 0, 0);
                        SysPar_pdcch = updateStruct(SysPar_pdcch, SysPar_pdcch_yaml);
                        SysPar_pdcch.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar_pdcch.SimCtrl.genTV.TVname = 'e2e_cfm_pdcch';
                        SysPar_pdcch.SimCtrl.genTV.launchPattern = 1;
                        SysPar_pdcch.SimCtrl.genTV.FAPIyaml = 1;
                        %update MCS
                        MCS_this_test = SysPar.pusch{1}.mcsIndex;
                        MCS_bin = de2bi(MCS_this_test,5,'left-msb');
                        DCI_payload_bits = SysPar_pdcch.pdcch{1}.DCI{1}.Payload;
                        DCI_payload_bits(23:27) = MCS_bin;
                        SysPar_pdcch.pdcch{1}.DCI{1}.Payload = DCI_payload_bits;
                        full_cfg_template_yaml_file_name_pdcch = fullfile(subscenario_folder_name,'cfg_template_pdcch.yaml');
                        fprintf('Generating subscenario: %s\n', full_cfg_template_yaml_file_name_pdcch)
                        WriteYaml(full_cfg_template_yaml_file_name_pdcch, SysPar_pdcch);
                      
                    end
                end
                continue;
            end
            SimCtrl = SysPar.SimCtrl;
        
            snr(idxSNR) = SysPar.Chan{1}.SNR; 
            results = SimCtrl.results;
            pusch = results.pusch;
            nPusch = length(pusch);
            cbCnt = 0;
            cbErrorCnt = 0;
            tbCnt = 0;
            tbErrorCnt = 0;
            harqErrCnt = 0;
            csi1ErrCnt = 0;
            csi2ErrCnt = 0;
            for idxPusch = 1:nPusch
                cbCnt = cbCnt + pusch{idxPusch}.cbCnt;
                cbErrorCnt = cbErrorCnt + pusch{idxPusch}.cbErrorCnt;
                tbCnt = tbCnt + pusch{idxPusch}.tbCnt;
                tbErrorCnt = tbErrorCnt + pusch{idxPusch}.tbErrorCnt;
                harqErrCnt = harqErrCnt + pusch{idxPusch}.harqErrCnt;
                csi1ErrCnt = csi1ErrCnt + pusch{idxPusch}.csi1ErrCnt;
                csi2ErrCnt = csi2ErrCnt + pusch{idxPusch}.csi2ErrCnt;
            end
            cber(idxSNR) = cbErrorCnt/cbCnt;
            tber(idxSNR) = tbErrorCnt/tbCnt;
            csi1ber(idxSNR) = csi1ErrCnt/tbCnt;
            csi2ber(idxSNR) = csi2ErrCnt/tbCnt;
            
            cfoErrHz(idxSNR) = sqrt(mean(abs(pusch{1}.cfoEstHz(end,:) - SysPar.Chan{1}.CFO).^2));
            toErrMicroSec(idxSNR) = sqrt(mean(abs(pusch{1}.toEstMicroSec(end,:) - SysPar.Chan{1}.delay*1e6).^2));
            if SimCtrl.alg.enable_get_genie_meas
                sinrdB(idxSNR) = sqrt(mean(abs(pusch{1}.sinrdB(end,:) - pusch{1}.genie_sinrdB ).^2));
                postEqSinrdB(idxSNR) = sqrt(mean(abs(pusch{1}.postEqSinrdB(end,:) - pusch{1}.genie_postEqSinrdB).^2));
            else
                sinrdB(idxSNR) = sqrt(mean(abs(pusch{1}.sinrdB(end,:) - SysPar.Chan{1}.SNR(end) - 10*log10(SysPar.pusch{1}.nrOfLayers)).^2));
                postEqSinrdB(idxSNR) = sqrt(mean(abs(pusch{1}.postEqSinrdB(end,:) - 10*log10(SysPar.carrier.Nant_gNB) - SysPar.Chan{1}.SNR).^2));
            end

            fprintf('%4d  % 4d   %12s  %15s  %4d   %4.1f     %4.1f   %4d   %4.1f  %5d  %5d  %4.3f  %4d  %4d   %4.3f   %4.3f   %4.3f %4d    %4.2f  %4.2f     %4.2f\n',...
                CFG{idxSet, 1}, Nframe, CFG{idxSet, 2}, CFG{idxSet, 3}, ...
                CFG{idxSet, 4}, CFG{idxSet, 5}, snr_offset, CFG{idxSet, 6}, ...
                1e6*CFG{idxSet, 7}, cbCnt, cbErrorCnt, cber(idxSNR), ...
                tbCnt, tbErrorCnt, tber(idxSNR), csi1ber(idxSNR), csi2ber(idxSNR), round(cfoErrHz(idxSNR)), toErrMicroSec(idxSNR), sinrdB(idxSNR), postEqSinrdB(idxSNR));
        end
        if batchsimCfg.batchsimMode>0 || puschTestMode == 99
            continue;
        end
        SNR(idxSet,:) = snr;
        CBER(idxSet,:) = cber;
        TBER(idxSet,:) = tber;
        CSI1BER(idxSet,:) = csi1ber;
        CSI2BER(idxSet,:) = csi2ber;
        CFOERR(idxSet,:) = cfoErrHz;
        TOERR(idxSet, :) = toErrMicroSec;
    end
end

if batchsimCfg.batchsimMode==0 && puschTestMode < 99
    fprintf('------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n');
    nPerf = 0;
    nFail = 0;
    for idxSet = 1:NallTest
        caseNum = CFG{idxSet, 1};
        if ismember(caseNum, TcToTest)
            nPerf = nPerf + 1;
            if ismember(caseNum, [7051:7052]) % UCI on PUSCH test CSI-P1
                if find(CSI1BER(idxSet,:) > 0) % should be 0.001 for CSI-P1 per 3GPP test requirement
                    nFail = nFail + 1;
                end
            elseif ismember(caseNum, [7053:7054]) % UCI on PUSCH test CSI-P2
                if find(CSI2BER(idxSet,:) > 0) % should be 0.01 for CSI-P2 per 3GPP test requirement
                    nFail = nFail + 1;
                end
            elseif ismember(caseNum, [7101]) % 0.001% BLER
                if find(TBER(idxSet,:) > 0)
                    nFail = nFail + 1;
                end
            elseif ismember(caseNum, [7701:7748]) % mMIMO PUSCH test, using SRS-based beamforming
                if find(TBER(idxSet,:) > 0.4)
                    nFail = nFail + 1;
                end
            else
                if find(TBER(idxSet,:) > 0.5) % should be 0.3 per 3GPP test requirement
                    nFail = nFail + 1;
                end
            end
        end
    end
    fprintf('Total TC = %d, PASS = %d, FAIL = %d\n\n', nPerf, nPerf-nFail, nFail);
else
    if (batchsimCfg.batchsimMode == 2) || (batchsimCfg.batchsimMode == 3)
        txt_file_name = fullfile(batchsimCfg.ws_folder_name,'info_for_cuPHY_unit_test.txt');
        fid = fopen(txt_file_name,'w+'); % create file and write to it 
        output_infor_for_cuphy = output_infor_for_cuphy(~cellfun(@isempty,output_infor_for_cuphy));
        for idx = 1:length(output_infor_for_cuphy)
            fprintf(fid, '%s', output_infor_for_cuphy{idx});
        end
        fclose(fid);
    end
    fprintf('Total number of subcenarios generated: %d\n',num_generated_subscenarios)
end
 
fprintf('\n');
return

% function plotMcsBler(SNR, TBER)
% 
% SNR1 = SNR(end-27:end,:);
% TBER1 = TBER(end-27:end, :);
% 
% [nTC, nSNR] = size(SNR1);
% 
% figure; 
% for idxTC = 1:nTC
%     semilogy(SNR1(idxTC,:), TBER1(idxTC, :), 'LineWidth',2); hold on; grid on;
% end
% 
% title('TS 38.214 Table 5.1.3.1-2, 1T1R, MCS 0-27, 24 PRBs, 2000 slots'); xlabel('SNR (dB)'); ylabel('TBER');
% 
% return

