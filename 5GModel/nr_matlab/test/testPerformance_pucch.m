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

function [nPerf, nFail, P_err] = testPerformance_pucch(caseSet, SNRoffset, Nframe, pucchFormat, pucchTestMode, batchsimCfg, relNum)

tic;
if nargin == 0
    caseSet = 'full';
    SNRoffset = 0;
    Nframe = 1;
    pucchFormat = 1;
    pucchTestMode = 2;
    batchsimCfg.batchsimMode = 0; % 0: disable batchsimMode, 1: phase2 perf study, 2: performance match test for 5GModel and cuPHY
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 1
    SNRoffset = 0;
    Nframe = 1;
    pucchFormat = 1;
    pucchTestMode = 2;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 2
    Nframe = 1;    
    pucchFormat = 1;
    pucchTestMode = 0;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 3
    pucchFormat = 1;
    pucchTestMode = 2;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 4
    pucchTestMode = 2;
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 5
    batchsimCfg.batchsimMode = 0;
    batchsimCfg.cfgMode = 0;
    relNum = 10000;
elseif nargin == 6
    relNum = 10000;
end

compact_TC = [6001:6999];
full_TC = [6001:6999];
selected_TC = [6001:6999];

if isnumeric(caseSet)
    TcToTest = caseSet;
else
    switch pucchFormat
        case 0
            switch pucchTestMode
                case 1 % DTX to ACK
                    TcToTest = [6001:6012];
                case 2 % ANACK to ACK
                    TcToTest = [6001:6012];
                case 3 % ACK missed
                    TcToTest = [6001:6012];
                case 6 % DTX to ACK with UCI multiplexing on the same PRB
                    TcToTest = [6013:6024];
                case 7 % ANACK to ACK with UCI multiplexing on the same PRB
                    TcToTest = [6013:6024];
                case 8 % ACK missed with UCI multiplexing on the same PRB
                    TcToTest = [6013:6024];
                case 9 % measurement
                    TcToTest = [6091:6092];
                otherwise
                    error('pucchTestMode is not supported...\n');
            end
        case 1
            switch pucchTestMode
                case 1 % DTX to ACK
                    TcToTest = [6101:6112];
                case 2 % NACK to ACK
                    TcToTest = [6101:6106];
                case 3 % ACK missed
                    TcToTest = [6107:6112];
                case 9 % measurement
                    TcToTest = [6191:6192];
                otherwise
                    error('pucchTestMode is not supported...\n');
            end
        case 2
            switch pucchTestMode
                case 1 % DTX to ACK
                    TcToTest = [6201:6210];
                case 3 % ACK missed
                    TcToTest = [6201:6210];
                case 4 % UCI BLER
                    TcToTest = [6211:6220];
                case 9 % measurement
                    TcToTest = [6291:6292];
                otherwise
                    error('pucchTestMode is not supported...\n');
            end
        case 3
            switch pucchTestMode
                case 4 % UCI BLER
                    TcToTest = [6301:6320];
                case 9 % measurement
                    TcToTest = [6391:6392];
                otherwise
                    error('pucchTestMode is not supported...\n');
            end
            
        otherwise
            error('pucchFormat is not supported...\n');
            
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
    % Format 0     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-104 V16.4 Table 8.3.2.2-2 100MHz, 30kHz SCS
      6001,     1,   273, 'TDLC300-100-Low',   2,    1,   9.2, 200,   0e-6; 
      6002,     1,   273, 'TDLC300-100-Low',   2,    2,   3.5, 200,   0e-6; 
      6003,     1,   273, 'TDLC300-100-Low',   4,    1,   3.3, 200,   0e-6; 
      6004,     1,   273, 'TDLC300-100-Low',   4,    2,  -0.8, 200,   0e-6; 
      6005,     1,   273, 'TDLC300-100-Low',   8,    1,  -1.0, 200,   0e-6; 
      6006,     1,   273, 'TDLC300-100-Low',   8,    2,  -3.9, 200,   0e-6;   
    % TS38-104 Table 8.3.2.2-1 20MHz, 15kHz SCS  
      6007,     0,   106, 'TDLC300-100-Low',   2,    1,   9.3, 200,   0e-6; 
      6008,     0,   106, 'TDLC300-100-Low',   2,    2,   3.3, 200,   0e-6; 
      6009,     0,   106, 'TDLC300-100-Low',   4,    1,   3.2, 200,   0e-6;
      6010,     0,   106, 'TDLC300-100-Low',   4,    2,  -0.8, 200,   0e-6; 
      6011,     0,   106, 'TDLC300-100-Low',   8,    1,  -1.1, 200,   0e-6; 
      6012,     0,   106, 'TDLC300-100-Low',   8,    2,  -4.0, 200,   0e-6;  
    % UCI multiplexing, 100MHz, 30kHz SCS
      6013,     1,   273, 'TDLC300-100-Low',   2,    1,   9.2, 200,   0e-6; 
      6014,     1,   273, 'TDLC300-100-Low',   2,    2,   3.5, 200,   0e-6; 
      6015,     1,   273, 'TDLC300-100-Low',   4,    1,   3.3, 200,   0e-6;
      6016,     1,   273, 'TDLC300-100-Low',   4,    2,  -0.8, 200,   0e-6; 
      6017,     1,   273, 'TDLC300-100-Low',   8,    1,  -1.0, 200,   0e-6; 
      6018,     1,   273, 'TDLC300-100-Low',   8,    2,  -3.9, 200,   0e-6;
    % UCI multiplexing, 20MHz, 15kHz SCS  
      6019,     0,   106, 'TDLC300-100-Low',   2,    1,   9.3, 200,   0e-6;
      6020,     0,   106, 'TDLC300-100-Low',   2,    2,   3.3, 200,   0e-6;
      6021,     0,   106, 'TDLC300-100-Low',   4,    1,   3.2, 200,   0e-6;
      6022,     0,   106, 'TDLC300-100-Low',   4,    2,  -0.8, 200,   0e-6; 
      6023,     0,   106, 'TDLC300-100-Low',   8,    1,  -1.1, 200,   0e-6;
      6024,     0,   106, 'TDLC300-100-Low',   8,    2,  -4.0, 200,   0e-6;  
    % measurement accuracy
      6091,     1,   273,            'AWGN',   4,    1,     0, 100,   1e-6;
      6092,     1,   273, 'TDLC300-100-Low',   4,    1,     0, 100,   1e-6; 

    % Format 1     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % NACK to ACK
    % TS38-104 Table 8.3.3.1.2-2 100MHz, 30kHz SCS
      6101,     1,   273, 'TDLC300-100-Low',   2,   14,  -3.5, 200,   0e-6; 
      6102,     1,   273, 'TDLC300-100-Low',   4,   14,  -8.0, 200,   0e-6; 
      6103,     1,   273, 'TDLC300-100-Low',   8,   14, -11.3, 200,   0e-6; 
    % TS38-104 Table 8.3.3.1.2-1 20MHz, 15kHz SCS  
      6104,     0,   106, 'TDLC300-100-Low',   2,   14,  -3.6, 200,   0e-6; 
      6105,     0,   106, 'TDLC300-100-Low',   4,   14,  -8.4, 200,   0e-6; 
      6106,     0,   106, 'TDLC300-100-Low',   8,   14, -11.4, 200,   0e-6;   
    % ACK missed
    % TS38-104 Table 8.3.3.2.2-2 100MHz, 30kHz SCS
      6107,     1,   273, 'TDLC300-100-Low',   2,   14,  -4.2, 200,   0e-6; 
      6108,     1,   273, 'TDLC300-100-Low',   4,   14,  -8.3, 200,   0e-6; 
      6109,     1,   273, 'TDLC300-100-Low',   8,   14, -11.4, 200,   0e-6; 
    % TS38-104 Table 8.3.3.2.2-1 20MHz, 15kHz SCS  
      6110,     0,   106, 'TDLC300-100-Low',   2,   14,  -5.0, 200,   0e-6; 
      6111,     0,   106, 'TDLC300-100-Low',   4,   14,  -8.5, 200,   0e-6; 
      6112,     0,   106, 'TDLC300-100-Low',   8,   14, -11.5, 200,   0e-6;  
    % measurement accuracy
      6191,     1,   273,            'AWGN',   4,   14,     0, 100,   1e-6;
      6192,     1,   273, 'TDLC300-100-Low',   4,   14,     0, 100,   1e-6;

    % Format 2     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-104 Table 8.3.4.1.2-2 100MHz, 30kHz SCS
      6201,     1,   273, 'TDLC300-100-Low',   2,    1,   5.7, 200,   0e-6; 
      6202,     1,   273, 'TDLC300-100-Low',   4,    1,   0.4, 200,   0e-6; 
      6203,     1,   273, 'TDLC300-100-Low',   8,    1,  -3.3, 200,   0e-6; 
    % TS38-104 Table 8.3.4.1.2-1 20MHz, 15kHz SCS
      6204,     0,   106, 'TDLC300-100-Low',   2,    1,   5.9, 200,   0e-6; 
      6205,     0,   106, 'TDLC300-100-Low',   4,    1,   0.3, 200,   0e-6; 
      6206,     0,   106, 'TDLC300-100-Low',   8,    1,  -3.5, 200,   0e-6;  
    % UCI BLER
    % TS38-104 Table 8.3.4.2.2-2 100MHz, 30kHz SCS
      6211,     1,   273, 'TDLC300-100-Low',   2,    2,   0.3, 200,   0e-6; 
      6212,     1,   273, 'TDLC300-100-Low',   4,    2,  -3.4, 200,   0e-6; 
      6213,     1,   273, 'TDLC300-100-Low',   8,    2,  -5.9, 200,   0e-6; 
    % TS38-104 Table 8.3.4.2.2-1 20MHz, 15kHz SCS
      6214,     0,   106, 'TDLC300-100-Low',   2,    2,   1.2, 200,   0e-6; 
      6215,     0,   106, 'TDLC300-100-Low',   4,    2,  -3.2, 200,   0e-6; 
      6216,     0,   106, 'TDLC300-100-Low',   8,    2,  -6.8, 200,   0e-6;  
    % measurement accuracy
      6291,     1,   273,            'AWGN',   4,    1,     0, 100,   1e-6;
      6292,     1,   273, 'TDLC300-100-Low',   4,    1,     0, 100,   1e-6;

    % Format 3     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % UCI BLER
    % TS38-104 Table 8.3.5.2-2 100MHz, 30kHz SCS
      6301,     1,   273, 'TDLC300-100-Low',   2,   14,   0.9, 200,   0e-6; 
      6302,     1,   273, 'TDLC300-100-Low',   2,   14,   0.1, 200,   0e-6; 
      6303,     1,   273, 'TDLC300-100-Low',   4,   14,  -3.5, 200,   0e-6; 
      6304,     1,   273, 'TDLC300-100-Low',   4,   14,  -4.2, 200,   0e-6; 
      6305,     1,   273, 'TDLC300-100-Low',   8,   14,  -6.8, 200,   0e-6; 
      6306,     1,   273, 'TDLC300-100-Low',   8,   14,  -7.7, 200,   0e-6; 
      6307,     1,   273, 'TDLC300-100-Low',   2,    4,   1.5, 200,   0e-6; 
      6308,     1,   273, 'TDLC300-100-Low',   4,    4,  -3.0, 200,   0e-6; 
      6309,     1,   273, 'TDLC300-100-Low',   8,    4,  -6.2, 200,   0e-6; 

    % TS38-104 Table 8.3.5.2-1 20MHz, 15kHz SCS  
      6311,     0,   106, 'TDLC300-100-Low',   2,   14,   0.3, 200,   0e-6; 
      6312,     0,   106, 'TDLC300-100-Low',   2,   14,  -0.1, 200,   0e-6; 
      6313,     0,   106, 'TDLC300-100-Low',   4,   14,  -3.8, 200,   0e-6; 
      6314,     0,   106, 'TDLC300-100-Low',   4,   14,  -4.0, 200,   0e-6; 
      6315,     0,   106, 'TDLC300-100-Low',   8,   14,  -6.9, 200,   0e-6; 
      6316,     0,   106, 'TDLC300-100-Low',   8,   14,  -7.7, 200,   0e-6; 
      6317,     0,   106, 'TDLC300-100-Low',   2,    4,   2.0, 200,   0e-6; 
      6318,     0,   106, 'TDLC300-100-Low',   4,    4,  -2.5, 200,   0e-6; 
      6319,     0,   106, 'TDLC300-100-Low',   8,    4,  -6.2, 200,   0e-6; 

    % measurement accuracy
      6391,     1,   273,            'AWGN',   4,   14,     0, 100,   1e-6;
      6392,     1,   273, 'TDLC300-100-Low',   4,   14,     0, 100,   1e-6;
    };    
elseif strcmp(test_version, '38.141-1') % SNR requirement based on 38.141-1
    CFG = {...
    % Format 0     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-104 Table 8.3.2.2-2 100MHz, 30kHz SCS
      6001,     1,   273, 'TDLC300-100-Low',   2,    1,   9.2,   0,   0e-6; 
      6002,     1,   273, 'TDLC300-100-Low',   2,    2,   3.5,   0,   0e-6; 
      6003,     1,   273, 'TDLC300-100-Low',   4,    1,   3.3,   0,   0e-6; 
      6004,     1,   273, 'TDLC300-100-Low',   4,    2,  -0.8,   0,   0e-6; 
      6005,     1,   273, 'TDLC300-100-Low',   8,    1,  -1.0,   0,   0e-6; 
      6006,     1,   273, 'TDLC300-100-Low',   8,    2,  -3.9,   0,   0e-6;   
    % TS38-104 Table 8.3.2.2-1 20MHz, 15kHz SCS  
      6007,     0,   106, 'TDLC300-100-Low',   2,    1,   9.3,   0,   0e-6; 
      6008,     0,   106, 'TDLC300-100-Low',   2,    2,   3.3,   0,   0e-6; 
      6009,     0,   106, 'TDLC300-100-Low',   4,    1,   3.2,   0,   0e-6;
      6010,     0,   106, 'TDLC300-100-Low',   4,    2,  -0.8,   0,   0e-6; 
      6011,     0,   106, 'TDLC300-100-Low',   8,    1,  -1.1,   0,   0e-6; 
      6012,     0,   106, 'TDLC300-100-Low',   8,    2,  -4.0,   0,   0e-6;  
    % UCI multiplexing, 100MHz, 30kHz SCS
      6013,     1,   273, 'TDLC300-100-Low',   2,    1,   9.2,   0,   0e-6; 
      6014,     1,   273, 'TDLC300-100-Low',   2,    2,   3.5,   0,   0e-6; 
      6015,     1,   273, 'TDLC300-100-Low',   4,    1,   3.3,   0,   0e-6; 
      6016,     1,   273, 'TDLC300-100-Low',   4,    2,  -0.8,   0,   0e-6; 
      6017,     1,   273, 'TDLC300-100-Low',   8,    1,  -1.0,   0,   0e-6; 
      6018,     1,   273, 'TDLC300-100-Low',   8,    2,  -3.9,   0,   0e-6;
    % UCI multiplexing, 20MHz, 15kHz SCS  
      6019,     0,   106, 'TDLC300-100-Low',   2,    1,   9.3,   0,   0e-6; 
      6020,     0,   106, 'TDLC300-100-Low',   2,    2,   3.3,   0,   0e-6; 
      6021,     0,   106, 'TDLC300-100-Low',   4,    1,   3.2,   0,   0e-6;
      6022,     0,   106, 'TDLC300-100-Low',   4,    2,  -0.8,   0,   0e-6; 
      6023,     0,   106, 'TDLC300-100-Low',   8,    1,  -1.1,   0,   0e-6; 
      6024,     0,   106, 'TDLC300-100-Low',   8,    2,  -4.0,   0,   0e-6;  
    % measurement accuracy
      6091,     1,   273,            'AWGN',   4,    1,     0, 100,   1e-6;
      6092,     1,   273, 'TDLC300-100-Low',   4,    1,     0, 100,   1e-6;

    % Format 1     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % NACK to ACK
    % TS38-104 Table 8.3.3.1.2-2 100MHz, 30kHz SCS
      6101,     1,   273, 'TDLC300-100-Low',   2,   14,  -3.5,   0,   0e-6; 
      6102,     1,   273, 'TDLC300-100-Low',   4,   14,  -8.0,   0,   0e-6; 
      6103,     1,   273, 'TDLC300-100-Low',   8,   14, -11.3,   0,   0e-6; 
    % TS38-104 Table 8.3.3.1.2-1 20MHz, 15kHz SCS  
      6104,     0,   106, 'TDLC300-100-Low',   2,   14,  -3.6,   0,   0e-6; 
      6105,     0,   106, 'TDLC300-100-Low',   4,   14,  -8.4,   0,   0e-6; 
      6106,     0,   106, 'TDLC300-100-Low',   8,   14, -11.4,   0,   0e-6;   
    % ACK missed
    % TS38-104 Table 8.3.3.2.2-2 100MHz, 30kHz SCS
      6107,     1,   273, 'TDLC300-100-Low',   2,   14,  -4.2,   0,   0e-6; 
      6108,     1,   273, 'TDLC300-100-Low',   4,   14,  -8.3,   0,   0e-6; 
      6109,     1,   273, 'TDLC300-100-Low',   8,   14, -11.4,   0,   0e-6; 
    % TS38-104 Table 8.3.3.2.2-1 20MHz, 15kHz SCS  
      6110,     0,   106, 'TDLC300-100-Low',   2,   14,  -5.0,   0,   0e-6; 
      6111,     0,   106, 'TDLC300-100-Low',   4,   14,  -8.5,   0,   0e-6; 
      6112,     0,   106, 'TDLC300-100-Low',   8,   14, -11.5,   0,   0e-6;  
    % measurement accuracy
      6191,     1,   273,            'AWGN',   4,   14,     0, 100,   1e-6;
      6192,     1,   273, 'TDLC300-100-Low',   4,   14,     0, 100,   1e-6;

    % Format 2     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-104 Table 8.3.4.1.2-2 100MHz, 30kHz SCS
      6201,     1,   273, 'TDLC300-100-Low',   2,    1,   5.7,   0,   0e-6; 
      6202,     1,   273, 'TDLC300-100-Low',   4,    1,   0.4,   0,   0e-6; 
      6203,     1,   273, 'TDLC300-100-Low',   8,    1,  -3.3,   0,   0e-6; 
    % TS38-104 Table 8.3.4.1.2-1 20MHz, 15kHz SCS
      6204,     0,   106, 'TDLC300-100-Low',   2,    1,   5.9,   0,   0e-6; 
      6205,     0,   106, 'TDLC300-100-Low',   4,    1,   0.3,   0,   0e-6; 
      6206,     0,   106, 'TDLC300-100-Low',   8,    1,  -3.5,   0,   0e-6;  
    % UCI BLER
    % TS38-104 Table 8.3.4.2.2-2 100MHz, 30kHz SCS
      6211,     1,   273, 'TDLC300-100-Low',   2,    2,   0.3,   0,   0e-6; 
      6212,     1,   273, 'TDLC300-100-Low',   4,    2,  -3.4,   0,   0e-6; 
      6213,     1,   273, 'TDLC300-100-Low',   8,    2,  -5.9,   0,   0e-6; 
    % TS38-141 Table 8.3.4.2.2-1 20MHz, 15kHz SCS
      6214,     0,   106, 'TDLC300-100-Low',   2,    2,   1.2,   0,   0e-6; 
      6215,     0,   106, 'TDLC300-100-Low',   4,    2,  -3.2,   0,   0e-6; 
      6216,     0,   106, 'TDLC300-100-Low',   8,    2,  -6.8,   0,   0e-6;  
    % measurement accuracy
      6291,     1,   273,            'AWGN',   4,    1,     0, 100,   1e-6;
      6292,     1,   273, 'TDLC300-100-Low',   4,    1,     0, 100,   1e-6;

    % Format 3     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % UCI BLER
    % TS38-141 Table 8.3.4.5-2 100MHz, 30kHz SCS
      6301,     1,   273, 'TDLC300-100-Low',   2,   14,   1.5,   0,   0e-6; 
      6302,     1,   273, 'TDLC300-100-Low',   2,   14,   0.7,   0,   0e-6; 
      6303,     1,   273, 'TDLC300-100-Low',   4,   14,  -2.9,   0,   0e-6; 
      6304,     1,   273, 'TDLC300-100-Low',   4,   14,  -3.6,   0,   0e-6; 
      6305,     1,   273, 'TDLC300-100-Low',   8,   14,  -6.2,   0,   0e-6; 
      6306,     1,   273, 'TDLC300-100-Low',   8,   14,  -7.1,   0,   0e-6; 
      6307,     1,   273, 'TDLC300-100-Low',   2,    4,   2.1,   0,   0e-6; 
      6308,     1,   273, 'TDLC300-100-Low',   4,    4,  -2.4,   0,   0e-6; 
      6309,     1,   273, 'TDLC300-100-Low',   8,    4,  -5.6,   0,   0e-6; 

    % TS38-141 Table 8.3.4.5-1 20MHz, 15kHz SCS  
      6311,     0,   106, 'TDLC300-100-Low',   2,   14,   0.9,   0,   0e-6; 
      6312,     0,   106, 'TDLC300-100-Low',   2,   14,   0.5,   0,   0e-6; 
      6313,     0,   106, 'TDLC300-100-Low',   4,   14,  -3.2,   0,   0e-6; 
      6314,     0,   106, 'TDLC300-100-Low',   4,   14,  -3.4,   0,   0e-6; 
      6315,     0,   106, 'TDLC300-100-Low',   8,   14,  -6.3,   0,   0e-6; 
      6316,     0,   106, 'TDLC300-100-Low',   8,   14,  -7.1,   0,   0e-6; 
      6317,     0,   106, 'TDLC300-100-Low',   2,    4,   2.6,   0,   0e-6; 
      6318,     0,   106, 'TDLC300-100-Low',   4,    4,  -1.9,   0,   0e-6; 
      6319,     0,   106, 'TDLC300-100-Low',   8,    4,  -5.6,   0,   0e-6; 

    % measurement accuracy
      6391,     1,   273,            'AWGN',   4,   14,     0, 100,   1e-6;
      6392,     1,   273, 'TDLC300-100-Low',   4,   14,     0, 100,   1e-6;
    };

elseif strcmp(test_version, '38.141-1.v15.14') % SNR requirement based on 38.141-1 for conformance tests
    CFG = {...
    % Format 0     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-141 V15.14 Table 8.3.1.5-2 100MHz, 30kHz SCS
      6001,     1,   273, 'TDLC300-100-Low',   2,    1,   9.8, 200,   0e-6; 
      6002,     1,   273, 'TDLC300-100-Low',   2,    2,   4.1, 200,   0e-6; 
      6003,     1,   273, 'TDLC300-100-Low',   4,    1,   3.9, 200,   0e-6; 
      6004,     1,   273, 'TDLC300-100-Low',   4,    2,  -0.2, 200,   0e-6; 
      6005,     1,   273, 'TDLC300-100-Low',   8,    1,  -0.4, 200,   0e-6; 
      6006,     1,   273, 'TDLC300-100-Low',   8,    2,  -3.3, 200,   0e-6;   
    % TS38-141 Table 8.3.1.5-1 20MHz, 15kHz SCS  
      6007,     0,   106, 'TDLC300-100-Low',   2,    1,   9.9, 200,   0e-6; 
      6008,     0,   106, 'TDLC300-100-Low',   2,    2,   3.9, 200,   0e-6; 
      6009,     0,   106, 'TDLC300-100-Low',   4,    1,   3.8, 200,   0e-6;
      6010,     0,   106, 'TDLC300-100-Low',   4,    2,  -0.2, 200,   0e-6; 
      6011,     0,   106, 'TDLC300-100-Low',   8,    1,  -0.5, 200,   0e-6; 
      6012,     0,   106, 'TDLC300-100-Low',   8,    2,  -3.3, 200,   0e-6;  

    % Format 1     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % NACK to ACK
    % TS38-141 Table 8.3.2.1.5-2 100MHz, 30kHz SCS
      6101,     1,   273, 'TDLC300-100-Low',   2,   14,  -2.9, 200,   0e-6; 
      6102,     1,   273, 'TDLC300-100-Low',   4,   14,  -7.4, 200,   0e-6; 
      6103,     1,   273, 'TDLC300-100-Low',   8,   14, -10.4, 200,   0e-6; 
    % TS38-141 Table 8.3.2.1.5-1 20MHz, 15kHz SCS  
      6104,     0,   106, 'TDLC300-100-Low',   2,   14,  -3.0, 200,   0e-6; 
      6105,     0,   106, 'TDLC300-100-Low',   4,   14,  -7.8, 200,   0e-6; 
      6106,     0,   106, 'TDLC300-100-Low',   8,   14, -10.8, 200,   0e-6;   
    % ACK missed
    % TS38-141 Table 8.3.2.2.5-2 100MHz, 30kHz SCS
      6107,     1,   273, 'TDLC300-100-Low',   2,   14,  -3.6, 200,   0e-6; 
      6108,     1,   273, 'TDLC300-100-Low',   4,   14,  -7.7, 200,   0e-6; 
      6109,     1,   273, 'TDLC300-100-Low',   8,   14, -10.8, 200,   0e-6; 
    % TS38-141 Table 8.3.2.2.5-1 20MHz, 15kHz SCS  
      6110,     0,   106, 'TDLC300-100-Low',   2,   14,  -4.4, 200,   0e-6; 
      6111,     0,   106, 'TDLC300-100-Low',   4,   14,  -7.9, 200,   0e-6; 
      6112,     0,   106, 'TDLC300-100-Low',   8,   14, -10.9, 200,   0e-6;  

    % Format 2     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % ACK missed
    % TS38-141 Table 8.3.3.1.5-2 100MHz, 30kHz SCS
      6201,     1,   273, 'TDLC300-100-Low',   2,    1,   6.3, 200,   0e-6; 
      6202,     1,   273, 'TDLC300-100-Low',   4,    1,   1.0, 200,   0e-6; 
      6203,     1,   273, 'TDLC300-100-Low',   8,    1,  -2.7, 200,   0e-6; 
    % TS38-141 Table 8.3.3.1.5-1 20MHz, 15kHz SCS
      6204,     0,   106, 'TDLC300-100-Low',   2,    1,   6.5, 200,   0e-6; 
      6205,     0,   106, 'TDLC300-100-Low',   4,    1,   0.9, 200,   0e-6; 
      6206,     0,   106, 'TDLC300-100-Low',   8,    1,  -2.9, 200,   0e-6;  
    % UCI BLER
    % TS38-141 Table 8.3.3.2.5-2 100MHz, 30kHz SCS
      6211,     1,   273, 'TDLC300-100-Low',   2,    2,   0.9, 200,   0e-6; 
      6212,     1,   273, 'TDLC300-100-Low',   4,    2,  -2.8, 200,   0e-6; 
      6213,     1,   273, 'TDLC300-100-Low',   8,    2,  -5.3, 200,   0e-6; 
    % TS38-141 Table 8.3.3.2.5-1 20MHz, 15kHz SCS
      6214,     0,   106, 'TDLC300-100-Low',   2,    2,   1.8, 200,   0e-6; 
      6215,     0,   106, 'TDLC300-100-Low',   4,    2,  -2.6, 200,   0e-6; 
      6216,     0,   106, 'TDLC300-100-Low',   8,    2,  -6.2, 200,   0e-6;  

    % Format 3     
    % TC#      mu    PRB         Chan        rxAnt  nSym  SNR   CFO   delay
    % UCI BLER
    % TS38-141 Table 8.3.4.5-2 100MHz, 30kHz SCS
      6301,     1,   273, 'TDLC300-100-Low',   2,   14,   1.5, 200,   0e-6; 
      6302,     1,   273, 'TDLC300-100-Low',   2,   14,   0.7, 200,   0e-6; 
      6303,     1,   273, 'TDLC300-100-Low',   4,   14,  -2.9, 200,   0e-6; 
      6304,     1,   273, 'TDLC300-100-Low',   4,   14,  -3.6, 200,   0e-6; 
      6305,     1,   273, 'TDLC300-100-Low',   8,   14,  -6.2, 200,   0e-6; 
      6306,     1,   273, 'TDLC300-100-Low',   8,   14,  -7.1, 200,   0e-6; 
      6307,     1,   273, 'TDLC300-100-Low',   2,    4,   2.1, 200,   0e-6; 
      6308,     1,   273, 'TDLC300-100-Low',   4,    4,  -2.4, 200,   0e-6; 
      6309,     1,   273, 'TDLC300-100-Low',   8,    4,  -5.6, 200,   0e-6; 

    % TS38-104 Table 8.3.4.5-1 20MHz, 15kHz SCS  
      6311,     0,   106, 'TDLC300-100-Low',   2,   14,   0.9, 200,   0e-6; 
      6312,     0,   106, 'TDLC300-100-Low',   2,   14,   0.5, 200,   0e-6; 
      6313,     0,   106, 'TDLC300-100-Low',   4,   14,  -3.2, 200,   0e-6; 
      6314,     0,   106, 'TDLC300-100-Low',   4,   14,  -3.4, 200,   0e-6; 
      6315,     0,   106, 'TDLC300-100-Low',   8,   14,  -6.3, 200,   0e-6; 
      6316,     0,   106, 'TDLC300-100-Low',   8,   14,  -7.1, 200,   0e-6; 
      6317,     0,   106, 'TDLC300-100-Low',   2,    4,   2.6, 200,   0e-6; 
      6318,     0,   106, 'TDLC300-100-Low',   4,    4,  -1.9, 200,   0e-6; 
      6319,     0,   106, 'TDLC300-100-Low',   8,    4,  -5.6, 200,   0e-6; 

    }; 
end

% export CFG into csv file
if isfield(batchsimCfg, 'export_cfg')
    if batchsimCfg.export_cfg
        if isfield(batchsimCfg,'export_fileName')
            cfg_table = cell2table(CFG);
            cfg_table.Properties.VariableNames = {'TC', 'mu', 'PRB', 'Chan', 'rxAnt', 'nSym', 'SNR','CFO','delay'};
            %filter table and just keep valid TCs
            idx_row = ismember(cfg_table.TC,caseSet);
            writetable(cfg_table(idx_row,:),batchsimCfg.export_fileName);
        else
            error('Please specify the exporting path!')
        end
        nPerf = nan;
        nFail = nan;
        P_err  = nan;         
        return;
    end
end

[NallTest, ~] = size(CFG);
N_SNR = length(SNRoffset);
P_err = zeros(NallTest, N_SNR);
to_ErrSec = zeros(NallTest, N_SNR);
sinr_dB = zeros(NallTest, N_SNR);

if batchsimCfg.cfgMode==2 % just one PUCCH payload for conformance test TVs
    Npucch = 1;
else
    switch pucchFormat
        case {0, 1}
            if (batchsimCfg.batchsimMode == 2) || (batchsimCfg.batchsimMode == 3) % batchsimMode 2 is used to generate TV for PerfMatchTest. TV limitation: 30 UCIs for PF0 and 24 UCIs for PF1
                Npucch = 20;
            else
                Npucch = 50;
            end
        case {2, 3}
            Npucch = 5;
        otherwise
             error('pucchFormat is not supported...\n');
    end
end
        
switch pucchTestMode 
    case 1
        fprintf('Test PUCCH format-%1d ACK false detection:\n', pucchFormat);
    case 2
        fprintf('Test PUCCH format-%1d NACK to ACK detection:\n', pucchFormat);
    case 3
        fprintf('Test PUCCH format-%1d ACK missed detection rate:\n', pucchFormat);
    case 6
        fprintf('Test PUCCH format-%1d ACK false detection with UCI multiplexing:\n', pucchFormat);
    case 7 
        fprintf('Test PUCCH format-%1d NACK to ACK detection with UCI multiplexing:\n', pucchFormat);
    case 8
        fprintf('Test PUCCH format-%1d ACK missed detection rate with UCI multiplexing:\n', pucchFormat);
    case 4
        fprintf('Test PUCCH format-%1d UCI BLER:\n', pucchFormat);
end

fprintf('\nTC#  Nframe    mu    PRB         Chan        rxAnt   nSym   SNR   SNRoffset   CFO  delay  total  error    Perr  taErr snrErr\n');
fprintf('----------------------------------------------------------------------------------------------------------------------------\n');

% initialize a txt file to save info used for cuPHY test bench
output_infor_for_cuphy = cell(1,1);

num_generated_subscenarios = 0;
parfor idxSet = 1:NallTest
    testAlloc = [];
    testAlloc.dl = 0;
    testAlloc.ul = 1;
    testAlloc_ssb = [];
    testAlloc_sib1 = [];
    testAlloc.pucch = Npucch;
    caseNum = CFG{idxSet, 1};
    if ismember(caseNum, TcToTest)  && ismember(caseNum, TcToTest_superset_from_batchsimCfg)
        Perr = [];
        toErrSec = [];
        sinrdB = [];
        prbSize = 0;
        for idxSNR = 1:N_SNR
            rng(caseNum,'threefry'); % clearly indicate the rng generator type although it is 'threefry' by default in parallel mode. 
            SysPar = initSysPar(testAlloc);
            SysPar.SimCtrl.relNum = relNum;
            SysPar.SimCtrl.N_frame = Nframe;
            SysPar.SimCtrl.N_slot_run = 0;
            SysPar.SimCtrl.timeDomainSim = 1;
            SysPar.SimCtrl.alg.useNrUCIDecode = 0;
            SysPar.SimCtrl.alg.dtxModePf2 = 1;
            SysPar.SimCtrl.alg.TdiModePf3 = 1;
            SysPar.SimCtrl.alg.listLength = 8;

            SysPar.carrier.mu = CFG{idxSet, 2};
            SysPar.carrier.N_grid_size_mu = CFG{idxSet, 3};
            for idxPucch = 1:Npucch
                switch pucchFormat
                    case 0
                        SysPar.pucch{idxPucch} = SysPar.pucch{1};
                        if ismember(caseNum, [6013:6024])
                            SysPar.pucch{idxPucch}.prbStart = ceil(idxPucch/2)-1;
                            SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + ceil(idxPucch/2)-1;
                            SysPar.pucch{idxPucch}.cs0 = 1 - mod(idxPucch ,2);
                        else
                            SysPar.pucch{idxPucch}.prbStart = idxPucch-1;
                            SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + idxPucch-1;
                            SysPar.pucch{idxPucch}.cs0 = 0;
                        end
                        SysPar.pucch{idxPucch}.BitLenHarq =  1;
                        SysPar.pucch{idxPucch}.FormatType =  0;
                        %                 SysPar.pucch{idxPucch}.secondHopPRB = CFG{idxSet, 3}-1;
                        
                        SysPar.pucch{idxPucch}.prbSize = 1;
                        SysPar.pucch{idxPucch}.nSym = CFG{idxSet, 6};
                        if SysPar.pucch{idxPucch}.nSym == 1
                            SysPar.pucch{idxPucch}.startSym = 13;
                            SysPar.pucch{idxPucch}.freqHopFlag = 0;
                        else
                            SysPar.pucch{idxPucch}.startSym = 12;
                            SysPar.pucch{idxPucch}.freqHopFlag = 1;
                        end
                        
                        SysPar.pucch{idxPucch}.tOCCidx = 0;
                        
                        SysPar.pucch{idxPucch}.GroupHopping = 0;
                        SysPar.pucch{idxPucch}.DTXthreshold = 1;
                        if pucchTestMode == 1 || pucchTestMode == 6
                            SysPar.pucch{idxPucch}.DTX = 1;
                        end
                        if batchsimCfg.cfgMode==2 % to generate conformance TVs
                            SysPar.pucch{idxPucch}.secondHopPRB = (CFG{idxSet, 3}-1) - (SysPar.pucch{idxPucch}.prbSize-1);
                        end
                    case 1
                        SysPar.pucch{idxPucch} = SysPar.pucch{1};
                        SysPar.pucch{idxPucch}.BitLenHarq =  2;
                        SysPar.pucch{idxPucch}.FormatType =  1;
                        SysPar.pucch{idxPucch}.prbStart = (idxPucch-1);
                        %                 SysPar.pucch{idxPucch}.secondHopPRB = CFG{idxSet, 3}-1;
                        SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + (idxPucch-1);
                        SysPar.pucch{idxPucch}.prbSize = 1;
                        SysPar.pucch{idxPucch}.startSym = 0;
                        SysPar.pucch{idxPucch}.nSym = CFG{idxSet, 6};
                        SysPar.pucch{idxPucch}.cs0 = 0;
                        SysPar.pucch{idxPucch}.tOCCidx = 0;
                        SysPar.pucch{idxPucch}.freqHopFlag = 1;
                        SysPar.pucch{idxPucch}.GroupHopping = 0;
                        SysPar.pucch{idxPucch}.DTXthreshold = 1;
                        if pucchTestMode == 1
                            SysPar.pucch{idxPucch}.DTX = 1;
                        end
                        if batchsimCfg.cfgMode==2 % to generate conformance TVs
                            SysPar.pucch{idxPucch}.secondHopPRB = (CFG{idxSet, 3}-1) - (SysPar.pucch{idxPucch}.prbSize-1);
                        end
                    case 2
                        if pucchTestMode == 4
                            SysPar.pucch{idxPucch} = SysPar.pucch{1};
                            SysPar.pucch{idxPucch}.FormatType =  2;
                            SysPar.pucch{idxPucch}.freqHopFlag = 1;
                            SysPar.pucch{idxPucch}.GroupHopping = 0;
                            SysPar.pucch{idxPucch}.sequenceHopFlag = 0;
                            prbSize = 9;
                            SysPar.pucch{idxPucch}.prbSize = prbSize;
                            SysPar.pucch{idxPucch}.nSym = CFG{idxSet, 6};
                            
                            SysPar.pucch{idxPucch}.prbStart = (idxPucch-1)*prbSize;
                            %                 SysPar.pucch{idxPucch}.secondHopPRB = CFG{idxSet, 3}-1;
                            SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + (idxPucch-1)*prbSize;
                            SysPar.pucch{idxPucch}.BitLenCsiPart1 = 22;
                            SysPar.pucch{idxPucch}.startSym = 12;
                            SysPar.pucch{idxPucch}.DmrsScramblingId = 0;
                        else
                            SysPar.pucch{idxPucch} = SysPar.pucch{1};
                            SysPar.pucch{idxPucch}.FormatType =  2;
                            SysPar.pucch{idxPucch}.freqHopFlag = 0;
                            SysPar.pucch{idxPucch}.GroupHopping = 0;
                            SysPar.pucch{idxPucch}.sequenceHopFlag = 0;
                            prbSize = 4;
                            SysPar.pucch{idxPucch}.prbSize = prbSize;
                            SysPar.pucch{idxPucch}.nSym = CFG{idxSet, 6};
                            
                            SysPar.pucch{idxPucch}.prbStart = (idxPucch-1)*prbSize;
                            %                 SysPar.pucch{idxPucch}.secondHopPRB = CFG{idxSet, 3}-1;
                            SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + (idxPucch-1)*prbSize;
                            SysPar.pucch{idxPucch}.BitLenHarq = 4;
                            SysPar.pucch{idxPucch}.startSym = 13;
                            SysPar.pucch{idxPucch}.DmrsScramblingId = 0;
                            SysPar.pucch{idxPucch}.DTXthreshold = 1;
                            if pucchTestMode == 1
                                SysPar.pucch{idxPucch}.DTX = 1;
                            end
                        end
                        if batchsimCfg.cfgMode==2 % to generate conformance TVs
                            SysPar.pucch{idxPucch}.secondHopPRB = (CFG{idxSet, 3}-1) - (SysPar.pucch{idxPucch}.prbSize-1);
                        end
                    case 3
                        SysPar.pucch{idxPucch} = SysPar.pucch{1};
                        SysPar.pucch{idxPucch}.FormatType =  3;
                        SysPar.pucch{idxPucch}.freqHopFlag = 1;
                        SysPar.pucch{idxPucch}.GroupHopping = 0;
                        SysPar.pucch{idxPucch}.sequenceHopFlag = 0;
                        SysPar.pucch{idxPucch}.hoppingId = 0;
                        if CFG{idxSet, 6} == 14 % Test 1
                            prbSize = 1;
                            SysPar.pucch{idxPucch}.prbSize = prbSize;
                            SysPar.pucch{idxPucch}.nSym = 14;
                            if mod(caseNum, 2) == 0
                                SysPar.pucch{idxPucch}.AddDmrsFlag = 1;
                            else
                                SysPar.pucch{idxPucch}.AddDmrsFlag = 0;
                            end
                        elseif CFG{idxSet, 6} == 4 % Test 2
                            prbSize = 3;
                            SysPar.pucch{idxPucch}.prbSize = prbSize;
                            SysPar.pucch{idxPucch}.nSym = 4;
                            SysPar.pucch{idxPucch}.AddDmrsFlag = 0;
                        end
                        SysPar.pucch{idxPucch}.prbStart = (idxPucch-1)*prbSize;
                        %                 SysPar.pucch{idxPucch}.secondHopPRB = CFG{idxSet, 3}-1;
                        SysPar.pucch{idxPucch}.secondHopPRB = floor(CFG{idxSet, 3}/2) + (idxPucch-1)*prbSize;
                        
                        SysPar.pucch{idxPucch}.BitLenCsiPart1 = 16;
                        SysPar.pucch{idxPucch}.startSym = 0;
                        
                        if batchsimCfg.cfgMode==2 % to generate conformance TVs
                            SysPar.pucch{idxPucch}.secondHopPRB = (CFG{idxSet, 3}-1) - (SysPar.pucch{idxPucch}.prbSize-1);
                        end
                    otherwise
                        error('pucchFormat is not supported...\n');
                end
                if batchsimCfg.cfgMode==2 % required by conformance TVs
                    SysPar.pucch{idxPucch}.SRFlag = 0;
                    SysPar.pucch{idxPucch}.groupHopFlag = 0;
                    SysPar.pucch{idxPucch}.sequenceHopFlag = 0;
                    SysPar.pucch{idxPucch}.hoppingId= 0;
                    SysPar.pucch{idxPucch}.cs0 = 0;
                end
            end
            SysPar.Chan{1}.type = CFG{idxSet, 4};
            SysPar.carrier.Nant_gNB = CFG{idxSet, 5};
            SysPar.carrier.Nant_UE = 1;
            snr_offset = SNRoffset(idxSNR);
            if snr_offset == 1000 % SNR = 1000dB is used for generating a noiseless TV for cuPHY SNR sweeping usage
                SysPar.Chan{1}.SNR =  snr_offset;
            else
                SysPar.Chan{1}.SNR =  CFG{idxSet, 7} + snr_offset;
            end
            SysPar.Chan{1}.CFO = CFG{idxSet, 8};
            SysPar.Chan{1}.delay = CFG{idxSet, 9};
            
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

            if batchsimCfg.batchsimMode==0
                [SysPar, UE, gNB] = nrSimulator(SysPar);
            else
                for idx_seed = 1:length(batchsimCfg.seed_list)
                    my_seed = batchsimCfg.seed_list(idx_seed);
                    SysPar.SimCtrl.seed = my_seed;
                    SysPar.SimCtrl.batchsim.save_results = 1;
                    SysPar.SimCtrl.batchsim.save_results_short = 1;
                    subscenario_name = sprintf('scenario_TC%d___seed_%d___SNR_%2.2f',caseNum, my_seed,SysPar.Chan{1}.SNR);
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
                    % enable logging tx Xtf into TV
                    SysPar.SimCtrl.enable_snapshot_gNB_UE_into_SimCtrl = 1;
                    SysPar.SimCtrl.genTV.enable_logging_tx_Xtf = 1;
                    SysPar.SimCtrl.genTV.enable_logging_carrier_and_channel_info = 1;
                    % force PUCCH payload
                    if batchsimCfg.cfgMode==2
                        if pucchTestMode == 2 || pucchTestMode == 7 % NACK to ACK
                            SysPar.SimCtrl.force_pucch_payload_as = 'zeros'; % in the latest FAPI version, NACK is denoted by 0; ACK is denoted by 1. Note that it was flipped in prev. versions.
                        elseif pucchTestMode == 3 || pucchTestMode == 8 % ACK miss detection
                            SysPar.SimCtrl.force_pucch_payload_as = 'ones';
                        end
                    else
                        if pucchTestMode == 2 || pucchTestMode == 7 % NACK to ACK
                            SysPar.SimCtrl.force_pucch_payload_as = 'ones'; 
                        elseif pucchTestMode == 3 || pucchTestMode == 8 % ACK miss detection
                            SysPar.SimCtrl.force_pucch_payload_as = 'zeros';
                        end
                    end
                    if (SysPar.Chan{1}.SNR == 1000) || (batchsimCfg.cfgMode==2)
                        SysPar.SimCtrl.N_frame = 1;
                        SysPar.SimCtrl.N_slot_run = 0;
                        SysPar.SimCtrl.genTV.enable = 1;
                        SysPar.SimCtrl.genTV.tvDirName = subscenario_folder_name;
                        SysPar.SimCtrl.genTV.FAPI = 1;
                        if batchsimCfg.cfgMode==2
                            SysPar.SimCtrl.genTV.TVname = 'e2e_cfm_pucch';
                            SysPar.SimCtrl.genTV.slotIdx = [4, 5, 14, 15];
                            SysPar.SimCtrl.genTV.launchPattern = 1;
                            SysPar.SimCtrl.genTV.FAPIyaml = 1;
                        else
                            SysPar.SimCtrl.genTV.TVname = 'TV';
                            SysPar.SimCtrl.genTV.slotIdx = 0;
                        end
                    end  
                    if snr_offset == 1000
                        SNR_range_for_cuPHY = CFG{idxSet, 7}+SNRoffset(SNRoffset<1000);
                        str_SNR_range_for_cuPHY = sprintf('%2.2f,', SNR_range_for_cuPHY);
                        str_SNR_range_for_cuPHY = str_SNR_range_for_cuPHY(1:end-1); % remove the last comma
                        output_infor_for_cuphy = [output_infor_for_cuphy;{sprintf('TV_subfolder_name=%s, pucchFormat=%d, pucchTestMode=%d, SNR_range=[%s], N_slots=%d;\n',subscenario_name, pucchFormat, pucchTestMode, str_SNR_range_for_cuPHY, Nframe*10*2^(SysPar.carrier.mu))}];
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
                    end

                end
                continue;
            end
            SimCtrl = SysPar.SimCtrl;
                        
            results = SimCtrl.results;
            pucch = results.pucch;
            totalCnt = 0;
            totalUciCnt = 0;
            missack = 0;
            nack2ack = 0;
            falseCnt = 0;
            uciErr = 0;
            errorCnt = 0;
            snrdBVec = [];
            taEstMicroSecVec = [];
            
            for idxPucch = 1:Npucch
                totalCnt = pucch{idxPucch}.totalCnt + totalCnt;
                totalUciCnt = pucch{idxPucch}.totalUciCnt + totalUciCnt;
                missack = pucch{idxPucch}.missack + missack;
                nack2ack = pucch{idxPucch}.nack2ack + nack2ack;
                falseCnt = pucch{idxPucch}.falseCnt + falseCnt;
                uciErr = pucch{idxPucch}.errorCnt + uciErr;
                snrdBVec = [snrdBVec, pucch{idxPucch}.snrdB];
                taEstMicroSecVec = [taEstMicroSecVec, pucch{idxPucch}.taEstMicroSec];
            end
            
            switch pucchTestMode 
                case {1, 6}
                    errorCnt = falseCnt;
                    Perr(idxSNR) = errorCnt/totalCnt;
                case {2, 7}
                    errorCnt = nack2ack;
                    Perr(idxSNR) = errorCnt/totalCnt;
                case {3, 8}
                    errorCnt = missack;
                    Perr(idxSNR) = errorCnt/totalCnt;
                case 4
                    errorCnt = uciErr;
                    Perr(idxSNR) = errorCnt/totalUciCnt;
                    totalCnt = totalUciCnt
            end
    
            toErrSec(idxSNR) = sqrt(mean(abs(taEstMicroSecVec*1e-6 - SysPar.Chan{1}.delay).^2));
            sinrdB(idxSNR) = sqrt(mean(abs(snrdBVec - SysPar.Chan{1}.SNR).^2));
            
            
            fprintf('%4d  % 4d   %4d   %4d   %15s  %4d    %4d   %5.1f     %4.1f   %4d   %4.1f   %5d   %4d   %6.4f  %4.2f  %4.2f\n',...
                CFG{idxSet, 1}, Nframe, CFG{idxSet, 2}, CFG{idxSet, 3}, ...
                CFG{idxSet, 4}, CFG{idxSet, 5}, CFG{idxSet, 6}, ...
                CFG{idxSet, 7}, snr_offset, CFG{idxSet, 8}, 1e6*CFG{idxSet, 9}, totalCnt, errorCnt, Perr(idxSNR),...
                toErrSec(idxSNR)*1e6, sinrdB(idxSNR));
        end
        if batchsimCfg.batchsimMode>0
            continue;
        end
        P_err(idxSet,:) = Perr;
        to_ErrSec(idxSet,:) = toErrSec;
        sinr_dB(idxSet,:) = sinrdB;
    end
end

if batchsimCfg.batchsimMode==0
    fprintf('----------------------------------------------------------------------------------------------------------------------------\n');
    nPerf = 0;
    nFail = 0;
    for idxSet = 1:NallTest
        caseNum = CFG{idxSet, 1};
        if ismember(caseNum, TcToTest)
            nPerf = nPerf + 1;
            if pucchFormat == 3
                allowed_err_rate = 0.1; % relax from 0.01 to 0.1
            elseif pucchFormat == 2
                if pucchTestMode == 1
                    allowed_err_rate = 0.1; % relax from 0.01 to 0.1
                else
                    allowed_err_rate = 0.1; % relax from 0.01 to 0.1
                end
            else
                if pucchTestMode == 2 || pucchTestMode == 7
                    allowed_err_rate = 0.001;
                elseif pucchFormat == 0 && (pucchTestMode == 3 || pucchTestMode == 8) %format 0 ack missed
                    allowed_err_rate = 0.02; % relax from 0.01 to 0.02
                else
                    allowed_err_rate = 0.01;
                end
            end
            if P_err(idxSet,end) > allowed_err_rate
                nFail = nFail + 1;
            end
            
            if (pucchTestMode ~= 1 && pucchTestMode ~= 6) && sum(isnan(to_ErrSec(idxSet,:))) > 0
                nFail = nFail + 1;
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
    fprintf('Total number of subcenarios generated: %d\n',num_generated_subscenarios);
end
toc; 
fprintf('\n');
return
