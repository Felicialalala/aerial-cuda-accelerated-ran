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

function SysPar = initSysPar(varargin)
% function SysPar = initSysPar(varargin)
%
% This function initializes all configurations inside SysPar, which includes
% simulation controller (SimCtrl), network configurations (carrier), allocation
% related configurations (such as prach), and channel configurations (Chan).
% SysPar is the only config variable that we need to modify to run all the
% simulations.
%
% Input:    Reserved
%
% Output:   SysPar: structure with all simulation configurations
%

if nargin == 0
    testAlloc.dl = 1;
    testAlloc.ul = 0;
    testAlloc.ssb = 1;
    testAlloc.pdcch = 1;
    testAlloc.pdsch = 1;
    testAlloc.csirs = 1;
    testAlloc.prach = 0;
    testAlloc.pucch = 0;
    testAlloc.pusch = 0;
    testAlloc.srs = 0;
    testAlloc.bfw = 0;
else
    inputAlloc = varargin{1};
    if isfield(inputAlloc, 'dl')
        testAlloc.dl = inputAlloc.dl;
    else
        testAlloc.dl = 0;
    end        
    if isfield(inputAlloc, 'ul')
        testAlloc.ul = inputAlloc.ul;
    else
        testAlloc.ul = 0;
    end 
    if isfield(inputAlloc, 'ssb')
        testAlloc.ssb = inputAlloc.ssb;
    else
        testAlloc.ssb = 0;
    end 
    if isfield(inputAlloc, 'pdcch')
        testAlloc.pdcch = inputAlloc.pdcch;
    else
        testAlloc.pdcch = 0;
    end 
    if isfield(inputAlloc, 'pdsch')
        testAlloc.pdsch = inputAlloc.pdsch;
    else
        testAlloc.pdsch = 0;
    end 
    if isfield(inputAlloc, 'csirs')
        testAlloc.csirs = inputAlloc.csirs;
    else
        testAlloc.csirs = 0;
    end 
    if isfield(inputAlloc, 'prach')
        testAlloc.prach = inputAlloc.prach;
    else
        testAlloc.prach = 0;
    end 
    if isfield(inputAlloc, 'pucch')
        testAlloc.pucch = inputAlloc.pucch;
    else
        testAlloc.pucch = 0;
    end 
    if isfield(inputAlloc, 'pusch')
        testAlloc.pusch = inputAlloc.pusch;
    else
        testAlloc.pusch = 0;
    end     
    if isfield(inputAlloc, 'srs')
        testAlloc.srs = inputAlloc.srs;
    else
        testAlloc.srs = 0;
    end
    if isfield(inputAlloc, 'bfw')
        testAlloc.bfw = inputAlloc.bfw;
    else
        testAlloc.bfw = 0;
    end
end

SysPar.testAlloc = testAlloc;

% carrier related config
SysPar.carrier = cfgCarrier();

% ssb related config
SysPar.ssb = cfgSsb();

% pdcch related config
SysPar.pdcch{1} = cfgPdcch();

% pdsch related config
SysPar.pdsch{1} = cfgPdsch();

% CSI-RS related config
SysPar.csirs{1} = cfgCsirs();

% prach related config
SysPar.prach{1} = cfgPrach();

% pucch related config
SysPar.pucch{1} = cfgPucch();

% srs related config
SysPar.srs{1} = cfgSrs();

% bfw related config
SysPar.bfw{1} = cfgBfw();

% channel related config
SysPar.Chan{1} = cfgChan();
% set interference channel
SysPar.Interf_Chan{1} = cfgInterferenceChan();

% controller for simulation code coverage
cfgSimCov();

% controller for simulation setup
SysPar.SimCtrl = cfgSimCtrl();

% RF impairments modeling
SysPar.RF = cfgRF();

% pusch related config
SysPar.pusch{1} = cfgPusch(SysPar.SimCtrl);

return
