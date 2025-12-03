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

function CFG = genPxschUegCfg(cfgUeg)
%GENERATE_pxsch_CFG_MATLAB Generate pxsch configuration cell array for MATLAB
%   Returns a cell array CFG, each row is a configuration for a UE.
%   
%   Input: cfgUeg - struct with the following fields:
%     REQUIRED FIELDS:
%       pxsch_cfg_idx: starting index for the pxsch configuration table
%       nUeg: number of User Equipment Groups (UEGs)
%       nUePerUeg: number of UEs per UEG
%       startRnti: starting RNTI value for the first UE
%       startPrb: first PRB index to allocate
%       endPrb: last PRB index to allocate (inclusive)
%       
%     OPTIONAL FIELDS (with defaults):
%       diffscId: (default false) if false, all UEs in a UEG have SCID=0, port0 increments by nl
%                 if true, first half of UEs in a UEG have SCID=0, second half SCID=1
%       nl: (default 2) number of layers per UE
%       dmrsMaxLen: (default 2) DMRS maximum length
%       mcsIdx: (default 27) MCS index to use for all UEs
%       sym0: (default 0) starting symbol index
%       Nsym: (default 14) number of symbols
%       BWP0: (default 0) starting BWP index
%       nBWP: (default 273) number of BWPs
%       prgSize: (default 2) PRG size, only used to calculate default prb allocation
%       prbAlloc: (default []) vector of PRB counts for each UEG (length nUeg)
%                 If empty, PRBs are divided as evenly as possible among UEGs.
%
%   Example usage:
%     cfgUeg = struct();
%     cfgUeg.pxsch_cfg_idx = 2001;
%     cfgUeg.nUeg = 8;
%     cfgUeg.nUePerUeg = 8;
%     cfgUeg.startRnti = 7;
%     cfgUeg.startPrb = 20;
%     cfgUeg.endPrb = 272;
%     cfgUeg.diffscId = true;
%     CFG = genPxschUegCfg(cfgUeg);

    % Set default values for optional fields
    if ~isfield(cfgUeg, 'diffscId'), cfgUeg.diffscId = false; end
    if ~isfield(cfgUeg, 'nl'), cfgUeg.nl = 2; end
    if ~isfield(cfgUeg, 'dmrsMaxLen'), cfgUeg.dmrsMaxLen = 2; end
    if ~isfield(cfgUeg, 'mcsIdx'), cfgUeg.mcsIdx = 27; end
    if ~isfield(cfgUeg, 'sym0'), cfgUeg.sym0 = 0; end
    if ~isfield(cfgUeg, 'Nsym'), cfgUeg.Nsym = 14; end
    if ~isfield(cfgUeg, 'BWP0'), cfgUeg.BWP0 = 0; end
    if ~isfield(cfgUeg, 'nBWP'), cfgUeg.nBWP = 273; end
    if ~isfield(cfgUeg, 'prgSize'), cfgUeg.prgSize = 2; end
    if ~isfield(cfgUeg, 'prbAlloc'), cfgUeg.prbAlloc = []; end
    
    % Validate required fields
    required_fields = {'pxsch_cfg_idx', 'nUeg', 'nUePerUeg', 'startRnti', 'startPrb', 'endPrb'};
    for i = 1:length(required_fields)
        if ~isfield(cfgUeg, required_fields{i})
            error('Required field %s is missing from cfgUeg struct', required_fields{i});
        end
    end
    
    % Extract values from struct for easier access
    pxsch_cfg_idx = cfgUeg.pxsch_cfg_idx;
    nUeg = cfgUeg.nUeg;
    nUePerUeg = cfgUeg.nUePerUeg;
    startRnti = cfgUeg.startRnti;
    startPrb = cfgUeg.startPrb;
    endPrb = cfgUeg.endPrb;
    diffscId = cfgUeg.diffscId;
    nl = cfgUeg.nl;
    dmrsMaxLen = cfgUeg.dmrsMaxLen;
    mcsIdx = cfgUeg.mcsIdx;
    sym0 = cfgUeg.sym0;
    Nsym = cfgUeg.Nsym;
    BWP0 = cfgUeg.BWP0;
    nBWP = cfgUeg.nBWP;
    prgSize = cfgUeg.prgSize;
    prbAlloc = cfgUeg.prbAlloc;

    total_ues = nUeg * nUePerUeg;
    if isempty(prbAlloc)
        total_prbs = endPrb - startPrb + 1;
        if (mod(startPrb, prgSize) ~= 0)
            updated_total_prbs = total_prbs - mod(startPrb, prgSize);
        else
            updated_total_prbs = total_prbs;
        end
        base_prb_per_ueg = floor(updated_total_prbs / nUeg / prgSize) * prgSize;
        leftover_rbs = updated_total_prbs - base_prb_per_ueg * nUeg;
        leftover_ues_oneMorePrg = floor(leftover_rbs / prgSize);
        prbAlloc = base_prb_per_ueg * ones(1, nUeg);
        prbAlloc(1) = prbAlloc(1) + total_prbs - updated_total_prbs;  % add the difference to the first UEG to align with prgSize
        prbAlloc(nUeg - leftover_ues_oneMorePrg + 1 : nUeg) = prbAlloc(end) + prgSize;  % add remainging PRBs to the last UEG
        prbAlloc(nUeg) = prbAlloc(nUeg) + leftover_rbs - leftover_ues_oneMorePrg * prgSize;
    end

    % Calculate PRB offset for each UEG (all UEs in same UEG share allocation)
    rb0_list = zeros(1, nUeg);
    rb0 = startPrb;
    for ueg_idx = 1:nUeg
        rb0_list(ueg_idx) = rb0;
        rb0 = rb0 + prbAlloc(ueg_idx);
    end

    % Build the CFG cell array
    CFG = cell(total_ues, 21);
    for i = 1:total_ues
        ueg_idx = floor((i-1)/nUePerUeg) + 1;
        ue_in_ueg = mod(i-1, nUePerUeg);

        % SCID and port0 logic
        if diffscId
            half = min(8, floor(nUePerUeg/2));
            if mod(ue_in_ueg, 16) < half
                scid = 0;
            else
                scid = 1;
            end
            port0 = mod(mod(ue_in_ueg, 16), half) * nl;
        else
            scid = 0;
            port0 = ue_in_ueg * nl;
            if (port0 > 8)  % ensure port0 does not exceeds 8
                error("genPxschUegCfg:Port0ExceedsLimit", ...
                      "port0 = %d exceeds limit 8; set diffscId = true and retry.\n", port0);
            end
        end

        % Use UEG-level PRB allocation (all UEs in same UEG share allocation)
        rb0_val = rb0_list(ueg_idx);
        nrb = prbAlloc(ueg_idx);

        % Fill the row (matching your Python output order)
        CFG{i,1}  = pxsch_cfg_idx + (i-1); % TC#
        CFG{i,2}  = 1;                     % mcsTable
        CFG{i,3}  = mcsIdx;                % mcs
        CFG{i,4}  = nl;                    % nl
        CFG{i,5}  = rb0_val;               % rb0
        CFG{i,6}  = nrb;                   % Nrb
        CFG{i,7}  = sym0;                  % sym0
        CFG{i,8}  = Nsym;                  % Nsym
        CFG{i,9}  = scid;                  % SCID
        CFG{i,10} = BWP0;                  % BWP0
        CFG{i,11} = nBWP;                  % nBWP
        CFG{i,12} = startRnti + (i-1);     % RNTI
        CFG{i,13} = 0;                     % rvIdx
        CFG{i,14} = 41;                    % dataScId
        CFG{i,15} = 2;                     % dmrs0
        CFG{i,16} = dmrsMaxLen;            % maxLen
        CFG{i,17} = 1;                     % addPos
        CFG{i,18} = 41;                    % dmrsScId
        CFG{i,19} = 2;                     % nCdm
        CFG{i,20} = port0;                 % port0
        CFG{i,21} = ueg_idx-1;             % idxUeg (0-based)
    end
end 