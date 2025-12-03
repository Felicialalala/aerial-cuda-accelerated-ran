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

%% Prepare PXSCH configs
% can be generated or directly read from ULMIX/DLMIX
CFG_PXSCH = [];

% Base configuration struct
baseCfgUeg = struct();
baseCfgUeg.startRnti = 7;
baseCfgUeg.mcsIdx = 27;
baseCfgUeg.sym0 = 0;
baseCfgUeg.Nsym = 14;
baseCfgUeg.BWP0 = 0;
baseCfgUeg.nBWP = 273;
baseCfgUeg.prgSize = 2;
baseCfgUeg.diffscId = false;
baseCfgUeg.nl = 2;
baseCfgUeg.dmrsMaxLen = 2;
baseCfgUeg.pxsch_cfg_idx = 2001;  % cfg index does not matter for BFW binding config

% 64TR MU-MIMO, 16 UEGs, 4 UE per UEG, 2 layers per UE, PUSCH prb 22~272
cfgUeg = baseCfgUeg;
cfgUeg.nUeg = 16;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 22;
cfgUeg.endPrb = 272;
cfgUeg.prbAlloc = [8,12,16,18,20,22,24,22,20,18,16,14,12,10,8,11];
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PXSCH = [CFG_PXSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UE per UEG, 2 layers per UE, PDSCH prb 20~272, OFDM symbol 1~13, 100 MHz
cfgUeg = baseCfgUeg;
cfgUeg.nUeg = 8;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 20;
cfgUeg.endPrb = 272;
cfgUeg.diffscId = true;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PXSCH = [CFG_PXSCH; CFG_PDSCH_temp];

% 64TR MU-MIMO, 16 UEGs, 4 UE per UEG, 2 layers per UE, PUSCH prb 6~224
cfgUeg = baseCfgUeg;
cfgUeg.nUeg = 16;
cfgUeg.nUePerUeg = 4;
cfgUeg.startPrb = 6;
cfgUeg.endPrb = 224;
cfgUeg.prbAlloc = [6,10,14,16,18,20,22,20,18,16,14,12,10,8,6,9];
CFG_PUSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PXSCH = [CFG_PXSCH; CFG_PUSCH_temp];

% 64TR MU-MIMO, 8 UEGs, 8 UE per UEG, 2 layers per UE, PDSCH prb  0~272, OFDM symbol 1~13, 100 MHz
cfgUeg = baseCfgUeg;
cfgUeg.nUeg = 8;
cfgUeg.nUePerUeg = 8;
cfgUeg.startPrb = 0;
cfgUeg.endPrb = 272;
cfgUeg.diffscId = true;
cfgUeg.sym0 = 1;
cfgUeg.Nsym = 13;
CFG_PDSCH_temp = genPxschUegCfg(cfgUeg);
CFG_PXSCH = [CFG_PXSCH; CFG_PDSCH_temp];

%% Prepare BFW binding configuration
% ueSlotMap: [64 64] means CFG_PXSCH(1:64) and CFG_PXSCH(65:128) are two groups
% tvIdx: [1000 2000] means first group starts at TV index 1000, second at 2000
% srsBufferIdx: [25 26] means SRS buffer indices for each group
ueSlotMap = [64 64 64 64];  % Two groups with 64 UEs each
tvIdx = [9381 9382 9383 9384];  % Starting TV indices for each group
srsBufferIdx = [23 23 23 23];  % SRS buffer indices for each group
UE_GRPS_CFG = genBfwSrsBinding(CFG_PXSCH, ueSlotMap, 64, 217, tvIdx, srsBufferIdx);

function UE_GRPS_CFG = genBfwSrsBinding(CFG_PXSCH, ueSlotMap, bsAnt, startCfgIdx, tvIdx, srsBufferIdx)
% GENBFWBINDING Generate BFW binding configurations from CFG_PXSCH
%
% Inputs:
%   CFG_PXSCH   - Cell array where each row represents a UE configuration
%                 Column 4: layers per UE
%                 Column 5: start PRB
%                 Column 6: nPRB  
%                 Column 12: UE index
%                 Column 21: UE group index
%   ueSlotMap   - Array specifying number of UEs per group/slot
%                 e.g., [64 64] means two groups with 64 UEs each
%   bsAnt       - Number of base station antennas (e.g., 32 or 64)
%   startCfgIdx - Starting configuration index for UE groups
%   tvIdx  - Array of starting TV indices for each group (should match length of ueSlotMap)
%   srsBufferIdx - Array of SRS buffer indices for each group (should match length of ueSlotMap)
%
% Output:
%   UE_GRPS_CFG - Cell array with BFW configuration in format:
%                 {UEGRP#, startPrb, nPrb, nRxAnt, nBfLayers, ueIdxs, ueLayersIdxs, tvIdx}

% Validate inputs
totalUEs = sum(ueSlotMap);
numCfgRows = size(CFG_PXSCH, 1);

if numCfgRows ~= totalUEs
    error('CFG_PXSCH rows (%d) does not equal sum of ueSlotMap (%d)', numCfgRows, totalUEs);
end

if length(tvIdx) ~= length(ueSlotMap)
    error('tvIdx length (%d) must match ueSlotMap length (%d)', length(tvIdx), length(ueSlotMap));
end

if length(srsBufferIdx) ~= length(ueSlotMap)
    error('srsBufferIdx length (%d) must match ueSlotMap length (%d)', length(srsBufferIdx), length(ueSlotMap));
end

% Extract relevant columns from CFG_PXSCH
layersPerUe = cell2mat(CFG_PXSCH(:, 4));
startPrbs = cell2mat(CFG_PXSCH(:, 5));
nPrbs = cell2mat(CFG_PXSCH(:, 6));
ueIndices = cell2mat(CFG_PXSCH(:, 12));
ueGroupIndices = cell2mat(CFG_PXSCH(:, 21));

% Initialize output cell array
UE_GRPS_CFG = cell(0, 8);  % Added one more column for tvIdx
configIdx = startCfgIdx;

% Process each group based on ueSlotMap
rowStart = 1;
for groupIdx = 1:length(ueSlotMap)
    numUEsInSlot = ueSlotMap(groupIdx);
    rowEnd = rowStart + numUEsInSlot - 1;
    
    % Extract current group's data
    groupCfg = CFG_PXSCH(rowStart:rowEnd, :);
    groupLayersPerUe = layersPerUe(rowStart:rowEnd);
    groupStartPrbs = startPrbs(rowStart:rowEnd);
    groupNPrbs = nPrbs(rowStart:rowEnd);
    groupUeIndices = ueIndices(rowStart:rowEnd);
    groupUeGroupIndices = ueGroupIndices(rowStart:rowEnd);
    
    % Find unique UE groups within this slot
    uniqueGroups = unique(groupUeGroupIndices);
    numUeGroups = length(uniqueGroups);
    
    % Store group configurations for this slot
    groupConfigs = [];
    
    % Process each UE group within this slot
    for i = 1:numUeGroups
        ueGroupIdx = uniqueGroups(i);
        
        % Find UEs belonging to this UE group within the current slot
        uesInGroup = find(groupUeGroupIndices == ueGroupIdx);
        numUesInGroup = length(uesInGroup);
        
        % Calculate group parameters
        groupStartPrb = min(groupStartPrbs(uesInGroup));
        groupEndPrb = max(groupStartPrbs(uesInGroup) + groupNPrbs(uesInGroup) - 1);
        groupNPrb = groupEndPrb - groupStartPrb + 1;
        
        % Get layers per UE for this group
        ueLayersCount = groupLayersPerUe(uesInGroup);
        totalLayers = sum(ueLayersCount);
        
        % Create UE indices and layer mapping
        ueIdxs = [];
        ueLayersIdxs = [];
        
        for j = 1:numUesInGroup
            ueIdx = groupUeIndices(uesInGroup(j)) - 1; % Convert to 0-based indexing
            numLayers = ueLayersCount(j);
            
            % Repeat UE index for each layer
            ueIdxs = [ueIdxs, repmat(ueIdx, 1, numLayers)];
            
            % Create layer indices for this UE (0-based)
            ueLayersIdxs = [ueLayersIdxs, 0:(numLayers-1)];
        end
        
        % Add to output configuration
        currentRow = size(UE_GRPS_CFG, 1) + 1;
        UE_GRPS_CFG{currentRow, 1} = configIdx;              % UEGRP#
        UE_GRPS_CFG{currentRow, 2} = groupStartPrb;          % startPrb
        UE_GRPS_CFG{currentRow, 3} = groupNPrb;             % nPrb
        UE_GRPS_CFG{currentRow, 4} = bsAnt;                 % nRxAnt
        UE_GRPS_CFG{currentRow, 5} = totalLayers;           % nBfLayers
        UE_GRPS_CFG{currentRow, 6} = ueIdxs;                % ueIdxs
        UE_GRPS_CFG{currentRow, 7} = ueLayersIdxs;          % ueLayersIdxs
        
        % Store config ID for this group
        groupConfigs = [groupConfigs, configIdx];
        configIdx = configIdx + 1;
    end
    
    % Print the additional format line for this group
    fprintf('  %-7d %-7d [', tvIdx(groupIdx), numUeGroups);
    fprintf("%d:%d", min(groupConfigs), max(groupConfigs));
    fprintf(']   %-12d %-12d %-5d     %-15d %-15d\n', ...
        srsBufferIdx(groupIdx), 0, -30, 2, 2);
    
    % Move to next group
    rowStart = rowEnd + 1;
end

% Display the generated configuration
fprintf('\nGenerated UE_GRPS_CFG:\n');
fprintf('%%   UEGRP#      startPrb    nPrb      nRxAnt nBfLayers         ueIdxs               ueLayersIdxs\n');
for i = 1:size(UE_GRPS_CFG, 1)
    fprintf('    %-11d %-11d %-9d %-6d %-9d [', ...
        UE_GRPS_CFG{i,1}, UE_GRPS_CFG{i,2}, UE_GRPS_CFG{i,3}, ...
        UE_GRPS_CFG{i,4}, UE_GRPS_CFG{i,5});
    
    % Print ueIdxs with 2-character width for each number
    ueIdxs = UE_GRPS_CFG{i,6};
    ueIdxsStr = '';
    for j = 1:length(ueIdxs)
        if j == length(ueIdxs)
            ueIdxsStr = [ueIdxsStr, sprintf('%2d', ueIdxs(j))];
        else
            ueIdxsStr = [ueIdxsStr, sprintf('%2d ', ueIdxs(j))];
        end
    end
    fprintf('%-20s', ueIdxsStr);  % Minimum width of 20 characters
    fprintf(']  [');
    
    % Print ueLayersIdxs
    ueLayersIdxs = UE_GRPS_CFG{i,7};
    for j = 1:length(ueLayersIdxs)
        if j == length(ueLayersIdxs)
            fprintf('%d', ueLayersIdxs(j));
        else
            fprintf('%d ', ueLayersIdxs(j));
        end
    end
    fprintf(']\n');
end

end 