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

function Xtf = dlTxBf(Xtf, pduList, srsState, pdschTable, Chan_DL, SimCtrl)
    % ULRXBF Perform uplink receive beamforming for allocated resources per UE group
    % Inputs:
    %   Xtf - Input time-frequency grid with noise
    %   Xtf_noNoise - Input time-frequency grid without noise
    %   pduList - List of PDSCH PDUs
    %   srsState - SRS state
    %   pdschTable - PDSCH configuration table
    %   Chan_DL - Channel information
    %   SimCtrl - Simulation control parameters
    % Outputs:
    %   Xtf - Beamformed time-frequency grid with noise
    %   Xtf_noNoise - Beamformed time-frequency grid without noise
    % Notes:
    %   - This function performs beamforming for each UEG (User Equipment Group)
    %     based on the PDSCH allocation parameters.
    %   - It supports both channel-based and SRS-based beamforming.
    %   - The beamforming weights are computed using either the estimated
    %     channel or the genie channel.
    

    % Return if beamforming is not enabled
    if ~SimCtrl.enableDlTxBf
        return;
    end

    % Initialize UEG structure
    nPdu = length(pduList);
    UegList = {};
    currentUeg = 0;
    
    % extract BWPSize and prgSize from first PDSCH PDU
    BWPSize = 273;
    prgSize = 2;
    for idxPdu = 1:nPdu
        if strcmp(pduList{idxPdu}.type, 'pdsch')
            BWPSize = pduList{idxPdu}.BWPSize;
            prgSize = pduList{idxPdu}.prgSize;
            break;
        end
    end
    
    % Construct UEG groups
    for idxPdu = 1:nPdu
        pdu = pduList{idxPdu};
        if strcmp(pdu.type, 'pdsch')  % Ensure we're only processing PDSCH PDUs
            if pdu.idxUeg == currentUeg
                % Initialize UegList entry if it doesn't exist
                if length(UegList) < currentUeg + 1 || isempty(UegList{currentUeg+1})
                    UegList{currentUeg+1}.idxPdu = [];
                    UegList{currentUeg+1}.nlUeg = 0;
                    UegList{currentUeg+1}.nPortPerUe = []; % Initialize nPort array
                end
                UegList{currentUeg+1}.idxPdu = [UegList{currentUeg+1}.idxPdu, idxPdu-1];
                UegList{currentUeg+1}.nlUeg = UegList{currentUeg+1}.nlUeg + pdu.nrOfLayers;

                % Derive and store nPort for this PDU
                dmrs  = derive_dmrs_main(pdu, pdschTable);
                alloc = derive_alloc_main(pdu, dmrs);
                UegList{currentUeg+1}.nPortPerUe = [UegList{currentUeg+1}.nPortPerUe, length(alloc.portIdx)];
            else
                % Initialize UegList entry for the new UEG
                currentUeg = currentUeg + 1;

                % Derive and store nPort for this PDU
                dmrs  = derive_dmrs_main(pdu, pdschTable);
                alloc = derive_alloc_main(pdu, dmrs);

                UegList{currentUeg+1}.idxPdu = [idxPdu-1];
                UegList{currentUeg+1}.nlUeg = pdu.nrOfLayers;
                UegList{currentUeg+1}.nPortPerUe = [length(alloc.portIdx)]; % Initialize nPort array
            end
        end
    end
    
    % Calculate total ports and initialize output matrices
    [nRe, nSym, nAnt] = size(Xtf);
    totalPorts = 0;
    for idxUeg = 1:length(UegList)
        totalPorts = totalPorts + sum(UegList{idxUeg}.nPortPerUe);
    end
    
    % Process each UEG
    Xtf1 = reshape(Xtf, nRe*nSym, nAnt);
    % Xtf1_noNoise = reshape(Xtf_noNoise, nRe*nSym, nAnt);
    
    for idxUeg = 1:length(UegList)
        % Get PDSCH parameters for this UEG
        idxPdu = UegList{idxUeg}.idxPdu(1) + 1;  % Get first PDU in UEG
        pdu = pduList{idxPdu};
        
        % Extract allocation parameters
        startSymbol = pdu.StartSymbolIndex;
        nSymbols = pdu.NrOfSymbols;
        startPrb = pdu.rbStart;
        nPrb = pdu.rbSize;
        
        % Calculate RE indices for the assigned resources
        reStart = startPrb * 12 + 1;  % Convert PRB to RE index
        reEnd = reStart + nPrb * 12 - 1;
        symStart = startSymbol + 1;  % Convert to 1-based indexing
        symEnd = symStart + nSymbols - 1;
        
        % Create mask for assigned REs
        reMask = zeros(nRe, nSym);
        reMask(reStart:reEnd, symStart:symEnd) = 1;
        
        if SimCtrl.enableSrsState == 0
            % Genie channel-based beamforming, wideband
            H_ueg = [];
            nAnt_ueg = zeros(length(UegList{idxUeg}.idxPdu), 1);
            nPort_ueg = zeros(length(UegList{idxUeg}.idxPdu), 1);
            
            % Pack all UE's channel matrices in the UEG
            for idxUe = 1:length(UegList{idxUeg}.idxPdu)
                idxPdu = UegList{idxUeg}.idxPdu(idxUe) + 1;
                pdu = pduList{idxPdu}; % Get the current PDU

                % Read Genie channel for current UE 
                H_ue = Chan_DL{pdu.idxUE+1}.chanMatrix.';
                H_ueg = [H_ueg, H_ue];

                % Derive necessary parameters (dmrs and alloc) to get portIdx length
                dmrs  = derive_dmrs_main(pdu, pdschTable);
                alloc = derive_alloc_main(pdu, dmrs);
                nAnt_ueg(idxUe) = size(H_ue, 2);
                nPort_ueg(idxUe) = length(alloc.portIdx); % Use derived portIdx length
            end
            
            % Calculate BFW using Zero-forcing algorithm
            W_ueg = inv(H_ueg'*H_ueg)*H_ueg';
            
            % Apply beamforming to assigned REs
            % Map antennas to layers
            % UE 1's layers from antIndex: 1:nPort_ueg(1)
            % UE 2's layers from antIndex: nAnt_ueg(1)+ (1:nPort_ueg(2))
            % UE 3's layers from antIndex: nAnt_ueg(1)+nAnt_ueg(2)+ (1:nPort_ueg(3))
            % ...
            % UE n's layers from antIndex: sum(nAnt_ueg(1:n-1))+ (1:nPort_ueg(n))
            tmpXtf = Xtf1(reMask(:) == 1, :);
            % tmpXtf_noNoise = Xtf1_noNoise(reMask(:) == 1, :);
            for idxUe = 1:length(UegList{idxUeg}.idxPdu)
                antIdx = sum(nAnt_ueg(1:idxUe-1)) + (1:nPort_ueg(idxUe));
                portIdx = sum(nPort_ueg(1:idxUe-1)) + (1:nPort_ueg(idxUe));
                W_ue = W_ueg(antIdx, :).';
                
                % Apply beamforming only to assigned REs
                Xtf1(reMask(:) == 1, portIdx) = tmpXtf * W_ue;
                
                % Same for no-noise version
                % Xtf1_noNoise(reMask(:) == 1, portIdx) = tmpXtf_noNoise * W_ue;
            end
        else
            % SRS-based beamforming using the estimated channel on PRG level
            H_ueg = [];
            nAnt_ueg = [];
            nPort_ueg = [];
            p3 = 0;
            % Pack all UE's channel matrices in the UEG
            for idxUe = 1:length(UegList{idxUeg}.idxPdu)
                idxPdu = UegList{idxUeg}.idxPdu(idxUe) + 1;
                pdu = pduList{idxPdu}; % Get the current PDU
                
                % read chEst from current UE
                RNTI = pdu.RNTI;
                if SimCtrl.BfKnownChannel
                    srsOutput = srsState.(['rnti_',num2str(RNTI)]).srsOutput_noNoise;
                else
                    srsOutput = srsState.(['rnti_',num2str(RNTI)]).srsOutput;
                end
                [n_prg, n_rxAnt, n_txAnt] = size(srsOutput.Hest);
                H_ueg(:,:,(p3 + 1):(p3 + n_txAnt)) = srsOutput.Hest;
                p3 = p3 + n_txAnt;

                % Derive necessary parameters (dmrs and alloc) to get portIdx length
                dmrs  = derive_dmrs_main(pdu, pdschTable);
                alloc = derive_alloc_main(pdu, dmrs);
                nAnt_ueg(idxUe) = size(srsOutput.Hest, 3);
                nPort_ueg(idxUe) = length(alloc.portIdx); % Use derived portIdx length
            end

            nPrg = ceil(BWPSize / prgSize);

            % Get the allocated PRBs for this UE
            allocatedPrbs = startPrb + (0:nPrb-1) + 1;  % Create array of allocated PRB indices
            allocatedPrgs = unique(ceil(allocatedPrbs / prgSize));  % Convert PRBs to PRGs

            % Pre-compute beamforming weights for each PRG
            
            for idxPrg = allocatedPrgs
                % Get RE indices for this PRG
                startRe = (idxPrg-1)*prgSize*12 + 1;
                endRe = min(idxPrg*prgSize*12, BWPSize*12);
                
                % Calculate beamforming weights for this PRG
                H_prg = squeeze(H_ueg(idxPrg, :, :));
                W_prg = inv(H_prg'*H_prg)*H_prg';
                
                % Apply beamforming to only allocated REs
                % Map antennas to layers
                % UE 1's layers from antIndex: 1:nPort_ueg(1)
                % UE 2's layers from antIndex: nAnt_ueg(1)+ (1:nPort_ueg(2))
                % UE 3's layers from antIndex: nAnt_ueg(1)+nAnt_ueg(2)+ (1:nPort_ueg(3))
                % ...
                % UE n's layers from antIndex: sum(nAnt_ueg(1:n-1))+ (1:nPort_ueg(n))
                reForThisPrg = repmat(startRe:endRe, 1, nSymbols) + repelem((symStart-1:symEnd-1)*BWPSize*12, 1, endRe-startRe+1);
                tmpXtf = Xtf1(reForThisPrg, :);
                Xtf1(reForThisPrg, :) = 0;
                % tmpXtf_noNoise = Xtf1_noNoise(reForThisPrg, :);
                for idxUe = 1:length(UegList{idxUeg}.idxPdu)
                    antIdx = sum(nAnt_ueg(1:idxUe-1)) + (1:nPort_ueg(idxUe));
                    portIdx = sum(nPort_ueg(1:idxUe-1)) + (1:nPort_ueg(idxUe));
                    W_ue = W_prg(antIdx, :).';
                    
                    % Apply beamforming only to assigned REs
                    % Xtf1(reForThisPrg, portIdx) = tmpXtf * W_ue;
                    Xtf1(reForThisPrg, :) = tmpXtf(:, portIdx) * W_ue.' + Xtf1(reForThisPrg, :);
                    
                    % Same for no-noise version
                    % Xtf1_noNoise(reForThisPrg, portIdx) = tmpXtf_noNoise * W_ue;
                end
            end
        end
    end
    % reduce the size of Xtf1 and Xtf1_noNoise to SimCtrl.CellConfigPorts
    % Xtf1 = Xtf1(:, 1:SimCtrl.CellConfigPorts);
    % Xtf1_noNoise = Xtf1_noNoise(:, 1:SimCtrl.CellConfigPorts);
    % Reshape back to original dimensions
    Xtf = reshape(Xtf1, nRe, nSym, n_rxAnt) * sqrt(n_rxAnt);
    % Xtf_noNoise = reshape(Xtf1_noNoise, nRe, nSym, SimCtrl.CellConfigPorts);
end