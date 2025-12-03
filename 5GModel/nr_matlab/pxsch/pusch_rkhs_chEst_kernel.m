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

function H_est = pusch_rkhs_chEst_kernel(Y, H_est, r_dmrs, compBlockPrm, ueGrpPrms, uePrms, gridPrms, push_rkhs_tables)

%%
% PARAMATERS

% User paramaters:
nUe                  = ueGrpPrms.nUe;              
numDmrsCdmGrpsNoData = ueGrpPrms.numDmrsCdmGrpsNoData;

% frequency allocation:
startInputSc   = compBlockPrm.startInputSc;
nSc            = ueGrpPrms.computeBlocksCommonPrm.nSc;
nDmrsSc        = ueGrpPrms.computeBlocksCommonPrm.nDmrsSc;
nPrb           = ueGrpPrms.computeBlocksCommonPrm.nPrb;
startScOfUeGrp = ueGrpPrms.startSc;

% input/output paramaters:
startOutputScInBlock  = compBlockPrm.startOutputScInBlock;
nOutputSc             = compBlockPrm.nOutputSc;
scOffsetIntoChEstBuff = compBlockPrm.scOffsetIntoChEstBuff;

% time allocation:
dmrsSymIdxs  = ueGrpPrms.dmrsSymIdxs;
nDmrsSym     = ueGrpPrms.nDmrsSym;

% frequency zero padding:
nZpDmrsSc   = ueGrpPrms.computeBlocksCommonPrm.nZpDmrsSc;
zpIdx       = ueGrpPrms.computeBlocksCommonPrm.zpIdx;
nCpInt      = ueGrpPrms.computeBlocksCommonPrm.nCpInt;

% noise measurment:
rolloff               = ueGrpPrms.computeBlocksCommonPrm.rolloff;
nNoiseMeasurments     = ueGrpPrms.computeBlocksCommonPrm.nNoiseMeasurments;
noiseMeasurmentMethod = ueGrpPrms.computeBlocksCommonPrm.noiseMeasurmentMethod;
hammingFlag           = push_rkhs_tables.rkhsPrms.hammingFlag;

%  MIMO dimensions:
nGnbAnt = ueGrpPrms.nGnbAnt;
nLayers = ueGrpPrms.nLayers;

% grid paramaters:
gridBitmask = ueGrpPrms.gridBitmask;
gridIdxs    = ueGrpPrms.gridIdxs;

% constants:
nEigs = push_rkhs_tables.rkhsPrms.nEigs;

% tables:
eigVecCob        = push_rkhs_tables.eigVecCobTable{nPrb};
corr             = push_rkhs_tables.corrTable{nPrb};
zpDmrsScEigenVec = push_rkhs_tables.zpDmrsScEigenVecTable{zpIdx + 1};
interpCob        = push_rkhs_tables.interpCobTable{nPrb};
zpInterpVec      = push_rkhs_tables.zpInterpVecTable{zpIdx + 1};

eigValues            = push_rkhs_tables.eigValTable{nPrb};
eigValues_rep        = reshape(eigValues,nEigs,1,1,1);
eigValues_rep        = repmat(eigValues_rep, 1,nGnbAnt,nLayers,nCpInt);
sumEigValues         = push_rkhs_tables.prbPrms{nPrb}.sumEigValues;
corr_half_nZpDmrsSc  = push_rkhs_tables.prbPrms{nPrb}.corr_half_nZpDmrsSc;

%%
% STEP 1: load eigenvectors

eigenVecs = eigVecCob * zpDmrsScEigenVec(:, 1 : nDmrsSc);

%%
% STEP 2: load data

Y_dmrsSym = zeros(nSc, nDmrsSym, nGnbAnt);
scIdxs    = startInputSc : (startInputSc + nSc - 1);

for antIdx = 0 : (nGnbAnt - 1)
    for i = 0 : (nDmrsSym - 1)
        Y_dmrsSym(:, i + 1, antIdx + 1) =  Y(scIdxs + 1, dmrsSymIdxs(i + 1) + 1, antIdx + 1);
    end
end

%%
% STEP 3: compute proj coeffs

totNoiseEnergy = 0;

projCoeffs = zeros(nEigs, nGnbAnt, nLayers, nCpInt);


for gridIdx = 0 : 1
    % grid sc indicies:
    grid_sc_idxs = 0 : 2 : (nSc - 1);
    grid_sc_idxs = grid_sc_idxs + gridIdx; 

    % check if grid is active:
    if bitand(uint8(1), bitshift(gridBitmask,-gridIdx))

        % if grid active, loop over tocc:
        gridPrm = gridPrms{gridIdx + 1};
        for toccIdx = 0 : (nDmrsSym - 1)

            % apply tocc, remove scrambling sequence:
            Y_dmrs_tocc_removed = zeros(nDmrsSc, nGnbAnt);
            tOCC = (-1).^(toccIdx * (0 : (nDmrsSym - 1))).' / nDmrsSym;
            scramblingIdx = (startInputSc - startScOfUeGrp) / 2 : ((startInputSc - startScOfUeGrp) / 2 + nDmrsSc - 1);
            for dmrsSymIdx = 0 : (nDmrsSym - 1)
                Y_dmrs_tocc_removed = Y_dmrs_tocc_removed + tOCC(dmrsSymIdx + 1)  * conj(r_dmrs(scramblingIdx + 1,dmrsSymIdx + 1)) .* squeeze(Y_dmrsSym(grid_sc_idxs + 1, dmrsSymIdx + 1, :));
            end

            % check if tocc is active:
            if(bitand(uint8(1), bitshift(gridPrm.toccBitmask, -toccIdx)))
                toccPrm = gridPrm.toccPrms{toccIdx + 1};

                % option to use Hamming window to measure noise:
                if((hammingFlag == 1) && (noiseMeasurmentMethod > 1))
                    hammingIdx    = (0 : (nDmrsSc - 1)).';
                    hammingWindow = 0.54 - 0.46*cos(2*pi*hammingIdx / (nDmrsSc - 1));
                    hammingWindow = repmat(hammingWindow, 1, nGnbAnt);
                    hammingCorr   = Y_dmrs_tocc_removed .* hammingWindow; 
    
                    fourierWorkspace                 = zeros(nGnbAnt, nZpDmrsSc);
                    fourierWorkspace(:, 1 : nDmrsSc) = hammingCorr.';
                    fourierWorkspace                 = ifft(fourierWorkspace, nZpDmrsSc, 2) * nZpDmrsSc;
    
                    totNoiseEnergy = measureNoise(totNoiseEnergy, fourierWorkspace, nCpInt, toccPrm.foccBitmask, noiseMeasurmentMethod, rolloff, nZpDmrsSc);
                end

                % compute projection coefficents:
                Y_dmrs_tocc_removed = Y_dmrs_tocc_removed.';
                Y_dmrs_tocc_removed = reshape(Y_dmrs_tocc_removed, 1, nGnbAnt, nDmrsSc);
                Y_dmrs_tocc_removed = repmat(Y_dmrs_tocc_removed, nEigs, 1, 1);

                fourierWorkspace = zeros(nEigs, nGnbAnt, nZpDmrsSc);
                for gnbAntIdx = 0 : (nGnbAnt - 1)
                    fourierWorkspace(:, gnbAntIdx + 1, 1 : nDmrsSc) = reshape(conj(eigenVecs) .* reshape(Y_dmrs_tocc_removed(:,gnbAntIdx + 1,:), nEigs, nDmrsSc), nEigs, 1, nDmrsSc);
                end
                fourierWorkspace = ifft(fourierWorkspace, nZpDmrsSc, 3) * nZpDmrsSc;

                % option to measure noise using fOCCs:
                if((hammingFlag == 0) && (noiseMeasurmentMethod > 1))
                    totNoiseEnergy = measureNoise(totNoiseEnergy, squeeze(fourierWorkspace(1,:,:)), nCpInt, toccPrm.foccBitmask, noiseMeasurmentMethod, rolloff, nZpDmrsSc);
                end

                % extract active focc:
                toccPrm = gridPrm.toccPrms{toccIdx + 1};
                for foccIdx = 0 : 1

                    % check if focc is active:
                    if(bitand(uint8(1), bitshift(toccPrm.foccBitmask, -foccIdx)))

                        % if focc active, store projection coefficents:
                        layerIdx    = toccPrm.foccPrms{foccIdx + 1}.layerIdx;
                        signal_ints = foccIdx * nZpDmrsSc / 2 + (0 : (nCpInt - 1));

                        extractedSignal                = fourierWorkspace(:,:,signal_ints + 1);
                        extractedSignal                = reshape(extractedSignal, nEigs, nGnbAnt, 1, nCpInt);
                        projCoeffs(:,:,layerIdx + 1,:) = extractedSignal;
                    end
                end
            else % NOTE: for now not using empty tOCC to measure noise 
            end
        end
    else
        % if grid in-active and no data present, add noise estimates:
        if(numDmrsCdmGrpsNoData == 2)
            for dmrsSymIdx = 0 : (nDmrsSym - 1)
                extractedNoise = Y_dmrsSym(grid_sc_idxs + 1, dmrsSymIdx + 1, :);
                E              = abs(extractedNoise).^2;
                totNoiseEnergy = totNoiseEnergy + sum(E(:));
            end
        end
    end
end

% estimate noise:
if(hammingFlag && (noiseMeasurmentMethod > 1))
    a                   = 0.3974;
    b                   = 0.0032;
    hammingWindowEnergy = a * nDmrsSc + b;

    N0 = totNoiseEnergy / (hammingWindowEnergy * nNoiseMeasurments);
else
    N0 = totNoiseEnergy / nNoiseMeasurments;
end


%%
% STEP 4: MATCHING PURSUIT

for ueIdx = 0 : (nUe - 1)
    ueLayerStartIdx    = uePrms{ueIdx + 1}.layerStartIdx;
    nUeLayers          = uePrms{ueIdx + 1}.nLayers;
    ueLayerIdxs        = ueLayerStartIdx : (ueLayerStartIdx + nUeLayers - 1);
    projCoeffsUe       = projCoeffs(:,:, ueLayerIdxs + 1, :);

    eqIntIdxs = zeros(nCpInt, 1);
    eqCoeffs  = zeros(nEigs, nGnbAnt, nUeLayers, nCpInt);
    nEqInt    = 0;

    for ompItrIdx = 0 : (nCpInt - 1)
        
        % estimate power-spectral-density:
        PSD = eigValues_rep(:, :, 1 : nUeLayers, :) .* abs(projCoeffsUe).^2;
        PSD = sum(PSD, 1);
        PSD = PSD / (nDmrsSc * sumEigValues);
        PSD = PSD - N0 / nDmrsSc;
        PSD = max(PSD, 0);

        % compute correlation:
        correlation = eigValues_rep(:, :, 1 : nUeLayers, :) .* abs(projCoeffsUe).^2;
        correlation = sum(correlation, 1);
        correlation = sum(correlation, 2);
        correlation = sum(correlation, 3);
        correlation = squeeze(correlation) / sumEigValues;
            
        % choose next coefficent:
        correlation(eqIntIdxs(1:nEqInt) + 1) = 0;
        [maxCorrelation, update_intIdx] = max(correlation);
        update_intIdx = update_intIdx - 1;

        % check for exit
        if(maxCorrelation <= (3.0*N0*nUeLayers*nGnbAnt))
            break;
        end

        % compute next coefficent:
        PSD_update         = PSD(:, :, :, update_intIdx + 1);
        PSD_update         = repmat(PSD_update, nEigs, 1, 1);
        projCoeffUe_update = projCoeffsUe(:, :, :, update_intIdx + 1);
        eigs_update        = reshape(eigValues, nEigs, 1, 1);
        eigs_update        = repmat(eigs_update, 1, nGnbAnt, nUeLayers);

        % if(mod(nUeLayers,2) ~= 0) % TODO: update to add focc sweeping for odd nUeLayers > 1
        % if(1)
        newEqCoeff = eigs_update .* PSD_update ./ (eigs_update .* PSD_update + N0) .* projCoeffUe_update;
        % else
        %     PSD_update         = reshape(PSD_update, nEigs, nGnbAnt * nUeLayers / 2, 2);
        %     projCoeffUe_update = reshape(projCoeffUe_update, nEigs, nGnbAnt * nUeLayers / 2, 2);
        %     newEqCoeff         = zeros(nEigs, nGnbAnt * nUeLayers / 2, 2);
        %     eigs_update        = eigs_update(:,:,nUeLayers/2);
        % 
        %     nSweeps = 5;
        %     for sweepIdx = 0 : (nSweeps - 1)
        %         projCoeffEvenLayers = projCoeffUe_update(:,:,1) - conj(corr_half_nZpDmrsSc) * newEqCoeff(:, :, 2);
        %         newEqCoeff(:, :, 1) = eigs_update(:,:,1) .* PSD_update(:, :, 1) ./ (eigs_update .* PSD_update(:, :, 1) + N0) .* projCoeffEvenLayers;
        % 
        %         projCoeffOddLayers  = projCoeffUe_update(:,:,2) - corr_half_nZpDmrsSc * newEqCoeff(:, :, 1);
        %         newEqCoeff(:, :, 2) = eigs_update(:,:,1) .* PSD_update(:, :, 2) ./ (eigs_update .* PSD_update(:, :, 2) + N0) .* projCoeffOddLayers;
        %     end
        %     newEqCoeff = reshape(newEqCoeff, nEigs, nGnbAnt, nUeLayers);
        % end

        % store new coefficent:
        eqIntIdxs(nEqInt + 1)      = update_intIdx;
        eqCoeffs(:,:,:,nEqInt + 1) = newEqCoeff;
        nEqInt                     = nEqInt + 1;

        % determine correlation between all boxes and updated box:
        C = zeros(nEigs, nEigs, nCpInt);
        for intIdx = 0 : (nCpInt - 1)
            
            if(intIdx < update_intIdx)
                boxIdxLessTheUpdateBox = 1;
            else
                boxIdxLessTheUpdateBox = 0;
            end

            distBoxIdxToUpdateBox = abs(intIdx - update_intIdx);
            if(boxIdxLessTheUpdateBox)
                C(:,:,intIdx + 1) = conj(corr(:,:,distBoxIdxToUpdateBox + 1));
            else
                C(:,:,intIdx + 1) = corr(:,:,distBoxIdxToUpdateBox + 1);
            end
        end

        % update projection coefficents:
        for ueLayerIdx = 0 : (nUeLayers - 1)
            for gnbAntIdx = 0 : (nGnbAnt - 1)
                tmp = squeeze(projCoeffsUe(:, gnbAntIdx + 1, ueLayerIdx + 1, :)) - squeeze(pagemtimes(C, newEqCoeff(:,gnbAntIdx + 1, ueLayerIdx + 1)));
                projCoeffsUe(:, gnbAntIdx + 1, ueLayerIdx + 1, :) = reshape(tmp, nEigs, 1, 1, nCpInt);
            end
        end
    end

    % interpolation:
    for ueLayerIdx = 0 : (nUeLayers - 1)
        layerIdx = ueLayerStartIdx + ueLayerIdx;
        gridIdx  = gridIdxs(layerIdx + 1);

        H_layerIdx = zeros(nSc, nGnbAnt);
        for gnbAntIdx = 0 : (nGnbAnt - 1)
            for i = 0 : (nEqInt - 1)
                eqIntIdx = eqIntIdxs(i + 1);
                freq     = (-gridIdx : (nSc - 1 - gridIdx)).' / 2;
                wave     = exp(-2*pi*1i*freq*eqIntIdx/nZpDmrsSc);

                interpCoeffs                = interpCob * ( eqCoeffs(:,gnbAntIdx + 1, ueLayerIdx + 1, i + 1));
                H_layerIdx(:,gnbAntIdx + 1) = H_layerIdx(:, gnbAntIdx + 1) + wave .* (zpInterpVec(((1 - gridIdx) : (nSc - 1 - gridIdx + 1)) + 1,:) * interpCoeffs);
            end
        end

        computeOutputScIdxs = startOutputScInBlock : (startOutputScInBlock + nOutputSc - 1);
        bufferIdxs          = scOffsetIntoChEstBuff : (scOffsetIntoChEstBuff + nOutputSc - 1);

        H_layerIdx                              = reshape(H_layerIdx.', 1, nGnbAnt, nSc) / sqrt(2^(numDmrsCdmGrpsNoData - 1));
        H_est(layerIdx + 1, : , bufferIdxs + 1) = H_layerIdx(:, :, computeOutputScIdxs + 1);
    end
end



end


function totNoiseEnergy = measureNoise(totNoiseEnergy, fourierWorkspace, nCpInt, foccBitmask, noiseMeasurmentMethod, rolloff, nZpDmrsSc)  
    for foccIdx = 0 : 1
        % check if focc is active:
        if(bitand(uint8(1), bitshift(foccBitmask, -foccIdx)))
    
            % measure noise in quite region of fOCC:
            if((noiseMeasurmentMethod == 3))
    
                noise_delayInts = foccIdx * nZpDmrsSc / 2 + ((nCpInt + rolloff) : (nZpDmrsSc / 2 - rolloff - 1));
                extractedNoise  = fourierWorkspace(:,noise_delayInts + 1);
                totNoiseEnergy  = totNoiseEnergy + sum(abs(extractedNoise(:)).^2);
            end
        elseif(noiseMeasurmentMethod == 2)
    
            % measure noise in in-active fOCC:
            noise_delayInts = foccIdx * nZpDmrsSc / 2 + (0 : (nZpDmrsSc / 2 - rolloff - 1));
            extractedNoise = fourierWorkspace(:,noise_delayInts + 1);
            totNoiseEnergy = totNoiseEnergy + sum(abs(extractedNoise(:)).^2);
        end
    end
end


