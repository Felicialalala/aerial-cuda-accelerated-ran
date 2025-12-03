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

%% Verify GPU TDL vs our own MATLAB TDL implementation
% using the h5 file generated from tdl_chan_ex, which include the random numbers needed to generated TDL channel mode

% Usage:
% 1. Run tdl_chan_ex with '-d', GPU TDL will generate tdlChan*.h5 files
% 2. Run this file with the above h5 files

% reason to use matlab verification: need to use the random initial phase generated on C platform, see thetaRand, which is saved per link

% tvFileName: h5 file name
% verbose: 0: only print final results; 1: print the max abs error per link
% ttiIdx: TTI index, use to calculate the current time stamp. Can be auto detected from TV file name or manual specify (not recommend)

% example: verify_tdl('tdlChan_1cell1Ue_4x4_A30_dopp5_cfo200_runMode0_freqConvert1_scSampling1_FP32_swap0_TTI0.h5')
% example: verify_tdl('tdlChan_1cell1Ue_4x4_A30_dopp5_cfo200_runMode0_freqConvert1_scSampling1_FP32_swap0_TTI0.h5', 1)

% for testing a batch of h5 flies, use below
% h5_dir = '.';
% h5_files = dir(fullfile(h5_dir, 'tdlChan*.h5'));
% for i = 1:numel(h5_files)
%     h5_file_name = fullfile(h5_dir, h5_files(i).name);
%     h5_info = h5info(h5_file_name);  % Query HDF5 file information
%     verify_tdl(h5_file_name);  % Call the verify function with the file name
% end

function checkRes = verify_tdl(tvFileName, verbose, ttiIdx)
fprintf("Checking TDL channel using file %s \n", tvFileName);

if nargin < 2
    verbose = 0;
end

if nargin < 3
    % use regular expression to extract the number between 'TTI' and '.h5'
    expression = 'TTI(\d+)';
    tokens = regexp(tvFileName, expression, 'tokens');

    if ~isempty(tokens)
        ttiIdx = str2double(tokens{1}{1});
        fprintf('Auto detected TTI index: %d \n', ttiIdx);
    else
        ttiIdx = 0;
        fprintf('Warning: TTI index not found, using default 0 \n');
    end
end

% detect precision from TV name: FP16 or FP32
expression = '\_FP(\d+)\_';
tokens = regexp(tvFileName, expression, 'tokens');

if ~isempty(tokens)
    precision = str2double(tokens{1}{1});
    fprintf('Auto detected data precision: FP%d \n', precision);
else
    precision = 32;
    fprintf('Warning: data precision not found, using default FP32 \n');
end

% detect precision from swap tx/rx
expression = '\_swap(\d+)\_';
tokens = regexp(tvFileName, expression, 'tokens');

if ~isempty(tokens)
    enableSwapTxRx = str2double(tokens{1}{1});
    if enableSwapTxRx
        fprintf('Auto detected swap tx/rx: %d (enabled) \n', enableSwapTxRx);
    else
        fprintf('Auto detected swap tx/rx: %d (disabled) \n', enableSwapTxRx);
    end
else
    enableSwapTxRx = 0;
    fprintf('Warning: swap tx/rx not found, using 0 (disabled) \n');
end

% read TDL config parameters
tdlCfg = h5read(tvFileName,'/tdlCfg');
delayProfile = char(tdlCfg.delayProfile);
delaySpread = double(tdlCfg.delaySpread);
nCell = double(tdlCfg.nCell);
nUe = double(tdlCfg.nUe);
nBsAnt = double(tdlCfg.nBsAnt);
nUeAnt = double(tdlCfg.nUeAnt);
f_doppler = double(tdlCfg.maxDopplerShift);
f_samp = double(tdlCfg.f_samp);
f_batch = double(tdlCfg.fBatch); % update rate of quasi-static channel
% defualt f_batch is 15e3, which is 50x larger than max doppler freq (300Hz). It should be fast enough to capture the channel variation over time
numPath = double(tdlCfg.numPath);
cfoHz = double(tdlCfg.cfoHz);
sigLenPerAnt = double(tdlCfg.sigLenPerAnt);
timeDelay = double(tdlCfg.delay);
N_sc = double(tdlCfg.N_sc);
N_sc_Prbg = double(tdlCfg.N_sc_Prbg);
scSpacingHz = double(tdlCfg.scSpacingHz);
runMode = double(tdlCfg.runMode);
procSigFreq = double(tdlCfg.procSigFreq);
freqConvertType = double(tdlCfg.freqConvertType);
% freqConvertType 0: use first SC for CFR on the Prbg
% freqConvertType 1: use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
% freqConvertType 2: use last SC for CFR on the Prbg
% freqConvertType 3: use average SC for CFR on the Prbg
% freqConvertType 4: use average SC for CFR on the Prbg with removing frequency ramping
scSampling = double(tdlCfg.scSampling);
useFFT = 0;

% Read TDL non-zero fir index, non-zero fir power, time channel
firNzIdxGpu = h5read(tvFileName,'/firNzIdx');
firNzPwGpu = h5read(tvFileName,'/firNzPw');
batchCumLen = double(h5read(tvFileName,'/batchCumLen')); % start tx sample per batch
tBatch = double(h5read(tvFileName,'/tBatch'))'; % start time per batch
nBatch = length(tBatch);
timeChan = h5read(tvFileName,'/timeChan');
fprintf("TDL channel verification: %d cells, %d UEs, %d nBsAnt, %d nUeAnt, runMode %d, freqConvertType %d. \n", nCell, nUe, nBsAnt, nUeAnt, runMode, freqConvertType);

ttiLen = 0.001 / (scSpacingHz / 15e3); % 500 us
N_fft = 4096;
N_Prbg = ceil(N_sc/N_sc_Prbg);
corrMatrix = diag(ones(1, nBsAnt*nUeAnt)); % load MIMO correlation matrix; TOOD: only support low MIMO correlation, so corrMatrix is an identity matrix
pdp = loadPdp(['TDL' delayProfile num2str(delaySpread) '-' num2str(f_doppler) '-Low']);
T_samp = 1/f_samp;
delay_samp = round(timeDelay/T_samp);
% quantize to fir filter with T_samp interval
PathDelays = pdp(:,1);
firTapMap = round(PathDelays * 1e-9 / T_samp) + 1;  % map to the same tap dure to sampling
firPw = 10.^(pdp(:,2)/10);
lenFir = round(PathDelays(end) * 1e-9 / T_samp) + 1;
firPw = sqrt(firPw/(sum(firPw)*double(numPath))); % sqrt(numPath) needs to be multipled to ensure over long term E{||chanMatrix(:, rxAntIdx, :, batchIdx)||_2^2} = 1 for all (rxAntIdx, batchIdx)
NtapPdp = length(PathDelays);  % number of taps in PDP table
Ntap = length(unique(firTapMap));  % number of taps in final CIR
timeChanSizePerLink = nBatch * Ntap * nBsAnt * nUeAnt;
switch(runMode)
    case 0
        freqChanPrbgSizePerLink = 0;
    case {1, 2}
        freqChanPrbg = h5read(tvFileName,'/freqChanPrbg'); % freq chan on PRBG
        freqChanPrbgSizePerLink = nBsAnt * nUeAnt * N_Prbg;
end

if(sigLenPerAnt > 0)
    txSigIn = h5read(tvFileName,'/txSigIn'); % tx time signal
    rxSigOut = h5read(tvFileName,'/rxSigOut'); % rx time signal
    if enableSwapTxRx
        sigOutLenPerLink = sigLenPerAnt * nBsAnt;
        assert(length(rxSigOut.re) == sigOutLenPerLink * nCell * nUe, "Error: length of input signal does not match sigLenPerAnt in TDL config!")
    else
        sigOutLenPerLink = sigLenPerAnt * nUeAnt;
        assert(length(rxSigOut.re) == sigOutLenPerLink * nCell * nUe, "Error: length of input signal does not match sigLenPerAnt in TDL config!")
    end
end

checkRes = zeros(nCell*nUe, 8); % saving TDL verification results, max dimension 8, format: [cid, uid, firNzIdx firNzPw, timeChan, sigOut, freqChanSc, freqChanPrg]

% check non-zero fir index and power, which are stored as sparse matrix on GPU
% results of non-zero fir index, same for all links
firNzIdx = sort(unique(round(PathDelays * 1e-9 * f_samp)));
checkRes(:, 3) = max(firNzIdx - double(firNzIdxGpu));
% results of non-zero fir power, same for all links
checkRes(:, 4) = max(firPw - firNzPwGpu);

for cellIdx = 1 : nCell
    for ueIdx = 1 : nUe
        cidUidIdx = (cellIdx - 1) * nUe + ueIdx;
        % save cid
        checkRes(cidUidIdx, 1) = cellIdx - 1;
        % save uid
        checkRes(cidUidIdx, 2) = ueIdx - 1;

        % check time channel
        thetaRand_file = h5read(tvFileName,['/thetaRandLink' num2str(cidUidIdx - 1)]);
        % read random phase used on GPU TDL
        thetaRand = zeros(nBsAnt, nUeAnt, NtapPdp, numPath,2);
        for rxIdx = 1:nUeAnt
            for txIdx = 1:nBsAnt
                for iTap = 1:NtapPdp
                    for iPath = 1:numPath
                        pos = (((rxIdx-1) * nBsAnt + txIdx - 1) * NtapPdp + iTap - 1)*numPath + iPath;
                        % thetaRand_file is uniform distributed in (0, 1]
                        thetaRand(txIdx, rxIdx, iTap, iPath, 1) = 2 * pi * thetaRand_file(2*pos-1);
                        thetaRand(txIdx, rxIdx, iTap, iPath, 2) = 2 * pi * thetaRand_file(2*pos);
                    end
                end
            end
        end

        chanMatrix = zeros(nBsAnt, nUeAnt, lenFir, nBatch);
        timeSeq = tBatch + ttiIdx * ttiLen;
        % superimpose N paths, generate time chan
        for idxIn = 1:nBsAnt
            for idxOut = 1:nUeAnt
                for idxTap = 1:NtapPdp
                    hsum = zeros(1, nBatch);
                    for idxPath = 1:numPath
                        alpha_0 = pi/4/numPath * idxTap/(NtapPdp+2);
                        % real part
                        freqRand_real = f_doppler * cos(pi/2/numPath * (idxPath - 0.5) + alpha_0);
                        % imag part
                        freqRand_imag = f_doppler * cos(pi/2/numPath * (idxPath - 0.5) - alpha_0);

                        % Note size(chanMatrix) = [nBsAnt, nUeAnt, lenFir, nBatch]
                        h1 = firPw(idxTap) * ( ...   % normalization coe
                            cos(2*pi*freqRand_real*timeSeq + thetaRand(idxIn, idxOut, idxTap, idxPath, 1)) + ...  % real part
                            + 1j * cos(2*pi*freqRand_imag*timeSeq + thetaRand(idxIn, idxOut, idxTap, idxPath, 2))...  % imag part
                            );
                        hsum = h1 + hsum;
                    end
                    chanMatrix(idxIn, idxOut, firTapMap(idxTap),:) = chanMatrix(idxIn, idxOut, firTapMap(idxTap),:) + reshape(hsum, [1, 1, 1, nBatch]);
                end
            end
        end

        % apply MIMO correlation: TODO, currently no effect
        for idxTap = 1:lenFir
            if ismember(idxTap, firTapMap)
                for batchIdx = 1:nBatch
                    tap1 = reshape(chanMatrix(:, :, idxTap, batchIdx), [1, nBsAnt*nUeAnt]);
                    tap2 = (corrMatrix*tap1.').';
                    chanMatrix(:, :, idxTap, batchIdx) = reshape(tap2, [nBsAnt, nUeAnt]);
                end
            end
        end

        % Check normalize chanMatrix power, long term average should be 1
        % sqrt(sum(sum(sum(sum(abs(chanMatrix).^2))))/(nUeAnt*nBatch));

        % compare tdl time chan
        chan_diff = zeros(nBatch * nBsAnt * nUeAnt, 1);
        for batchIdx = 1:nBatch
            for txIdx = 1:nBsAnt
                for rxIdx = 1:nUeAnt
                    batchtxRxIdx = ((batchIdx-1)*nUeAnt + rxIdx-1)*nBsAnt + txIdx-1;
                    timeChan_ref = squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx));
                    chan_gpu_pos = (cidUidIdx - 1) * timeChanSizePerLink + batchtxRxIdx*Ntap+ (1:Ntap);
                    tempChan_gpu = timeChan.re(chan_gpu_pos) + 1i*timeChan.im(chan_gpu_pos);
                    timeChan_gpu = zeros(size(timeChan_ref));
                    timeChan_gpu(firNzIdxGpu+1) = tempChan_gpu;
                    chan_diff(batchtxRxIdx + 1) = max(abs(timeChan_ref - timeChan_gpu));
                end
            end
        end
        checkRes(cidUidIdx, 5) = max(abs(chan_diff(:)));

        % compare freq chan on sc and/or prbg, only generate in the first batch
        % freqChanSc is saved and read per link
        freqChanSC_ref    = zeros(nBatch, nUeAnt, nBsAnt, N_sc);
        freqChanSc_diff   = zeros(nBatch*nBsAnt*nUeAnt, 1);
        freqChanPrbg_diff = zeros(nBatch*nBsAnt*nUeAnt, 1);
        if(runMode > 0)
            freqChanNormalizeCoe = 1; % sqrt(nBsAnt);
            if (runMode == 2)
                freqChanSc = h5read(tvFileName,['/freqChanScLink' num2str(cidUidIdx - 1)]);
            end
            for batchIdx = 1:nBatch
                if (runMode == 2) % CFR on SC calculated by GPU
                    freqChanPerBatchLen = nUeAnt * nBsAnt * N_sc;
                    freqChanPerBatchIdx = (batchIdx-1)*freqChanPerBatchLen+1:batchIdx*freqChanPerBatchLen;
                    freqChanScBatch = freqChanSc.re(freqChanPerBatchIdx) + 1i * freqChanSc.im(freqChanPerBatchIdx);
                end
                for txIdx = 1:nBsAnt
                    for rxIdx = 1:nUeAnt
                        txRxIdx = (rxIdx-1)*nBsAnt + txIdx-1;
                        cfr = calCfrFromCir(N_sc, N_sc_Prbg, scSpacingHz, f_samp, timeDelay, cfoHz, timeSeq(batchIdx), squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx)), useFFT, N_fft, freqConvertType);
                        freqChanAntSC_ref = freqChanNormalizeCoe * cfr;
                        freqChanSC_ref(batchIdx, rxIdx, txIdx, :) = freqChanAntSC_ref;
                        if (scSampling > 1)
                            scIdx = 0:N_sc-1;
                            notCalSc = scIdx(mod(0:N_sc-1, scSampling) ~= 0);
                            freqChanAntSC_ref(notCalSc + 1) = 0;
                        end

                        if (runMode == 2)
                            scIdx = 0:N_sc-1;
                            notCalSc = scIdx(mod(0:N_sc-1, scSampling) ~= 0);
                            freqChanScBatch(notCalSc + 1) = 0;
                            freqChanSCGpu = freqChanScBatch(N_sc*txRxIdx + (1:N_sc));
                            freqChanSc_diff((batchIdx-1)*nUeAnt*nBsAnt+txRxIdx+1) = max(abs(freqChanAntSC_ref-freqChanSCGpu));
                        end

                        freqChanPrbg_ref = zeros(N_Prbg,1);
                        switch freqConvertType
                            case 0 % use first SC for CFR on the Prbg
                                freqChanPrbg_ref = freqChanAntSC_ref(1:N_sc_Prbg:end);
                            
                            case 1 % use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = freqChanAntSC_ref(N_sc_Prbg/2+1 : N_sc_Prbg : end);
                                else
                                    freqChanPrbg_ref(1:N_Prbg-1) = freqChanAntSC_ref(N_sc_Prbg/2+1 : N_sc_Prbg : end);
                                    freqChanPrbg_ref(N_Prbg)     = freqChanAntSC_ref((N_Prbg-1)*N_sc_Prbg + floor((N_sc - (N_Prbg-1)*N_sc_Prbg)/2) + 1);
                                end
                            
                            case 2 % use last SC for CFR on the Prbg
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = freqChanAntSC_ref(N_sc_Prb : N_sc_Prbg : end);
                                else
                                    freqChanPrbg_ref(1:N_Prbg-1) = freqChanAntSC_ref(N_sc_Prbg : N_sc_Prbg : end);
                                    freqChanPrbg_ref(N_Prbg)     = freqChanAntSC_ref(N_sc);
                                end

                            case {3, 4} % use average SC for CFR on the Prbg
                                if(N_Prbg * N_sc_Prbg == N_sc)    
                                    freqChanPrbg_ref = transpose(mean(reshape(freqChanAntSC_ref, N_sc_Prbg, N_Prbg))) * N_sc_Prbg / length(0:scSampling:N_sc_Prbg-1);
                                else
                                    N_sc_last_Prbg = N_sc - (ceil(N_sc / N_sc_Prbg) - 1) * N_sc_Prbg;
                                    freqChanPrbg_ref(1:N_Prbg-1) = transpose(mean(reshape(freqChanAntSC_ref(1:(N_Prbg-1)*N_sc_Prbg), N_sc_Prbg, N_Prbg-1))) * N_sc_Prbg / length(0:scSampling:N_sc_Prbg-1);
                                    freqChanPrbg_ref(N_Prbg)     = mean(freqChanAntSC_ref((N_Prbg-1)*N_sc_Prbg+1:end)) * N_sc_last_Prbg / length(0:scSampling:N_sc_last_Prbg-1);
                                end
                            
                            otherwise
                                fprintf('Invalid freqConvertType %d\n', freqConvertType);
                        end
                                        
                        freqChanPrbgOffset = ((cidUidIdx - 1) * nBatch + batchIdx - 1) * freqChanPrbgSizePerLink;
                        freqChanPrbgGpu = freqChanPrbg.re(freqChanPrbgOffset + N_Prbg*txRxIdx + (1:N_Prbg)) + 1i * freqChanPrbg.im(freqChanPrbgOffset + N_Prbg*txRxIdx + (1:N_Prbg));
                        freqChanPrbg_diff((batchIdx-1)*nUeAnt*nBsAnt+txRxIdx+1) = max(abs(freqChanPrbg_ref-freqChanPrbgGpu));
                    end
                end
            end
            % results of freq chan on sc 
            checkRes(cidUidIdx, 7) = max(abs(freqChanSc_diff));
            % results of freq chan on prbg 
            checkRes(cidUidIdx, 8) = max(abs(freqChanPrbg_diff));
        end

        % generate and compare rx singal
        if (sigLenPerAnt > 0)
            if enableSwapTxRx
                nTxAnt = nUeAnt;
                nRxAnt = nBsAnt;
            else
                nTxAnt = nBsAnt;
                nRxAnt = nUeAnt;
            end
            if (procSigFreq == 0)  % processing signal in time domain
                rxSigOut_ref  = zeros(sigOutLenPerLink, 1);
                for batchIdx = 1:nBatch
                    if (batchCumLen(batchIdx) + 1 > sigLenPerAnt) % end of tx data samples
                        break;
                    end
                    batchSampleIdx = batchCumLen(batchIdx)+1 : min(sigLenPerAnt, batchCumLen(batchIdx + 1));
                    for txIdx = 1:nTxAnt  % UL: nUeAnt, DL: nBsAnt
                        % prepare tx samples to be filtered by TDL time chan
                        sigInIdx = sigLenPerAnt*((cidUidIdx - 1) * nTxAnt + txIdx - 1) + batchSampleIdx;
                        temp_sigIn = txSigIn.re(sigInIdx) + 1i * txSigIn.im(sigInIdx);
                        N_padding = lenFir;
                        if(batchIdx == 1)
                            pad_sigIn = zeros(N_padding, 1);
                        else
                            pad_sigInIdx = sigLenPerAnt*((cidUidIdx - 1) * nTxAnt + txIdx - 1) + batchSampleIdx(1)-(N_padding:-1:1);
                            pad_sigIn = txSigIn.re(pad_sigInIdx) + 1i * txSigIn.im(pad_sigInIdx);
                        end
                        temp_sigIn = [pad_sigIn ;temp_sigIn];
                        for rxIdx = 1:nRxAnt
                            if (enableSwapTxRx)
                                timeChan_ref = squeeze(chanMatrix(rxIdx, txIdx, :, batchIdx));
                            else
                                timeChan_ref = squeeze(chanMatrix(txIdx, rxIdx, :, batchIdx));
                            end
                            % compare rx singal
                            if(sigLenPerAnt > 0)
                                temp_rxSigOut = filter(timeChan_ref, 1, temp_sigIn);
                                temp_rxSigOut = temp_rxSigOut(N_padding+1:end);
                                temp_rxSigOut = circshift(temp_rxSigOut, delay_samp); % add delay
                                rxTimeIdx = sigLenPerAnt*(rxIdx-1)+batchSampleIdx;
                                rxSigOut_ref(rxTimeIdx) = rxSigOut_ref(rxTimeIdx) + temp_rxSigOut;
                            end
                        end
                    end
                end

                % add CFO
                CFOseq = exp(1j*2*pi*([0:sigLenPerAnt-1]'*T_samp + ttiIdx * ttiLen)*cfoHz);
                rxSigOut_ref = rxSigOut_ref.*repmat(CFOseq(:), [nRxAnt, 1]);
                
                % read GPU data out
                rxSigOutOffset = (cidUidIdx - 1) * sigOutLenPerLink;
                rxSigOutGpu = rxSigOut.re(rxSigOutOffset + (1 : sigOutLenPerLink)) + 1i*rxSigOut.im(rxSigOutOffset + (1 : sigOutLenPerLink));
            else
                % freqChanAntSC_ref is the CFR on all N_sc
                % txSigIn is the input signal
                rxSigOut_ref  = zeros(nUeAnt * N_sc * nBatch, 1);
                for batchIdx = 1:nBatch
                    batchSampleIdx = (batchIdx-1)*N_sc+1 : batchIdx*N_sc;
                    for rxIdx = 1:nRxAnt
                        rxTimeIdx = (rxIdx-1) * nBatch * N_sc + batchSampleIdx;
                        for txIdx = 1:nBsAnt
                            sigInIdx = ((cidUidIdx - 1) * nTxAnt + txIdx - 1) * nBatch * N_sc + batchSampleIdx;
                            temp_sigIn = txSigIn.re(sigInIdx) + 1i * txSigIn.im(sigInIdx);
                            freqChanAntSC_ref = squeeze(freqChanSC_ref(batchIdx, rxIdx, txIdx, :));
                            rxSigOut_ref(rxTimeIdx) = rxSigOut_ref(rxTimeIdx) + temp_sigIn .* freqChanAntSC_ref;
                        end
                    end
                end
                % read GPU data out
                rxSigOutOffset = (cidUidIdx - 1) * nRxAnt * N_sc * nBatch;
                rxSigOutGpu = rxSigOut.re(rxSigOutOffset + (1 : nRxAnt * N_sc * nBatch)) + 1i*rxSigOut.im(rxSigOutOffset + (1 : nRxAnt * N_sc * nBatch));
            end
            % results of rx signal comparison
            checkRes(cidUidIdx, 6) = max(abs(rxSigOut_ref - rxSigOutGpu));
        end
    end
end

% print verification results
printTestRes(checkRes, runMode, sigLenPerAnt, precision, verbose);
end

% load power delay profile (PDP) in 38.141
function pdp = loadPdp(chanType)

switch chanType
    case 'TDLA30-5-Low'
        pdp = [...
            0       -15.5;
            10      0;
            15      -5.1;
            20      -5.1;
            25      -9.6;
            50      -8.2;
            65      -13.1;
            75      -11.5;
            105     -11.0;
            135  	-16.2;
            150	    -16.6;
            290   	-26.2
            ];
    case 'TDLA30-10-Low'
        pdp = [...
            0       -15.5;
            10      0;
            15      -5.1;
            20      -5.1;
            25      -9.6;
            50      -8.2;
            65      -13.1;
            75      -11.5;
            105     -11.0;
            135  	-16.2;
            150	    -16.6;
            290   	-26.2
            ];
    case 'TDLB100-400-Low'
        pdp = [...
            0       0;
            10      -2.2;
            20      -0.6;
            30      -0.6;
            35      -0.3;
            45      -1.2;
            55      -5.9;
            120     -2.2;
            170     -0.8;
            245 	-6.3;
            330     -7.5;
            480     -7.1
            ];
    case 'TDLC300-100-Low'
        pdp = [...
            0       -6.9;
            65      0;
            70      -7.7;
            190     -2.5;
            195     -2.4;
            200     -9.9;
            240     -8.0;
            325     -6.6;
            520     -7.1;
            1045	-13.0;
            1510	-14.2;
            2595	-16.0
            ];
    otherwise
        error('chanType is not supported ... \n');
end

end

function printTestRes(checkRes, runMode, sigLenPerAnt, precision, verbose)
maxAbsError = max(max(abs(checkRes(:,3:end))));
% FP16 processing input signal or freq chan typically has a higher error compared to FP32
% these threshold are emperical, can be adjusted
if( (maxAbsError < 0.015 && precision == 16) || (maxAbsError < 0.002 && precision == 32))
    fprintf("TDL verification PASS! maxAbsError is %e\n", maxAbsError);
else
    fprintf("TDL verification FAIL! maxAbsError is %e\n", maxAbsError);
end

if(verbose)
    fprintf("TDL verification results (max absolute error per link)\n")
    if(sigLenPerAnt <= 0) % no input tx signals
        switch(runMode)
            case 0
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan\n");
                display(checkRes(:,1:5));
            case 1
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t freqChanPrg\n");
                display(checkRes(:,[1:5 8:end]));
            case 2
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t freqChanSc \t freqChanPrg\n");
                display(checkRes(:,[1:5 7:end]));
        end
    else
        switch(runMode)
            case 0
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut\n");
                display(checkRes(:,1:6));
            case 1
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut \t freqChanPrg\n");
                display(checkRes(:,[1:6 8:end]));
            case 2
                fprintf("\t cid \t uid \t firNzIdx \t firNzPw \t timeChan \t sigOut \tfreqChanSc \t freqChanPrg\n");
                display(checkRes); % all results printed
        end
    end
end
end

function cfr = calCfrFromCir(N_sc, N_sc_Prbg, scSpacingHz, f_samp, timeDelay, cfoHz, tBatch, cir, useFFT, N_fft, freqConvertType)
    % Calculate the CFR from the complex interference response (CIR)
    %
    % Inputs:
    % N_sc: number of subcarriers
    % scSpacingHz: subcarrier spacing in Hz
    % f_samp: sampling frequency in Hz
    % timeDelay: time domain delay in second
    % cfoHz: carrier offset frequency in Hz
    % cfrBatchRotationCfo: rotation of CFR per batch by CFO
    % cir: time chan (firNzLen x 1)
    % useFFT: use FFT for CFR calculation
    % N_fft: number of FFT points
    % freqConvertType: conversion of CFR on Sc to Prbg, removing freq ramping if freqConvertType == 4

    % Outputs:
    % cfr: freq chan

    N_Prbg = ceil(N_sc / N_sc_Prbg);
    N_sc_last_Prbg = N_sc - (N_Prbg - 1) * N_sc_Prbg;
    
    if(useFFT)
        tmp = fftshift(fft(cir, N_fft)); % only generate in the first batch
        cfr = tmp((N_fft - N_sc)/2 + (1:N_sc));
    else
        delays = (0:length(cir)-1) / f_samp;
        if (freqConvertType == 4)
            tmpScIdx = repmat([0:N_sc_Prbg:N_sc-N_sc_last_Prbg-1] + N_sc_Prbg/2, N_sc_Prbg, 1);
            tmpScIdx = tmpScIdx(:);
            scIdx = [tmpScIdx; ((N_Prbg-1) * N_sc_Prbg + N_sc_last_Prbg/2) * ones(N_sc_last_Prbg, 1)] - N_sc/2;
        else
            scIdx = (-N_sc/2 : N_sc/2-1)';
        end
        % add impact of CFO per SC: shift of sc frequency
        frequencies = scSpacingHz * scIdx - cfoHz;
        firNzDelayScFreq2Pi = exp(-1j * 2 * pi * frequencies * delays);
        cfr = firNzDelayScFreq2Pi * cir;
        % add impact of time delay per SC: phase ratation of cfr per Sc
        cfrPhaseShift = exp(-1j * 2 * pi * scIdx * round(timeDelay * f_samp) / N_fft);
        cfr = cfr .* cfrPhaseShift;
        % add impact of CFO per batch (whole SCs)
        cfrBatchRotationCfo = exp(1j * 2 * pi * tBatch * cfoHz);
        cfr = cfr * cfrBatchRotationCfo;
    end
end