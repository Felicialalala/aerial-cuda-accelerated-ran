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

function [pUciF1] = pucchF1_rx_kernel(nRxAnt, pucchF1tables, pucchF1dynamicDesc, pUciF1, grpIdx)
%% load group common parameters
rBase               = pucchF1tables.rBase;
csPhaseRamp         = pucchF1tables.csPhaseRamp;
Wf                  = pucchF1tables.Wf; %% frequency domain filter
s                   = pucchF1tables.s;
W_noiseIso          = pucchF1tables.W_noiseIso;
noiseDim            = pucchF1tables.noiseDim;
Xtf                 = pucchF1dynamicDesc.Xtf;
BWPStart            = pucchF1dynamicDesc.BWPStart{grpIdx};
freqHopFlag         = pucchF1dynamicDesc.freqHopFlag{grpIdx};
startSym            = pucchF1dynamicDesc.startSym{grpIdx};
startPrb            = pucchF1dynamicDesc.startPrb{grpIdx};
nSym                = pucchF1dynamicDesc.nSym{grpIdx};
groupHopFlag        = pucchF1dynamicDesc.groupHopFlag{grpIdx};
secondHopPrb        = pucchF1dynamicDesc.secondHopPrb{grpIdx};
u                   = pucchF1dynamicDesc.u{grpIdx};
csCommon            = pucchF1dynamicDesc.csCommon{grpIdx};
scs                 = pucchF1dynamicDesc.scs;

startCrb            = startPrb + BWPStart;
secondHopCrb        = secondHopPrb + BWPStart;

nSymDataFirstHop = 0;
nSymFirstHop = 0;
nSymDMRSFirstHop = 0;
nSymDataSecondHop = 0;
nSymDMRSSecondHop = 0;
nSym_data = pucchF1dynamicDesc.nSym_data{grpIdx}; % number of data symbols
nSym_dmrs = pucchF1dynamicDesc.nSym_dmrs{grpIdx}; % number of dmrs symbols

tOCC_cell = pucchF1tables.tOCC_cell;

Wt_matrices = pucchF1tables.Wt;

if freqHopFlag
    nSymDataFirstHop = pucchF1dynamicDesc.nSymDataFirstHop{grpIdx};
    nSymFirstHop = pucchF1dynamicDesc.nSymFirstHop{grpIdx};
    nSymDMRSFirstHop = pucchF1dynamicDesc.nSymDMRSFirstHop{grpIdx};
    nSymDataSecondHop = pucchF1dynamicDesc.nSymDataSecondHop{grpIdx};
    nSymDMRSSecondHop = pucchF1dynamicDesc.nSymDMRSSecondHop{grpIdx};
    
    tOCC_data{1} = tOCC_cell{nSymDataFirstHop};
    tOCC_data{2} = tOCC_cell{nSymDataSecondHop};
    tOCC_dmrs{1} = tOCC_cell{nSymDMRSFirstHop};
    tOCC_dmrs{2} = tOCC_cell{nSymDMRSSecondHop};
else
    tOCC_data = tOCC_cell{nSym_data};
    tOCC_dmrs = tOCC_cell{nSym_dmrs};
end

%v                   = dynamicDesc.v{grpIdx};
%             %% time domain filter
%             Wt_cell = cell(11,1);
%             for i = 1 : 11
%                 Wt_cell{i} = pucch_time_filter(3 + i, mu, pucchTable);
%             end
%             Wt = Wt_cell{nSym - 3};

%% load data from Xtf
Y_pucch = [];
freq_idx = (startCrb*12 + 1) : 12*(startCrb + 1);

if freqHopFlag
    %%First hop
    time_idx = (startSym + 1) : (startSym + nSymFirstHop);
    
    Y_pucch = Xtf(freq_idx,time_idx,1:nRxAnt);
    
    %%Second hop
    time_idx = (startSym + nSymFirstHop + 1):(startSym + nSym);
    freq_idx = (secondHopCrb*12 + 1) : 12*(secondHopCrb + 1);
    
    Y_pucch = [Y_pucch Xtf(freq_idx,time_idx,1:nRxAnt)];
else
    time_idx = (startSym + 1) : (startSym + nSym);
    Y_pucch = Xtf(freq_idx,time_idx,1:nRxAnt);
end

%% Seperate pucch signal into dmrs and data signals  
Y_dmrs = zeros(12,nSym_dmrs,nRxAnt);
Y_data = zeros(12,nSym_data,nRxAnt);

for i = 1 : nSym_dmrs
    Y_dmrs(:,i,:) = Y_pucch(:,2*(i-1) + 1,:);
end
for i = 1 : nSym_data
    Y_data(:,i,:) = Y_pucch(:,2*(i-1) + 2,:);
end

rssiTemp = mean(abs(Y_dmrs).^2, 'all');
rssi_dB = 10*log10(rssiTemp*nRxAnt*12);
rssiSqrd = rssiTemp^2;

switch nRxAnt
   case {1, 2}
       alpha = 31/3;
   case {3, 4, 5}
       alpha = 23.5;
   otherwise
       alpha = 370/4;
end


%% remove cell code
r = rBase(:,u(1)+1);

if freqHopFlag
    %%first hop:
    %data:
    for i = 1 : nSymDataFirstHop
        for bs = 1 : nRxAnt
            Y_data(:,i,bs) = conj(r) .* Y_data(:,i,bs);
        end
    end
    %dmrs:
    for i = 1 : nSymDMRSFirstHop
        for bs = 1 : nRxAnt
            Y_dmrs_centered(:,i,bs) = conj(r) .* s.* Y_dmrs(:,i,bs);
        end
    end
    
    %%second hop:
    if groupHopFlag
        r = rBase(:,u(2)+1);
    end
    %data:
    for i = nSymDataFirstHop+1 : nSym_data
        for bs = 1 : nRxAnt
            Y_data(:,i,bs) = conj(r) .* Y_data(:,i,bs);
        end
    end
    %dmrs:
    for i = nSymDMRSFirstHop+1 : nSym_dmrs
        for bs = 1 : nRxAnt
            Y_dmrs_centered(:,i,bs) = conj(r) .* s.* Y_dmrs(:,i,bs);
        end
    end
else
    %dmrs:
    for i = 1 : nSym_dmrs
        for bs = 1 : nRxAnt
            Y_dmrs_centered(:,i,bs) = conj(r) .* s.* Y_dmrs(:,i,bs);
        end
    end
    %data:
    for i = 1 : nSym_data
        for bs = 1 : nRxAnt
            Y_data(:,i,bs) = conj(r) .* Y_data(:,i,bs);
        end
    end
end


%%%%%%%%%%%%%%%%%%% per UCI processing %%%%%%%%%%%%%%%%%%%%%%%
nUci = pucchF1dynamicDesc.nUciInGrp{grpIdx};

for UciIdx = 1:nUci %% iterate over PUCCH UCIs within a group
   %% load per-UCI parameters
   nBits           = pucchF1dynamicDesc.bitLenHarq{grpIdx}(UciIdx);
   cs0             = pucchF1dynamicDesc.cs0{grpIdx}(UciIdx);
   uciOutputIdx    = pucchF1dynamicDesc.uciOutputIdx{grpIdx}(UciIdx);
   tOCCidx         = pucchF1dynamicDesc.timeDomainOccIdx{grpIdx}(UciIdx);
   DTXthreshold    = pucchF1dynamicDesc.DTXthreshold{grpIdx}(UciIdx);
   srFlag          = pucchF1dynamicDesc.srFlag{grpIdx}(UciIdx);
   
   if nBits == 0 && ~srFlag
       pUciF1{uciOutputIdx}.SinrDB   = -99; % placeholder for SINR in dB
       pUciF1{uciOutputIdx}.InterfDB =  99; % placeholder for interference plus noise in dB
       pUciF1{uciOutputIdx}.RSRP     = -99;
       pUciF1{uciOutputIdx}.RSSI     = -99;
       pUciF1{uciOutputIdx}.HarqValues = [];
       pUciF1{uciOutputIdx}.NumHarq = 0;
       pUciF1{uciOutputIdx}.SRindication = 0;
       pUciF1{uciOutputIdx}.SRconfidenceLevel = 1;
       pUciF1{uciOutputIdx}.HarqconfidenceLevel = 1;
       continue;
   end
    %% remove time and frequency cover codes
   cs = zeros(nSym,1);
   for i = 1 : nSym
       cs(i) = mod(csCommon(i) + cs0,12);
   end
      
   if freqHopFlag
       Y_dmrs_perUci = zeros(12,nSym_dmrs,nRxAnt);
       Y_data_perUci = zeros(12,nSym_data,nRxAnt);
       
       %%First hop
       for i = 1 : nSymDMRSFirstHop
           for bs = 1 : nRxAnt
               Y_dmrs_perUci(:,i,bs) = conj(tOCC_dmrs{1}(tOCCidx+1,i)) * conj(csPhaseRamp(:,cs(2*(i-1)+1)+1)) .* ...
                   Y_dmrs_centered(:,i,bs);
           end
       end
       
       for i = 1 : nSymDataFirstHop
           for bs = 1 : nRxAnt
               Y_data_perUci(:,i,bs) = conj(tOCC_data{1}(tOCCidx+1,i)) * conj(csPhaseRamp(:,cs(2*(i-1)+2)+1)) .* ...
                   Y_data(:,i,bs);
           end
       end
       
       %%Second hop
       for i = nSymDMRSFirstHop+1 : nSym_dmrs
           for bs = 1 : nRxAnt
               Y_dmrs_perUci(:,i,bs) = conj(tOCC_dmrs{2}(tOCCidx+1,i-nSymDMRSFirstHop)) * conj(csPhaseRamp(:,cs(2*(i-1)+1)+1)) .* ...
                   Y_dmrs_centered(:,i,bs);
           end
       end
       
       for i = nSymDataFirstHop+1 : nSym_data
           for bs = 1 : nRxAnt
               Y_data_perUci(:,i,bs) = conj(tOCC_data{2}(tOCCidx+1,i-nSymDataFirstHop)) * conj(csPhaseRamp(:,cs(2*(i-1)+2)+1)) .* ...
                   Y_data(:,i,bs);
           end
       end
   else
       Y_dmrs_perUci = zeros(12,nSym_dmrs,nRxAnt);
       
       for i = 1 : nSym_dmrs
           for bs = 1 : nRxAnt
               Y_dmrs_perUci(:,i,bs) = conj(tOCC_dmrs(tOCCidx+1,i)) * conj(csPhaseRamp(:,cs(2*(i-1)+1)+1)) .* ...
                   Y_dmrs_centered(:,i,bs);
           end
       end
       
       Y_data_perUci = zeros(12,nSym_data,nRxAnt);
       for i = 1 : nSym_data
           for bs = 1 : nRxAnt
               Y_data_perUci(:,i,bs) = conj(tOCC_data(tOCCidx+1,i)) * conj(csPhaseRamp(:,cs(2*(i-1)+2)+1)) .* ...
                   Y_data(:,i,bs);
           end
       end
   end
   
   % DtxThrNum = abs(mean(Y_dmrs_perUci, 'all'))^2;
   
   DTXthreshold = DTXthreshold*alpha*rssiSqrd^0.94/(nUci^(5.25/12));
   if DTXthreshold < 1e-16
       DTXthreshold = 1e-16;
   end
   
   %% TA estimate
   avgScCorr = 0;
   for scIdx = 0:10
       for symIdx = 0:(nSym_dmrs-1)
           for antIdx = 0:(nRxAnt-1)
               avgScCorr = avgScCorr + conj(conj(s(scIdx+1))*Y_dmrs_perUci(scIdx+1, symIdx+1, antIdx+1)) * conj(s(scIdx+2))* Y_dmrs_perUci(scIdx+2, symIdx+1, antIdx+1);
           end
       end
   end
   
   avgScCorr = avgScCorr / (nRxAnt * 11 * nSym_dmrs);
   taEstMicroSec = -10^6 * atan2(imag(avgScCorr), real(avgScCorr)) / (2*pi*scs);

   %% noise estimation
   
   avgNoiseAndIntEnergy = 0;
   for i = 1 : nSym_dmrs
       for bs = 1 : nRxAnt
           n                    = W_noiseIso * Y_dmrs_perUci(:,i,bs); % isolate noise for given gNB ant and DMRS symbol
           avgNoiseAndIntEnergy = avgNoiseAndIntEnergy + sum(abs(n).^2);
       end
   end
   
   avgNoiseAndIntEnergy = avgNoiseAndIntEnergy / (nSym_dmrs * nRxAnt * noiseDim);
   avgSigEnergy         = rssiTemp - avgNoiseAndIntEnergy;
   
   avgNoiseAndIntEnergyDb = 10*log10(avgNoiseAndIntEnergy);
   % Handle situation where noise power is greater than signal and clip SINR
   % at a min and max reportable value
   snr_dB = 10*log10(abs(avgSigEnergy)) - avgNoiseAndIntEnergyDb;
   
   if(avgSigEnergy <= 0)
       snr_dB = -65.6;
   end
   snr_dB = min(65.6,max(snr_dB,-65.6));
  
   rsrpTemp = 0;
   rsrp_dB = 0;
   %% estimate users channel on the data symbols, undo centering
   if freqHopFlag 
       %% determine time domain filter matrices for first and second hops
       switch nSym
           case 4
               Wt{1} = 1;
               Wt{2} = 1;
           case 5
               Wt{1} = 1;
               Wt{2} = 1/2*ones(2,1);
           case 6
               Wt{1} = 1/2*ones(2,1);
               Wt{2} = ones(1,2);
           case 7
               Wt{1} = 1/2*ones(2,1);
               Wt{2} = Wt_matrices{2};
           case 8
               Wt{1} = Wt_matrices{1};
               Wt{2} = Wt_matrices{1};
           case 9
               Wt{1} = Wt_matrices{1};
               Wt{2} = Wt_matrices{4};
           case 10
               Wt{1} = Wt_matrices{4};
               Wt{2} = Wt_matrices{3};
           case 11
               Wt{1} = Wt_matrices{4};
               Wt{2} = Wt_matrices{6};
           case 12
               Wt{1} = Wt_matrices{5};
               Wt{2} = Wt_matrices{5};
           case 13
               Wt{1} = Wt_matrices{5};
               Wt{2} = Wt_matrices{8};
           case 14
               Wt{1} = Wt_matrices{8};
               Wt{2} = Wt_matrices{7};
       end
       
       %%first hop
       for bs = 1 : nRxAnt
           temp_Y_dmrs_perUci = zeros(12,1);
           for i = 1 : nSymDMRSFirstHop
                temp_Y_dmrs_perUci = temp_Y_dmrs_perUci + Y_dmrs_perUci(:, i, bs);
           end
            temp_Y_dmrs_perUci = conj(s) .* temp_Y_dmrs_perUci/nSymDMRSFirstHop;
            rsrpTemp = rsrpTemp + abs(mean(temp_Y_dmrs_perUci))^2;
       end

       H_est_iue = zeros(12,nSymDataFirstHop,nRxAnt);
       
       for bs = 1 : nRxAnt
           H_est_iue(:,:,bs) = (Wf * Y_dmrs_perUci(:,1:nSymDMRSFirstHop,bs)) * Wt{1};
       end
       
       for i = 1 : nSymDataFirstHop
           for bs = 1 : nRxAnt
               H_est_iue(:,i,bs) = conj(s) .* H_est_iue(:,i,bs);
           end
       end
       
       m = Y_data_perUci(:,1:nSymDataFirstHop,:) .* conj(H_est_iue);
       qam_est = sum(m(:));

       %%second hop
       for bs = 1 : nRxAnt
           temp_Y_dmrs_perUci = zeros(12,1);
           for i = 1 : nSymDMRSSecondHop
                temp_Y_dmrs_perUci = temp_Y_dmrs_perUci + Y_dmrs_perUci(:, nSymDMRSFirstHop+i, bs);
           end
           temp_Y_dmrs_perUci = conj(s) .* temp_Y_dmrs_perUci/nSymDMRSSecondHop;
           rsrpTemp = rsrpTemp + abs(mean(temp_Y_dmrs_perUci))^2;
       end

       H_est_iue = zeros(12,nSymDataSecondHop,nRxAnt);
       
       for bs = 1 : nRxAnt
           H_est_iue(:,:,bs) = (Wf * Y_dmrs_perUci(:,(nSymDMRSFirstHop+1):nSym_dmrs,bs)) * Wt{2};
       end
       
       for i = 1 : nSymDataSecondHop
           for bs = 1 : nRxAnt
               H_est_iue(:,i,bs) = conj(s) .* H_est_iue(:,i,bs);
           end
       end

       rsrpTemp = rsrpTemp/nRxAnt/2;
       rsrp_dB = 10*log10(rsrpTemp);
       
       m = Y_data_perUci(:,(nSymDataFirstHop+1):nSym_data,:) .* conj(H_est_iue);
       qam_est = qam_est + sum(m(:));
   else
       
       switch nSym
           case 4
               Wt = Wt_matrices{1};
           case 5
               Wt = Wt_matrices{4};
           case 6
               Wt = Wt_matrices{5};
           case 7
               Wt = Wt_matrices{8};
           case 8
               Wt = Wt_matrices{9};
           case 9
               Wt = Wt_matrices{10};
           case 10
               Wt = Wt_matrices{11};
           case 11
               Wt = Wt_matrices{12};
           case 12
               Wt = Wt_matrices{13};
           case 13
               Wt = Wt_matrices{14};
           case 14
               Wt = Wt_matrices{15};
       end
       
       for bs = 1 : nRxAnt
           temp_Y_dmrs_perUci = zeros(12,1);
           for i = 1 : nSym_dmrs
                temp_Y_dmrs_perUci = temp_Y_dmrs_perUci + Y_dmrs_perUci(:, i, bs);
           end
           temp_Y_dmrs_perUci = conj(s) .* temp_Y_dmrs_perUci/nSym_dmrs;
           rsrpTemp = rsrpTemp + abs(mean(temp_Y_dmrs_perUci))^2;
       end
       rsrpTemp = rsrpTemp/nRxAnt;
       rsrp_dB = 10*log10(rsrpTemp);

       H_est_iue = zeros(12,nSym_data,nRxAnt);
              
       for bs = 1 : nRxAnt
           H_est_iue(:,:,bs) = (Wf * Y_dmrs_perUci(:,:,bs)) * Wt;
       end

       for i = 1 : nSym_data
           for bs = 1 : nRxAnt
               H_est_iue(:,i,bs) = conj(s) .* H_est_iue(:,i,bs);
           end
       end
        
       m = Y_data_perUci .* conj(H_est_iue);
       qam_est = sum(m(:));
   end
   
   %% estimate users bits
   SR = 0;
   
   b_est = [];

   abs2_qam = abs(qam_est)^2;
   
   %% Determine SR/HARQ confidence levels
   %% According to SCF FAPI, Table 3-68, 0 stands for "Good" and 1 stands for "Bad"
   SRconfidenceLevel = 0;
   HarqconfidenceLevel = 0;
   
   confidenceDtxThr = 0.1; % threshold for determining confidence levels of SR and HARQ values
   
   gapPercDtx = (abs2_qam - DTXthreshold)/abs2_qam;
   
   if gapPercDtx < confidenceDtxThr
        SRconfidenceLevel = 1;
        HarqconfidenceLevel = 1;
   end

   %% Detection
   if abs2_qam <= DTXthreshold % DTX detected
       if nBits > 0
           b_est = 2*ones(1, nBits);
       end

       taEstMicroSec = 0;
   else
       if nBits == 0
           if real(qam_est) > 0
               SR = 1;
           end
       else
           SR = srFlag;
           b_est = zeros(1, nBits);
           if nBits == 2
               if real(qam_est) <= 0
                   b_est(1) = 1;
               else
                   b_est(1) = 0;
               end
               
               if imag(qam_est) <= 0
                   b_est(2) = 1;
               else
                   b_est(2) = 0;
               end
           elseif nBits == 1
               a = 1 - 1i;
               qam_est = a*qam_est;
               
               if real(qam_est) <= 0
                   b_est(1) = 1;
               else
                   b_est(1) = 0;
               end
           end
       end
   end

   pUciF1{uciOutputIdx}.taEstMicroSec = taEstMicroSec;
   pUciF1{uciOutputIdx}.SinrDB   = snr_dB;
   pUciF1{uciOutputIdx}.InterfDB = avgNoiseAndIntEnergyDb;
   pUciF1{uciOutputIdx}.RSRP     = rsrp_dB;
   pUciF1{uciOutputIdx}.RSSI     = rssi_dB;
   pUciF1{uciOutputIdx}.HarqValues = zeros(1,2);
   pUciF1{uciOutputIdx}.HarqValues(1 : nBits) = b_est;
   pUciF1{uciOutputIdx}.NumHarq = nBits;
   pUciF1{uciOutputIdx}.SRindication = SR;
   pUciF1{uciOutputIdx}.SRconfidenceLevel = SRconfidenceLevel;
   pUciF1{uciOutputIdx}.HarqconfidenceLevel = HarqconfidenceLevel;
end       

return
