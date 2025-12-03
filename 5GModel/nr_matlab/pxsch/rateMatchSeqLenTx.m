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

function [G_harq, G_harq_rvd, G_csi1, G_csi2, G]= rateMatchSeqLenTx(nBitsHarq,nBitsCsi1,nBitsCsi2,pduBitmap,alphaScaling,...
                                         betaOffsetHarqAck,betaOffsetCsi1,betaOffsetCsi2,...
                                         nPuschSym,dataSymLoc_array,dmrsSymLoc_array,nPrb,...
                                         Nl,Qm,C,K,codeRate,startSym,numDmrsCdmGrpsNoData)

                                            
% This function calculates rate matching output sequence lengths for HARQ,
% CSI-PART1, CSI-PART2 and ULSCH bitstreams, when carried over PUSCH (with or
% without UL-SCH)

% Ref.: TS 38.212 Sec. 6.3.2.4.1

% Input parameters


% nBitsHarq:            number of HARQ ACK bits (not including CRC bits)==> FAPI parameter "harqAckBitLength"
% nBitsCsi1:            number of CSI Part 1 bits (not including CRC bits) ==> FAPI parameter "csiPart1BitLength"
% nBitsCsi2:            number of CSI Part 2 bits (not including CRC bits) ==> Derived from L2 vendor defined parameters (RI and RI_offset, for example)
% pduBitmap:            pduBitmap in decimal; in binary (uint16) --> bit0:ULSCH, bit1: UCI, bits0&1: UCI+ULSCH; bit 4: CSI-2
% alphaScaling:         alpha parameter(α)to calculate number of coded modulation symbols per layer ==> FAPI parameter
% betaOffsetHarqAck:    beta offset of HARQ ACK [TS38.213, Table 9.3-1]==> FAPI parameter
% betaOffsetCsi1:       beta offset of CSI Part 1 [TS 38.213 Table 9.3-2] ==> FAPI parameter
% betaOffsetCsi2:       beta offset of CSI Part 2 [TS 38.213 Table 9.3-2] ==> FAPI parameter
% nPuschSym:            PUSCH duration in symbols (including DMRS, Ref. TS 38.214 table 6.1.2.1-1==> FAPI parameter "NrOfSymbols")
% dataSymLoc_array:     symbol indexes for PUSCH data allocation(Matlab 1 indexing)
% dmrsSymLoc_array:     symbol indexes for PUSCH dmrs allocation (Matlab 1 indexing)
% nPrb:                 number of PRBs allocated for PUSCH over one symbol==> FAPI parammeter "rbSize" for RA type 1
% Nl:                   number of layers for PUSCH (38.211 Sec. 6.3.1.3) ==> FAPI parameter "nrOfLayers"
% Qm:                   modulation order [TS 38.214 Sec. 6.1.4.1] ==> FAPI parameter "qamModOrder"
% C:                    number of code blocks for UL-SCH
% K:                    number of bits per code block for UL-SCH
% coderate:             coding rate [TS38.214 sec 6.1.4.1]= number of information bits per 1024 coded bits ==> FAPI parameter "targetCodeRate"
% startSym:             Starting symbol index of PUSCH (MATLAB 1 indexing)

% Output 

% G_harq:             rate matched sequence length of ACK
% G_harq_rvd:         rate matched seq. length for reserved HARQ
% G_csi1:             rate matched sequence length for CSI Part 1
% G:                  rate matched sequence length for ULSCH data


%% Calculation of EAck


alpha = alphaScalingToAlphaMapping(alphaScaling);                          % Ref.: SCF FAPI Table 3-48 (optional puschUci information) 

betaOffsetPusch = betaOffsetHarqMapping(betaOffsetHarqAck);                % Ref.: Table 9.3-1, TS 38.213

isDataPresent = bitand(uint16(pduBitmap),uint16(2^0));                     % Bit0 = 1 in pduBitmap,if data is present

mScPuschGrid = ones(nPuschSym,1)*nPrb*12;                                  % array of number of REs on each OFDM symbol in the PUSCH resource grid

mScUlsch = mScPuschGrid;                                                     % initialization of mScUci with mScPuschGrid (number of REs on each PUSCH PFDM symbol available for UCI)
dmrsSymLoc_array_from_startSym = dmrsSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1
dataSymLoc_array_from_startSym = dataSymLoc_array - startSym + 1;          % Converting to MATLAB 1 indexing with +1

if numDmrsCdmGrpsNoData == 1 && isDataPresent
    mScUlsch(dmrsSymLoc_array_from_startSym,:) = 0.5 * mScUlsch(dmrsSymLoc_array_from_startSym,:); 
else
    mScUlsch(dmrsSymLoc_array_from_startSym,:) = 0;                              % No REs are available for UCI on DMRS symbols
end
mUlsch = sum(mScUlsch(:));      

mScUci = mScPuschGrid;              % #NOTE: Assuming nDmrsCdmGroupWithoutData = 2, no REs on DMRS symbols available for PUSCH data
mScUci(dmrsSymLoc_array_from_startSym,:) = 0;   
mScUciSum = sum(mScUci(:));                                                % Total number of REs available for UCI transmission
l0 = dataSymLoc_array_from_startSym(find(dmrsSymLoc_array_from_startSym(1)<dataSymLoc_array_from_startSym,1));       % first OFDM symbol (Matlab 1 indexing) after the first set of consecutive OFDM symbol(s) carrying DMRS
mScUciSumFroml0 = sum(mScUci(l0:end));   

CUlsch = ones(C,1);                                                        % Asssumption: all code blocks are transmitted, C_ULsch is the number of code blocks for UL-SCH of the PUSCH transmission
Kr = K;
CodedBitsSum = sum(Kr*CUlsch);                                             % Summation of K_r, r=0,..,(C_ULSCH-1), the denominator of first term in Q'_ACK [Ref. TS 38.212 Sec. 6.3.2.4.1.1]

% Number of reserved ACK bits, Ref. 38.212 Sec. 6.2.7 (Step 1)
if nBitsHarq <=2
    nBitsAckRvd = 2;
else
    nBitsAckRvd =0;
end

% Bit capacity, and modulation symbol capacity (per layer) and rate matched
% output sequence length for HARQ-ACK payload

[qPrimeAck_rvd,G_harq_rvd] = rateMatchAck(isDataPresent,nBitsAckRvd,betaOffsetPusch,mScUciSum,CodedBitsSum,alpha,mScUciSumFroml0,Nl,Qm,codeRate);
[qPrimeAck,G_harq] = rateMatchAck(isDataPresent,nBitsHarq,betaOffsetPusch,mScUciSum,CodedBitsSum,alpha,mScUciSumFroml0,Nl,Qm,codeRate);


%% Calculation of ECsi1

if nBitsCsi1
    betaOffsetPusch = betaOffsetCsiMapping(betaOffsetCsi1); % Ref.: Table 9.3-2, TS 38.213
    lCsi1 = crcLength(nBitsCsi1);
    
% Estimation of Q^prime_ACK_CSI-1

    if nBitsHarq>2
        QPrimeAckCsi1 = qPrimeAck;
    else
       QPrimeAckCsi1 = qPrimeAck_rvd;
    end
    
% Bit capacity, and modulation symbol capacity (per layer) and rate matched
% output sequence length for CSI Part 1 payload

    firstTermCsi1 = firstTerm(isDataPresent,nBitsCsi1,lCsi1,betaOffsetPusch,mScUciSum,CodedBitsSum,codeRate,Qm);
    
    if isDataPresent
        qPrimeCsi1 = min(firstTermCsi1,ceil(alpha*mScUciSum)-QPrimeAckCsi1);   % CSI-1 modulation symbol capacity per layer
     else
        if nBitsCsi2
            qPrimeCsi1 = min(firstTermCsi1,mScUciSum- QPrimeAckCsi1);
        else
            qPrimeCsi1 = mScUciSum- QPrimeAckCsi1;
        end
    end
    
    G_csi1 = Nl*qPrimeCsi1*Qm;                 % CSI-1 bit capacity
    
else
     qPrimeCsi1 = 0;
     G_csi1   = 0;
end
%% Calculation of ECsi2

if nBitsCsi2
    betaOffsetPusch = betaOffsetCsiMapping(betaOffsetCsi2); % Ref.: Table 9.3-2, TS 38.213
    lCsi2 = crcLength(nBitsCsi2);
    
% Estimation of Q^prime_ACK_CSI-1

    if nBitsHarq >2
        QPrimeAckCsi2 = qPrimeAck;
    else
       QPrimeAckCsi2  = 0;
    end

% Bit capacity, and modulation symbol capacity (per layer) and rate matched
% output sequence length for CSI Part 2 payload

    firstTermCsi2 = firstTerm(isDataPresent,nBitsCsi2,lCsi2,betaOffsetPusch,mScUciSum,CodedBitsSum,codeRate,Qm);
    
    if isDataPresent
        qPrimeCsi2 = min(firstTermCsi2,ceil(alpha*mScUciSum)-QPrimeAckCsi2-qPrimeCsi1);   % CSI-2 modulation symbol capacity per layer
    else
        qPrimeCsi2 = mScUciSum- QPrimeAckCsi2-qPrimeCsi1;
    end
    
    G_csi2 = Nl*qPrimeCsi2*Qm;                 % CSI-1 bit capacity
else
     G_csi2   = 0;
end 

%% Calculation of EUlsch
G_Ulsch = mUlsch * Qm * Nl;      % bit capacity of UL-SCH without UCI

if isDataPresent
    G = G_Ulsch -G_harq*(nBitsAckRvd ==0)-G_csi1-G_csi2;
else
    G = 0;
end

return

%% Auxilliary functions
function [Qack, EAck] = rateMatchAck(isDataPresent,oAck,betaOffsetPusch,mScUciSum,CodedBitsSum,alpha,mScUciSumFroml0,nl,qam,codeRate)

lAck = crcLength(oAck);                                     % Ref. 38.212 Sec. 6.3.1.2.1 and 6.3.1.2.2
                                          
firstTermAck = firstTerm(isDataPresent,oAck,lAck,betaOffsetPusch,mScUciSum,CodedBitsSum,codeRate,qam); 
secondTermAck = ceil(alpha * mScUciSumFroml0);
Qack = min(firstTermAck,secondTermAck);

EAck = nl*Qack*qam;

return

%==========================================================================
function lCRC = crcLength(nBits)

if nBits <=11
    lCRC = 0;
elseif (nBits >=12) && (nBits <=19)
    lCRC =6;
else
    lCRC = 11;
end
return
%==========================================================================        

function firstTermValue = firstTerm(isDataPresent,oUci,lUci,betaOffsetPusch,mScUciSum,CodeBlockSizeSum,R,Qm)

if isDataPresent
    numerator = (oUci + lUci)*betaOffsetPusch*mScUciSum;
    denominator = CodeBlockSizeSum;
else
 numerator = (oUci + lUci)*betaOffsetPusch;
 denominator = R*Qm; 
end

firstTermValue = ceil (numerator/denominator);
return

%==========================================================================        
     
function alpha = alphaScalingToAlphaMapping(alphaScaling)

switch alphaScaling  
    case 0
        alpha = 0.5;
    case 1
        alpha = 0.65;
    case 2
        alpha = 0.8;
    case 3
        alpha = 1;
    otherwise
        error ('alpha scaling value not supported...\');
end
return
%==========================================================================        

function betaOffsetHarqValue = betaOffsetHarqMapping(betaOffsetHarqAck)

switch betaOffsetHarqAck
    case 0
       betaOffsetHarqValue = 1.000; 
    case 1
       betaOffsetHarqValue = 2.000;
       case 2
       betaOffsetHarqValue = 2.500; 
    case 3
       betaOffsetHarqValue = 3.125;
    case 4
        betaOffsetHarqValue = 4.000;
    case 5
        betaOffsetHarqValue = 5.000;
    case 6
        betaOffsetHarqValue = 6.250;
    case 7
        betaOffsetHarqValue = 8.000;
    case 8
        betaOffsetHarqValue = 10.000;
    case 9
        betaOffsetHarqValue = 12.625;
    case 10
        betaOffsetHarqValue = 15.875;
    case 11
        betaOffsetHarqValue = 20.000;
    case 12
        betaOffsetHarqValue = 31.000;
    case 13
        betaOffsetHarqValue = 50.000;
    case 14
        betaOffsetHarqValue = 80.000;
    case 15
        betaOffsetHarqValue =126.000;
    otherwise
        error('betaOffsetHarqAckValue is either reserved or not supported...\');
end
return
%==========================================================================        

function betaOffsetCsi1Value = betaOffsetCsiMapping(betaOffsetCsi)

switch betaOffsetCsi
    case 0
       betaOffsetCsi1Value = 1.125;
    case 1
        betaOffsetCsi1Value = 1.250;
    case 2
         betaOffsetCsi1Value = 1.375;
    case 3
        betaOffsetCsi1Value = 1.625;
    case 4
        betaOffsetCsi1Value = 1.750;
    case 5
        betaOffsetCsi1Value = 2.000;
    case 6
        betaOffsetCsi1Value = 2.250;
    case 7
        betaOffsetCsi1Value = 2.500;
    case 8
        betaOffsetCsi1Value = 2.875;
    case 9
        betaOffsetCsi1Value = 3.125;
    case 10
        betaOffsetCsi1Value = 3.500;
    case 11
        betaOffsetCsi1Value = 4.000;
    case 12
        betaOffsetCsi1Value = 5.000;
    case 13
        betaOffsetCsi1Value = 6.250;
    case 14
        betaOffsetCsi1Value = 8.000;
    case 15
        betaOffsetCsi1Value = 10.000;
    case 16
        betaOffsetCsi1Value = 12.625;
    case 17
        betaOffsetCsi1Value = 15.875;
    case 18 
        betaOffsetCsi1Value = 20.000;
    otherwise
        error('betaOffsetCsiValue is either reserved or not supported...\');
end
return
