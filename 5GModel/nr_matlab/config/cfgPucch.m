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

function pucch = cfgPucch

% pucch related config
pucch.BWPSize = 273;
pucch.BWPStart = 0;
pucch.RNTI = 46; 
pucch.FormatType = 0;
pucch.multiSlotTxIndicator = 0;
pucch.pi2Bpsk = 0;
pucch.maxCodeRate = 0;          % in FAPIv4
pucch.prbStart = 1;
pucch.prbSize = 1;
pucch.startSym = 0;             % StartSymbolIndex in PDU
pucch.nSym = 8;                 % NrOfSymbols in PDU
pucch.freqHopFlag = 1;
pucch.secondHopPRB = 200;
pucch.groupHopFlag = 1;         % 0 - disable group hopping / 1 - enable group hopping, compliant with FAPI
pucch.sequenceHopFlag = 0;      % 0 - disable sequence hopping / 1 - enable sequence hopping, compliant with FAPI
pucch.hoppingId = 100;
pucch.cs0 = 2;                  % InitialCyclicShift in PDU (value: 0->11)
pucch.dataScramblingId = 0;
pucch.tOCCidx = 0;              % TimeDomainOccIdx in PDU (Value: 0->6)
pucch.PreDftOccIdx = 0;
pucch.PreDftOccLen = 0;
pucch.AddDmrsFlag = 0;
pucch.DmrsScramblingId = 0;
pucch.DMRScyclicshift = 0;
pucch.digBFInterfaces = 4;
pucch.beamIdx = [1 2 3 4];
pucch.SRFlag = 1; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pucch.BitLenHarq = 0;%Valid for all formats. Value: 0 - no HARQ bit, or 1, 2 bit(s) for Formats 0 and 1, or 2 ~ 1706 bits for Formats 2, 3 and 4
% The following field BitLenSr is named "SrBitLen" in FAPIv10, 
% For PUCCH Format 0/1, Value: 0 - no SR; 1 - SR occasion
% For PUCCH Format 2/3/4, Value: 0 - no SR, or 1, 2, 3, 4 SR bit(s);
% Note: When MAC is implemented according to 3GPP TS 38.213 [4][22], the maximum number of SR bits expected for a single UE is 4.
pucch.BitLenSr = 0;             
pucch.BitLenCsiPart1 = 0; % Bit length of CSI part 1 payload. Valid for formats 2, 3 and 4. Value: 0 - no CSI bit, or 1 ~ 1706 CSI bit(s)
% Note: the summation of BitLenHarq, BitLenSr and BitLenCsiPart1 should be
% no larger than 1706 (maximum Polar codeword size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pucch.positiveSR = 0;
pucch.DTX = 0;
pucch.idxUE = 0;
pucch.DTXthreshold = 1;

pucch.rank           = 1;
pucch.rankBitOffset  = 0;
pucch.rankBitSize    = 2;

% FAPI Release 10, Table 3–95
pucch.UciP1ToP2Crpd.numPart2s = 0;

return
