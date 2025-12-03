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

function [rssiReportedDb, rssiReportedDb_ehq, rssiDb] = measureRssi(Xtf,measStartPrb,nMeasPrb,measSymIdxs, maxLength)

   NscPerPrb = 12;
   
   % +1 for 1-based indexing
   startScIdx = (measStartPrb-1)*NscPerPrb + 1;
   endScIdx   = (((measStartPrb-1) + nMeasPrb)*NscPerPrb - 1) + 1;
   scIdxs     = startScIdx:endScIdx;
   
   absVals   = abs(Xtf(scIdxs,measSymIdxs,:));
   avgPwrLin = sum(absVals.^2,1); % accumulate across allocated PRBs
   avgPwrLin = reshape(avgPwrLin, [size(avgPwrLin, 2), size(avgPwrLin, 3)]); % remove the first dimension which is a frequency dimension
   
   rssiDb   = 10*log10(avgPwrLin); % linear to dB. Per symbol, per Rx antenna measurement
   
   rssi = sum(mean(avgPwrLin,1),2);% average across all symbols and sum across all Rx antenna
                                   % 5G FAPI Table 3-16: "RSSI reported will be total received power summed across all antennas"
   rssiReportedDb = 10*log10(rssi);
   % for early-HARQ report %
   absVals_ehq   = abs(Xtf(scIdxs,measSymIdxs(1:maxLength),:));
   avgPwrLin_ehq = sum(absVals_ehq.^2,1); % accumulate across allocated PRBs
   avgPwrLin_ehq = reshape(avgPwrLin_ehq, [size(avgPwrLin_ehq, 2), size(avgPwrLin_ehq, 3)]); % remove the first dimension which is a frequency dimension
   rssi_ehq = sum(mean(avgPwrLin_ehq,1),2);% average across all symbols and sum across all Rx antenna
                                   % 5G FAPI Table 3-16: "RSSI reported will be total received power summed across all antennas"
   rssiReportedDb_ehq = 10*log10(rssi_ehq);
return
