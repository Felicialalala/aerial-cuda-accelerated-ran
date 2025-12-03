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

function min_SINR = computeBfSinr(bfwPdu, bfwBuf, chanBuf, noiseEnergy_dB, pduIdx, srsChEstBuff)

%%
%PARAMATERS:

noiseEnergy_linear = 10^(noiseEnergy_dB / 10);

nRxAnt = bfwPdu.gnbAntIdxEnd(1) - bfwPdu.gnbAntIdxStart(1) + 1;

nUes = bfwPdu.nUes;
RNTI = bfwPdu.RNTI;

numOfUeAnt = bfwPdu.numOfUeAnt;
nBfLayers  = sum(bfwPdu.numOfUeAnt);

maxNumAntPerUe = 4;
ueAntIdxs      = zeros(maxNumAntPerUe, bfwPdu.nUes);
ueAntIdxs(1,:) = bfwPdu.ueAntIdx0;
ueAntIdxs(2,:) = bfwPdu.ueAntIdx1;
ueAntIdxs(3,:) = bfwPdu.ueAntIdx2;
ueAntIdxs(4,:) = bfwPdu.ueAntIdx3;

bfwPrbGrpSize = bfwPdu.bfwPrbGrpSize;
startPrbGrp = floor(bfwPdu.rbStart / bfwPrbGrpSize);
nPrbGrp     = bfwPdu.numPRGs;

coefBufIdx = pduIdx - 1;

%%
%START

min_SINR = inf;

for prbGrpIdx = 0 : (nPrbGrp - 1)

    % bfw for this prbGrp
    bfw = bfwBuf{coefBufIdx + 1}(:,:,prbGrpIdx + 1);

    % assemble MU-MIMO channel for this prbGrp
    H = zeros(nBfLayers, nRxAnt);
    ueGrpLayerIdx = 0;
    for ueIdx = 0 : (nUes - 1)
        for i = 0 : (numOfUeAnt(ueIdx + 1) - 1)
            
            chEstBufIdx  =  RNTI(ueIdx + 1)-1;
            ueAntIdx     =  ueAntIdxs(i + 1, ueIdx + 1);

            % use the closest valid SRS chEst on an unestimated PRB
            srsPrbGrpSize = srsChEstBuff{chEstBufIdx + 1}.prbGrpSize;
            srsStartValidPrg = srsChEstBuff{chEstBufIdx + 1}.startValidPrg;
            srsNValidPrg     = srsChEstBuff{chEstBufIdx + 1}.nValidPrg;
            bfwToSrs_ueGrpSizeRatio = bfwPrbGrpSize / srsPrbGrpSize;
            srsUeGrpOffset          = floor(bfwToSrs_ueGrpSizeRatio / 2);
            startSrsPrbGrp          = floor(bfwPdu.rbStart / srsPrbGrpSize);
            srsPrgIdx  = startSrsPrbGrp + bfwToSrs_ueGrpSizeRatio * prbGrpIdx + srsUeGrpOffset;
            srs_prbGrpIdx = min(srsStartValidPrg + srsNValidPrg - 1, max(srsPrgIdx, srsStartValidPrg));

            for gnbAntIdx = 0 : (nRxAnt - 1)
                H(ueGrpLayerIdx + 1, gnbAntIdx + 1) = chanBuf{chEstBufIdx + 1}(srs_prbGrpIdx + 1, gnbAntIdx + 1, ueAntIdx + 1);
            end
            ueGrpLayerIdx = ueGrpLayerIdx + 1;
        end
    end

    % compute SINR for this prbGrp
    H_precoded         = H * bfw;
    signalEnergy       = abs(diag(H_precoded)).^2;
    interferenceEnergy = sum(abs(H_precoded).^2,2) - signalEnergy;
    SINR               = 10*log10(signalEnergy ./ (interferenceEnergy + noiseEnergy_linear));

    % update min SINR
    if(min_SINR > min(SINR))
        min_SINR = min(SINR);
    end
end



end
        

