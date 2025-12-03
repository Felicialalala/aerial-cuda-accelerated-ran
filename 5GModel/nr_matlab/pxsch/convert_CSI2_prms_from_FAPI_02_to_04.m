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

function SysPar = convert_CSI2_prms_from_FAPI_02_to_04(SysPar)

    % number of PUSCH users:
    nPuschUes = length(SysPar.pusch);

    % constants:
    MAX_NUM_CSI1_PRM     = SysPar.SimCtrl.MAX_NUM_CSI1_PRM;
    MAX_NUM_CSI2_REPORTS = SysPar.SimCtrl.MAX_NUM_CSI2_REPORTS;

    for ueIdx = 0 : (nPuschUes - 1)

        %check if csi2 present:
        isCsi2Present = bitand(uint16(SysPar.pusch{ueIdx + 1}.pduBitmap),uint16(2^5));

        if((SysPar.SimCtrl.enable_multi_csiP2_fapiv3 == 0) && (isCsi2Present > 0))

            % paramaters to compute CSI-P2 sizes:
            calcCsi2Size_csi2MapIdx = zeros(MAX_NUM_CSI2_REPORTS , 1);
            calcCsi2Size_nPart1Prms = zeros(MAX_NUM_CSI2_REPORTS , 1);
            calcCsi2Size_prmOffsets = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);
            calcCsi2Size_prmSizes   = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);
            calcCsi2Size_prmValues  = zeros(MAX_NUM_CSI1_PRM     , MAX_NUM_CSI2_REPORTS);

            % first csi2 size calc structure:
            calcCsi2Size_csi2MapIdx(1)   = 0;
            calcCsi2Size_nPart1Prms(1)   = 1;
            calcCsi2Size_prmOffsets(1,1) = SysPar.pusch{ueIdx + 1}.rankBitOffset;
            calcCsi2Size_prmSizes(1,1)   = SysPar.pusch{ueIdx + 1}.rankBitSize;
            calcCsi2Size_prmValues(1,1)  = SysPar.pusch{ueIdx + 1}.rank - 1;

            % reshape to linear buffers, store in pusch:
            SysPar.pusch{ueIdx + 1}.calcCsi2Size_csi2MapIdx = calcCsi2Size_csi2MapIdx(:);
            SysPar.pusch{ueIdx + 1}.calcCsi2Size_nPart1Prms = calcCsi2Size_nPart1Prms(:);
            SysPar.pusch{ueIdx + 1}.calcCsi2Size_prmOffsets = calcCsi2Size_prmOffsets(:);
            SysPar.pusch{ueIdx + 1}.calcCsi2Size_prmSizes   = calcCsi2Size_prmSizes(:);
            SysPar.pusch{ueIdx + 1}.calcCsi2Size_prmValues  = calcCsi2Size_prmValues(:);

            % a single CSI2 report:
            SysPar.pusch{ueIdx + 1}.nCsi2Reports = 1;

            % mark csi2 present in FAPI 10.04 API:
            SysPar.pusch{ueIdx + 1}.flagCsiPart2 = 65535;
        end
    end
end