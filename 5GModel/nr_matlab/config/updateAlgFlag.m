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

function SysPar = updateAlgFlag(SysPar)

if isfield(SysPar, 'pusch')
    for idxUe = 1:length(SysPar.pusch)
        DmrsSymbPos = SysPar.pusch{idxUe}.DmrsSymbPos;
        dmrsIdx = find(DmrsSymbPos);
        nDmrs =  length(dmrsIdx);
        if nDmrs > 1
            if dmrsIdx(1) == dmrsIdx(2)-1
                addPos = nDmrs/2-1;
            else
                addPos = nDmrs-1;
            end
        else
            addPos = 0;
        end
        
        nrOfLayers = SysPar.pusch{idxUe}.nrOfLayers;
        if addPos == 0 || nrOfLayers > 4  % || (SysPar.carrier.Nant_gNB > 8)
            SysPar.SimCtrl.alg.TdiMode = 0;
            SysPar.SimCtrl.alg.enableCfoEstimation = 0;
            SysPar.SimCtrl.alg.enableCfoCorrection = 0;
            SysPar.SimCtrl.alg.enableWeightedAverageCfo = 0;
            SysPar.SimCtrl.alg.enableToEstimation = 0;
        end

        if ~isfield(SysPar.SimCtrl,'enable_static_dynamic_beamforming')
            SysPar.SimCtrl.enable_static_dynamic_beamforming = 0;
        end

        if SysPar.SimCtrl.enable_static_dynamic_beamforming
            SysPar.SimCtrl.alg.enableIrc = 1;
        end

        if nrOfLayers > 4 || ((SysPar.carrier.Nant_gNB > 8) && (SysPar.SimCtrl.enable_static_dynamic_beamforming==0))
            SysPar.SimCtrl.alg.enableIrc = 0;
        end
    end
end
                
return
