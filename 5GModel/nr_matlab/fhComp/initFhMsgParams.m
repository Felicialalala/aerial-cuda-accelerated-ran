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

function fhMsgParams = initFhMsgParams(sectionType, extType)

switch sectionType
    case 1
        % fhMsgParams.dataDirection = 0;
        % fhMsgParams.payloadVersion = 1;
        % fhMsgParams.filterIndex = 0;
        % fhMsgParams.frameId = 0;
        % fhMsgParams.subframeId = 0;
        % fhMsgParams.slotID = 0;
        fhMsgParams.startSymbolid = 0;
        % fhMsgParams.numberOfsections = 0;
        fhMsgParams.sectionType = sectionType;
        % fhMsgParams.udCompHdr = 0;
        % fhMsgParams.reserved = 0;
        
        % fhMsgParams.sectionId = 0;
        fhMsgParams.rb = 0;
        fhMsgParams.symInc = 0;
        fhMsgParams.startPrbc = 0;
        fhMsgParams.numPrbc = 0;
        fhMsgParams.reMask = ones(1, 12);
        % fhMsgParams.numSymbol = 0;
        % if extType == 0
        %     fhMsgParams.ef = 0;
        % else
        %     fhMsgParams.ef = 1;
        % end
        % fhMsgParams.beamId = 0;
        fhMsgParams.udIqWidth = 0; % 0 -> 16 bits
        fhMsgParams.udCompMeth = 0; % 0 -> no compression
        fhMsgParams.extType = extType;
        switch extType
            case 0 % no compression
                % reserved
            case 4
                fhMsgParams.csf = 1;
                fhMsgParams.modCompScaler = 1;
            case 5
                fhMsgParams.mcScaleReMask = ones(2, 12);
                fhMsgParams.csf = ones(1,2);
                fhMsgParams.mcScaleOffset = ones(1, 2);
            otherwise
                error('extType is not supported ...\n');
        end
        fhMsgParams.csiRsCompression = 0; % reserved for csirs compression
    otherwise
        error('sectionType is not supported .../n');
end

return
