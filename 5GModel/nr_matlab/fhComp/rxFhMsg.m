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

function Xtf = rxFhMsg(fhMsg, carrier)

% CSI-RS can potentially have a different number of ports, so check here
% Apply CSI-RS compression logic when calculating maxPorts
maxPorts = carrier.numTxPort;
for fhIdx = 1:length(fhMsg);
    portIdx = fhMsg{fhIdx}.header.portIdx;
    % Check if CSI-RS compression is active
    if isfield(fhMsg{fhIdx}.header, 'csiRsCompression') && fhMsg{fhIdx}.header.csiRsCompression > 0
        portIdx = maxPorts - 1;
    end
    maxPorts = max([maxPorts portIdx+1]);
end

Xtf = zeros(12*carrier.N_grid_size_mu, 14, maxPorts);

selectiveRe = false;
nMsg = length(fhMsg);
bw_scale = sqrt(size(Xtf,1));

for idxMsg = 1:nMsg
    msgParams = fhMsg{idxMsg}.header;
    startSymbolid = msgParams.startSymbolid;
    startPrbc = msgParams.startPrbc;
    numPrbc = msgParams.numPrbc;
    reMask = msgParams.reMask;
    rb = msgParams.rb;
    udIqWidth = msgParams.udIqWidth;
    portIdx = msgParams.portIdx;
    % Reduce CSI-RS ports if needed
    portIdx = mod(portIdx, maxPorts);
    extType = msgParams.extType;
    switch extType
        case 0
            iqSamp = fhMsg{idxMsg}.payload;
            Xtf(12*startPrbc+1:12*(startPrbc+numPrbc), startSymbolid+1, portIdx+1) = iqSamp;
        case {4, 5}
            if extType == 4
                scaler = msgParams.modCompScaler;
            else
                reMask = msgParams.mcScaleReMask ;
                scaler = msgParams.mcScaleOffset;
            end
            % Remove BW Scaling before computing compressed bits
            scaler = scaler*bw_scale;
            csf = msgParams.csf;
            %modCompSamp = fhMsg{idxMsg}.payload;
            modCompSamp = bin2dec(reshape(dec2bin(fhMsg{idxMsg}.payload,8)',udIqWidth*2,[])');
            if(selectiveRe)
                numSamp = fhMsg{idxMsg}.header.numPrbc*sum(reMask);
                % Drop any padded segments
                modCompSamp = modCompSamp(1:numSamp);
                repMask = repmat(reMask,[1,numSamp/sum(reMask)])';
                expandSamp = zeros(size(repMask));
                expandSamp(repMask==1) = modCompSamp;
                iqSamp = modDecomp(expandSamp, udIqWidth, reMask, scaler, csf);
            else
                iqSamp = modDecomp(modCompSamp, udIqWidth, reMask, scaler, csf);
            end

            if rb == 0
                Xtf(12*startPrbc+1:12*(startPrbc+numPrbc), startSymbolid+1, portIdx+1) = iqSamp + ...
                    Xtf(12*startPrbc+1:12*(startPrbc+numPrbc), startSymbolid+1, portIdx+1);
            else                
                for idxPrbc = 0:numPrbc-1
                    Xtf(12*(startPrbc+2*idxPrbc)+1:12*(startPrbc+2*idxPrbc+1), startSymbolid+1, portIdx+1) = ...
                        iqSamp(12*idxPrbc+1:12*(idxPrbc+1), :, :) + ...
                        Xtf(12*(startPrbc+2*idxPrbc)+1:12*(startPrbc+2*idxPrbc+1), startSymbolid+1, portIdx+1);
                end
            end
    end
end

return