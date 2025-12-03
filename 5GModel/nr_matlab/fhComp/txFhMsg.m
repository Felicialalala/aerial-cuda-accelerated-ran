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

function [fhMsg, modCompMsg] = txFhMsg(fhMsgParams, Xtf, modCompList)

% derive modCompMsg for SE4/5 

nList = length(modCompList);
modCompMsg = [];
selectiveRe = false;
bw_scale = sqrt(size(Xtf,1));

for idxList = 1:nList
    thisModComp = modCompList(idxList);
    if thisModComp.nMask == 1
        extType = 4;
    else
        extType = 5;
    end
    sectionType = 1;
    thisFhMsg = initFhMsgParams(sectionType, extType);
    thisFhMsg.udCompMeth = 4; 
    thisFhMsg.startSymbolid = thisModComp.idxSym-1;
    thisFhMsg.startPrbc = thisModComp.idxPrb-1;
    thisFhMsg.numPrbc = thisModComp.nPrb;
    thisFhMsg.portIdx = thisModComp.idxPort-1;
    thisFhMsg.udIqWidth = thisModComp.udIqWidth;
    thisFhMsg.reMask = sum(thisModComp.reMask,1);
    if extType == 4
        thisFhMsg.modCompScaler = thisModComp.scaler(1);
        thisFhMsg.csf = thisModComp.csf(1);
        thisFhMsg.chanType = thisModComp.chanType(1);
    else
        thisFhMsg.nMask = thisModComp.nMask; 
        thisFhMsg.mcScaleReMask = thisModComp.reMask;
        thisFhMsg.csf = thisModComp.csf;
        thisFhMsg.mcScaleOffset = thisModComp.scaler;
        thisFhMsg.chanType = thisModComp.chanType;
    end
    
    % Apply CSI-RS compression logic for proper port indexing (same as SE4 section)
    portIdx = thisFhMsg.portIdx;
    % Check if this might be CSI-RS with compression by looking for csiRsCompression in fhMsgParams
    csiRsCompression = 0;
    if ~isempty(fhMsgParams)
        % Find matching fhMsg parameters to get csiRsCompression
        for idxFh = 1:length(fhMsgParams)
            fhParam = fhMsgParams{idxFh};
            if any(thisFhMsg.chanType == 4) && fhParam.startSymbolid == thisFhMsg.startSymbolid && ...
               fhParam.portIdx == thisFhMsg.portIdx && ...
               isfield(fhParam, 'csiRsCompression') && fhParam.csiRsCompression
                csiRsCompression = fhParam.csiRsCompression;
                break;
            end
        end
    end
    if csiRsCompression
	portIdx = mod(portIdx, size(Xtf,3));
    end
    % Store csiRsCompression in the header for rxFhMsg to use
    thisFhMsg.csiRsCompression = csiRsCompression;
    
    iqSamp = Xtf(12*thisFhMsg.startPrbc+1:12*(thisFhMsg.startPrbc+thisFhMsg.numPrbc), ...
        thisFhMsg.startSymbolid+1, portIdx+1);
    modCompSamp = modComp(iqSamp, thisModComp.udIqWidth, thisModComp.reMask, thisModComp.scaler*bw_scale, thisModComp.csf);
    bitstr = reshape(dec2bin(modCompSamp,thisModComp.udIqWidth*2)',[],1)';
    numPad = mod(8 - mod(length(bitstr),8),8);
    bitstr = [bitstr repmat('0',1,numPad)];
    payload_bytes = bin2dec(reshape(bitstr,8,[])');
    %modCompMsg{idxList}.payload = modCompSamp;
    modCompMsg{idxList}.payload = payload_bytes;
    modCompMsg{idxList}.header = thisFhMsg;
end
            

% derive fhMsg for SE4 only

nMsgParams = length(fhMsgParams);

startSymbolid = [];
for idxMsgParams = 1:nMsgParams
    startSymbolid(idxMsgParams) = fhMsgParams{idxMsgParams}.startSymbolid;
end

[~, idxSort] = sort(startSymbolid);
fhMsg = [];

for idxMsg = 1:nMsgParams
    idx = idxSort(idxMsg);
    msgParams = fhMsgParams{idx};
    fhMsg{idxMsg}.header = msgParams;
    startSymbolid = msgParams.startSymbolid;
    startPrbc = msgParams.startPrbc;
    numPrbc = msgParams.numPrbc;
    reMask = msgParams.reMask;
    rb = msgParams.rb;
    udIqWidth = msgParams.udIqWidth;
    portIdx = msgParams.portIdx;
    if isfield(msgParams, 'csiRsCompression') && msgParams.csiRsCompression
        portIdx = mod(portIdx, size(Xtf,3));
    end
    extType = msgParams.extType;
    switch extType
        case 0
            iqSamp = Xtf(12*startPrbc+1:12*(startPrbc+numPrbc), startSymbolid+1, portIdx+1);
            fhMsg{idxMsg}.payload = iqSamp;
        case 4
            if rb == 0
                iqSamp = Xtf(12*startPrbc+1:12*(startPrbc+numPrbc), startSymbolid+1, portIdx+1);
            else
                iqSamp = [];
                for idxPrbc = 0:numPrbc-1
                    iqSamp = [iqSamp; Xtf(12*(startPrbc+2*idxPrbc)+1:12*(startPrbc+2*idxPrbc+1), startSymbolid+1, portIdx+1)];
                end
            end
            scaler = msgParams.modCompScaler;
            csf = msgParams.csf;
            modCompSamp = modComp(iqSamp, udIqWidth, reMask, scaler*bw_scale, csf);
            if(selectiveRe && length(modCompSamp) == length(reMask))
                modCompSamp = modCompSamp(reMask==1);
            end
            bitstr = reshape(dec2bin(modCompSamp,udIqWidth*2)',[],1)';
            numPad = mod(8 - mod(length(bitstr),8),8);
            bitstr = [bitstr repmat('0',1,numPad)];
            payload_bytes = bin2dec(reshape(bitstr,8,[])');
            %fhMsg{idxMsg}.payload = modCompSamp;
            fhMsg{idxMsg}.payload = payload_bytes;
    end
end

return
