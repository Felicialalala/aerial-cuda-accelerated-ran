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

function [uciSegPayloads_cell, polCbs_cell, polCbsCrcEncoded_cell, polCws_cell, polCwsRmItl_cell, polUciSegsEncoded_cell] = uci_polar_tx(nPolUciSegs, A_seg, E_seg)

% Function runs uci polar transmitter pipeline using cuPHY matlab model.
% Outputs intermediate buffers.

%inputs:
% nPolUciSegs --> number of polar uci segments to process
% A_seg       --> number of info bits per segment. Dim: nPolUciSegs x 1
% E_seg       --> number of tx bits per segment.   Dim: nPolUciSegs x 1

%outputs:
% uciSegPayloads_cell --> cell containing uci segment payloads. Dim: nPolUciSegs x 1
% polCbsEncoded_cell  --> cell containing crc + polar encoded codeblocks. Dim: totNumPolCbs x  1

%%
% DERIVE UCI PARAMATERS

polarUciSegPrms_cell = cell(nPolUciSegs,1);

for segIdx = 0 : (nPolUciSegs - 1)
    polarUciSegPrms_cell{segIdx + 1} = derive_polarUciSegPrms(A_seg(segIdx + 1), E_seg(segIdx + 1));
end

[totNumPolCbs, polarCbPrms_cell, polarUciSegPrms_cell] = derive_polarCbPrms(nPolUciSegs, polarUciSegPrms_cell);
 
%%
% PAYLOADS

uciSegPayloads_cell = cell(nPolUciSegs,1);
for segIdx = 0 : (nPolUciSegs - 1)
    uciSegPayloads_cell{segIdx + 1} = round(rand(polarUciSegPrms_cell{segIdx + 1}.A_seg, 1));
end

%%
% CB SEGMENT 

polCbs_cell = cell(totNumPolCbs,1);
for segIdx = 0 : (nPolUciSegs - 1)
    polCbs = polarCbSegment(polarUciSegPrms_cell{segIdx + 1}, uciSegPayloads_cell{segIdx + 1});
    
    for i = 1 : polarUciSegPrms_cell{segIdx + 1}.nCbs
        childIdx                  = polarUciSegPrms_cell{segIdx + 1}.childCbIdxs(i);
        polCbs_cell{childIdx + 1} = polCbs(:,i);
    end
end

%%
% CRC ENCODE

polCbsCrcEncoded_cell = cell(totNumPolCbs,1);

for cbIdx = 0 : (totNumPolCbs - 1)
    crcStr = num2str(polarCbPrms_cell{cbIdx + 1}.nCrcBits);
    polCbsCrcEncoded_cell{cbIdx + 1} = add_CRC(polCbs_cell{cbIdx + 1}, crcStr);
end
    

%%
% BIT TYPES

% cw bit types for each uci segment
cwBitTypes_seg_cell  = cell(nPolUciSegs,1);
for segIdx = 0 : (nPolUciSegs - 1)
    cwBitTypes_seg_cell{segIdx + 1} = compBitTypesKernel(polarUciSegPrms_cell{segIdx + 1});
end

% cw bit types for each codeblock
cwBitTypes_cb_cell = cell(totNumPolCbs,1);
for segIdx = 0 : (nPolUciSegs - 1)
    for i = 1 : polarUciSegPrms_cell{segIdx + 1}.nCbs
        childIdx = polarUciSegPrms_cell{segIdx + 1}.childCbIdxs(i);
        cwBitTypes_cb_cell{childIdx + 1} = cwBitTypes_seg_cell{segIdx + 1};
    end
end

%%
% POLAR ENCODE

polCws_cell = cell(totNumPolCbs, 1);

for cbIdx = 0 : (totNumPolCbs - 1)
    polCws_cell{cbIdx + 1} = uplinkPolarCbEncoder(polarCbPrms_cell{cbIdx + 1}, cwBitTypes_cb_cell{childIdx + 1}, polCbsCrcEncoded_cell{cbIdx + 1});
end

%%
% INTERLEAVE + RATEMATCH

polCwsRmItl_cell = cell(totNumPolCbs, 1);

for cbIdx = 0 : (totNumPolCbs - 1)
    polCwsRmItl_cell{cbIdx + 1} = uplinkPolarRmItl(polarCbPrms_cell{cbIdx + 1}, polCws_cell{cbIdx + 1});
end

%%
% CB SEGMENTATION 

polUciSegsEncoded_cell = cell(nPolUciSegs, 1);

for segIdx = 0 : (nPolUciSegs - 1)  
    polarUciSegPrm = polarUciSegPrms_cell{segIdx + 1};
    
    nCbs        = polarUciSegPrm.nCbs;
    E_cw        = polarUciSegPrm.E_cw;
    E_seg       = polarUciSegPrm.E_seg;
    childCbIdxs = polarUciSegPrm.childCbIdxs;
    
    if(nCbs == 1)
        polUciSegsEncoded_cell{segIdx + 1} = polCwsRmItl_cell{childCbIdxs(1) + 1};
    else
        polUciSegsEncoded_cell{segIdx + 1} = zeros(E_seg,1);
        polUciSegsEncoded_cell{segIdx + 1}(1 : E_cw)          = polCwsRmItl_cell{childCbIdxs(1) + 1};
        polUciSegsEncoded_cell{segIdx + 1}(E_cw + 1 : 2*E_cw) = polCwsRmItl_cell{childCbIdxs(2) + 1};
    end
    
    polUciSegsEncoded_cell{segIdx + 1} = uciSegPolarEncode(A_seg(segIdx + 1), E_seg, uciSegPayloads_cell{segIdx + 1});
end




        
    
    



    


















