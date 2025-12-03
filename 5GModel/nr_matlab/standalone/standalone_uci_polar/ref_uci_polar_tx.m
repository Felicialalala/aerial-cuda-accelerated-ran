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

function [polCbs_cell, polCbsCrcEncoded_cell, polCws_cell, polCwsRmItl_cell, polUciSegsEncoded_cell] = ref_uci_polar_tx(nPolUciSegs, A_seg, E_seg, uciSegPayloads_cell)

% Function runs uci polar transmitter pipeline using cuPHY matlab model.
% Outputs intermediate buffers.

%inputs:
% nPolUciSegs         --> number of polar uci segments to process
% A_seg               --> number of info bits per segment.  Dim: nPolUciSegs x 1
% E_seg               --> number of tx bits per segment.    Dim: nPolUciSegs x 1
% uciSegPayloads_cell --> cell containing uci seg payloads. Dim: nPolUciSegs x 1 

%outputs:
% polCbs_cell            --> cell containing polar codeblocks. Dim: totNumCbs x 1
% polCbsCrcEncoded_cell  --> cell containing crc-encoded polar codeblocks. Dim: totNumCbs x 1 
% polCws_cell            --> cell containing polar codewords. Dim: totNumCbs x 1 
% polCwsRmItl_cell       --> cell containing rate-match and interleaved polar codeword. Dim: totNumCbs x 1
% polUciSegsEncoded_cell --> cell containing polar encoded uci segments. Dim: nPolUciSegs x 1 


%%
%START

maxNumCbs              = 1000;
polCbs_cell            = cell(maxNumCbs, 1);
polCbsCrcEncoded_cell  = cell(maxNumCbs, 1);
polCws_cell            = cell(maxNumCbs, 1);
polCwsRmItl_cell       = cell(maxNumCbs, 1);
polUciSegsEncoded_cell = cell(nPolUciSegs, 1);

totNumCbs = 0;

for segIdx = 0 : (nPolUciSegs - 1)
    msg = polar_encoder_pucch(A_seg(segIdx + 1), E_seg(segIdx + 1), uciSegPayloads_cell{segIdx + 1}.');
    
    for i = 1 : msg.C
        polCbs_cell{totNumCbs + 1}           = msg.a(i,:)';
        polCbsCrcEncoded_cell{totNumCbs + 1} = msg.b(i,:)';
        polCws_cell{totNumCbs + 1}           = msg.cw(i,:)';
        polCwsRmItl_cell{totNumCbs + 1}      = msg.cwRmItl(i,:)';
        totNumCbs                            = totNumCbs + 1;
    end
    
    polUciSegsEncoded_cell{segIdx + 1} = PUCCH_encoder( uciSegPayloads_cell{segIdx + 1}.', E_seg(segIdx + 1)).';
end

polCbs_cell           = polCbs_cell(1 : totNumCbs);
polCbsCrcEncoded_cell = polCbsCrcEncoded_cell(1 : totNumCbs);
polCws_cell           = polCws_cell(1 : totNumCbs);



end



        














