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

function layerMap = compute_LayerMap(dmrsPort,nL_tot)

% function computes equalizer output index for a given dmrs port

% inputs:
% dmrsPort --> input dmrs ports
% nL_tot   --> total number of layers

% outpus:
% equalIdx --> equalizer output indies of antenna ports

%%
%START

% compute equalizer indices of each antenna port
switch nL_tot
    case 1
        M = 1;
    case 2
        M = 1 : 2;
    case 3
        M = 1 : 3;        
    case 4
        M = 1 : 4;
    case 8
        M = [1 2 5 6 3 4 7 8];
    otherwise
        M = 1:nL_tot;
%         error('Error: total number of layer unsupported');
end

% M = [1 2 5 6 3 4 7 8];
dmrsPort = mod(dmrsPort-1, nL_tot) + 1;

equalIdx = M(dmrsPort);

% build equalizer output bit-mask
bmsk = zeros(32,1);
bmsk(equalIdx) = 1;

% bit-mask to layer-map
layerMap = 0;
for i = 1 : 32
    layerMap = layerMap + 2^(i-1)*bmsk(i);
end

end



        
        
        
