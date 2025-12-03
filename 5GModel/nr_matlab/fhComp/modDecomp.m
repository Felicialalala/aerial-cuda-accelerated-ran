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

function iqSamp = modDecomp(modCompSamp, udIqWidth, reMask, scaler, csf)

[nSc, nSym, nPort] = size(modCompSamp);
nRB = nSc/12;
iqSamp = zeros(nSc, nSym, nPort);
nMask = size(reMask, 1);

for idxRB =  1:nRB
    for idxSym = 1:nSym
        for idxPort = 1:nPort
            RB = modCompSamp(12*(idxRB-1)+1:12*idxRB, idxSym, idxPort);
            for idxMask = 1:nMask
                usedRe = find(reMask(idxMask, :));
                nRe = length(usedRe);
                RB1 = RB(usedRe);
                %RB1 = [floor(RB1/(2^udIqWidth)) mod(RB1,2^udIqWidth)];
                RB1 = [mod(RB1,2^udIqWidth) floor(RB1/(2^udIqWidth))];
                RB1 = RB1 - (RB1 >= 2^(udIqWidth-1))*2^udIqWidth;
                RB1 = RB1(:,2) + 1j* RB1(:,1);
                
                % de-shift
                RB2 = RB1/2^(udIqWidth-1);
                if csf(idxMask) == 1
                    RB2 = RB2 + (1+1j)/2^udIqWidth;
                end
                
                % scaler
                RB3 = RB2 * scaler(idxMask) * sqrt(2);
                iqSamp(12*(idxRB-1)+usedRe, idxSym, idxPort) = RB3;
            end
        end
    end
end

return