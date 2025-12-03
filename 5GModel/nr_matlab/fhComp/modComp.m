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

function modCompSamp = modComp(iqSamp, udIqWidth, reMask, scaler, csf)

[nSc, nSym, nPort] = size(iqSamp);
nRB = nSc/12;
modCompSamp = zeros(nSc, nSym, nPort);
nMask = size(reMask, 1);

for idxRB =  1:nRB
    for idxSym = 1:nSym
        for idxPort = 1:nPort
            RB = iqSamp(12*(idxRB-1)+1:12*idxRB, idxSym, idxPort);
            for idxMask = 1:nMask
                usedRe = find(reMask(idxMask,:));
                nRe = length(usedRe);
                % de-scale
                RB1 = RB(usedRe)/(scaler(idxMask) * sqrt(2));
                % shift
                if csf(idxMask) == 1
                    RB1 = RB1-(1+1j)/2^udIqWidth;
                end
                RB2 = round(RB1*2^(udIqWidth-1));
                RB3 = [real(RB2) imag(RB2)];
                RB3 = RB3 + (RB3<0)*2^udIqWidth;
                %RB3 = RB3(:,1) + 1j*RB3(:,2);
                % Concatentate I & Q binary values
                %RB3 = bin2dec(reshape(dec2bin(reshape(RB3',1,[]),udIqWidth)',2*udIqWidth,[])');
                %RB3 = RB3(:,1)*2^udIqWidth + RB3(:,2);
                RB3 = RB3(:,2) + RB3(:,1)*2^udIqWidth; % Q LSb
                modCompSamp(12*(idxRB-1)+usedRe, idxSym, idxPort) = RB3;
            end
        end
    end
end

return