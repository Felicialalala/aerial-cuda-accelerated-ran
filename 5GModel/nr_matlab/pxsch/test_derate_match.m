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

function test_derate_match(rvIdx)
%rvIdx = 3;
%clear all;
%close all;

c = 1;
E(c) = 192;
Qm = 2;
N_cb = 400;
F = 32;
K_prime = 48;
Zc = 8;
K = 64+2*Zc;
BGN = 2;

if BGN == 1
    switch rvIdx
        case 0
            k0 = 0;
        case 1
            k0 = floor(17*N_cb/(66*Zc))*Zc;
        case 2
            k0 = floor(33*N_cb/(66*Zc))*Zc;
        case 3
            k0 = floor(56*N_cb/(66*Zc))*Zc;
        otherwise
            error('rv is not supported...\n');
    end
elseif BGN == 2
    switch rvIdx
        case 0
            k0 = 0;
        case 1
            k0 = floor(13*N_cb/(50*Zc))*Zc;
        case 2
            k0 = floor(25*N_cb/(50*Zc))*Zc;
        case 3
            k0 = floor(43*N_cb/(50*Zc))*Zc;
        otherwise
            error('rv is not supported...\n');
    end
else
    error('BGN is not supported...\n');
end

derateCbsIndex1 = zeros(E(c),1);
derateCbsIndex2 = zeros(E(c),1);

Kcuda = K - 2*Zc; % rate-matching kernel definition of K has 2*Zc punctured bits removed
Kd = Kcuda - F;



k = 0;
j = 0;

while k < E(c)
    %avoid filler bits:
    idx = mod(k0+j,N_cb) + 1;
    if (( idx <= (K_prime - 2*Zc) ) || ( idx > (K - 2*Zc) ))
        derateCbsIndex1(k+1,c) = idx;
        %display([num2str(j),' : ',num2str(idx),' : ',num2str(k)]);
        k = k + 1;
    else
        %display(['Skipping ',num2str(j),' : ',num2str(idx)]);
    end
    j = j + 1;
end


for tid=0:(E(c)-1)
    j = floor(tid / Qm);
    k = tid - j * Qm;
    inIdx = k * E(c) / Qm + j;
    if ((k0 > 0) || (E(c) > N_cb))
        %offset = mod(k * E(c) / Qm + j + k0, N_cb - F);
        offset = mod(k * E(c) / Qm + j + k0, N_cb);
        if ((offset < Kd) || (k0 > Kcuda))
            outIdx = 2 * Zc + offset;
        else
            outIdx = 2 * Zc + (offset + F);
        end
    else
        if(k * E(c) / Qm + j < Kd)
            outIdx = 2 * Zc + k * E(c) / Qm + j;
        else
            outIdx = 2 * Zc + k * E(c) / Qm + j + F;
        end
    end
    %outIdx = derateCbsIndex1(inIdx+1,c) + 2*Zc-1;

    if (1)
        k = 0;
        j = 0;
        while k < (inIdx+1)
            %avoid filler bits:
            idx = mod(k0+j,N_cb) + 1;
            if (( idx <= (Kd) ) || ( idx > (K - 2*Zc) ))
                k = k + 1;
            end
            j = j + 1;
        end
        outIdx = idx + 2*Zc-1;
    end



    derateCbsIndex2(inIdx+1,c) = outIdx;
end

test1 = derateCbsIndex1(derateCbsIndex1(:,c) ~= 0,c);
test2 = derateCbsIndex2(derateCbsIndex2(:,c) ~= 0,c);
test2 = test2 - 2*Zc + 1;
test_err = sum(abs(test1-test2));
if (test_err > 0)
    warning('De-rate-match indexing failure');
end
