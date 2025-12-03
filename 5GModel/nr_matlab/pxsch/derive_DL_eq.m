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

function eq_coef = derive_DL_eq(dmrs_ch, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData)

[nRe, nSym, nLayer, nAnt] = size(dmrs_ch);

nDmrs = length(symIdx_dmrs);
addPos = 1;
if nDmrs == 1
    addPos = 0;
else
    if symIdx_dmrs(2) - symIdx_dmrs(1) == 1
        symIdx_dmrs = symIdx_dmrs(1:2:end);
        if nDmrs == 2
            addPos = 0;
        end
    end
end
nDmrs = length(symIdx_dmrs);

if addPos > 0
    for idxSym = 1:14
        if idxSym <= symIdx_dmrs(1)
            dmrs1 = 1;
            dmrs2 = 2;
        elseif idxSym >= symIdx_dmrs(nDmrs)
            dmrs1 = nDmrs-1;
            dmrs2 = nDmrs;
        else
            idxDmrs = find((symIdx_dmrs > idxSym), 1);
            dmrs1 = idxDmrs-1;
            dmrs2 = idxDmrs;
        end
        d = symIdx_dmrs(dmrs2) - symIdx_dmrs(dmrs1);
        alpha1(idxSym) = (symIdx_dmrs(dmrs2)-idxSym)/d;
        alpha2(idxSym) = (idxSym - symIdx_dmrs(dmrs1))/d;
        dmrsIdx1(idxSym) = dmrs1;
        dmrsIdx2(idxSym) = dmrs2;
    end
end

if numDmrsCdmGrpsNoData == 1
    symIdx_data = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
end

data_ch = zeros(nRe, length(symIdx_data), nLayer, nAnt);
for posSym = 1:length(symIdx_data)
    if addPos > 0
        idxSym = symIdx_data(posSym);
        data_ch(:, posSym, :, :) = alpha1(idxSym)*dmrs_ch(:, dmrsIdx1(idxSym), :, :) + alpha2(idxSym)*dmrs_ch(:, dmrsIdx2(idxSym), :, :);
    else
        data_ch(:, posSym, :, :) = dmrs_ch(:, 1, :, :);
    end
end

eq_coef = [];
for posSym = 1:length(symIdx_data)
    for posRe = 1:nRe
        H = squeeze(data_ch(posRe, posSym, :, :));
        if size(H, 2) == 1
            H = H.'; 
        end
        eq = H'*inv(H*H');
        eq_coef(posRe, posSym, :, :) = eq;
    end
end

return
