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

function H_int = apply_interpolation(H_est, symIdx_data, symIdx_dmrs, numDmrsCdmGrpsNoData)

global SimCtrl;
fp_flag = SimCtrl.fp_flag_pusch_equalizer;

if symIdx_dmrs(2) - symIdx_dmrs(1) == 1
    symIdx_dmrs = symIdx_dmrs(1:2:end);
end    

nDmrs = length(symIdx_dmrs);
alpha1 = Varray(zeros(14,1), fp_flag);
alpha2 = Varray(zeros(14,1), fp_flag);
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
    d = Varray(symIdx_dmrs(dmrs2) - symIdx_dmrs(dmrs1),fp_flag);
    alpha1(idxSym) = Varray((symIdx_dmrs(dmrs2)-idxSym),fp_flag)/d;
    alpha2(idxSym) = Varray((idxSym - symIdx_dmrs(dmrs1)),fp_flag)/d;
    dmrsIdx1(idxSym) = dmrs1;
    dmrsIdx2(idxSym) = dmrs2;
end

if numDmrsCdmGrpsNoData == 1
    symIdx_data = [min(symIdx_data(1), symIdx_dmrs(1)) : max(symIdx_data(end), symIdx_dmrs(end))];
end

if strcmp(SimCtrl.alg.tdi1_alg,'linInterp')
    for posSym = 1:length(symIdx_data)
        idxSym          = symIdx_data(posSym);
        tmp             = alpha1(idxSym)*Varray(H_est{dmrsIdx1(idxSym)},fp_flag) + alpha2(idxSym)*Varray(H_est{dmrsIdx2(idxSym)},fp_flag);
        H_int{posSym}   = tmp.value;
    end
elseif strcmp(SimCtrl.alg.tdi1_alg, 'avg')
    tmp_avg = mean(cat(4,H_est{:}),4);
    for posSym = 1:length(symIdx_data)
        H_int{posSym} = tmp_avg;
    end
elseif strcmp(SimCtrl.alg.tdi1_alg, 'nearest')  
    for posSym = 1:length(symIdx_data)
        idxSym = symIdx_data(posSym);
        [~,ii] = min(abs(idxSym-symIdx_dmrs));
        H_int{posSym} = H_est{ii};
    end
elseif strcmp(SimCtrl.alg.tdi1_alg, 'avg_linInterp_avg')  
    for posSym = 1:length(symIdx_data)
        idxSym = symIdx_data(posSym);
        tmp_avg = mean(cat(4,H_est{:}),4);
        if (idxSym<=symIdx_dmrs(1)) || (idxSym>=symIdx_dmrs(end))
            H_int{posSym} = tmp_avg;
        else
            H_int{posSym} = alpha1(idxSym)*H_est{dmrsIdx1(idxSym)} + alpha2(idxSym)*H_est{dmrsIdx2(idxSym)};
        end
    end
else
    error('Undefined TDI alg!')
end

return