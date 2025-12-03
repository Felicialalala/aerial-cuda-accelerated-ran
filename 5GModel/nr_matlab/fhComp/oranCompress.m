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

function [cSamples_uint8, X_tf_fp16] = oranCompress(X_tf, sim_is_uplink)

global SimCtrl;

X_tf_fp16 = [];
for ii=1:size(X_tf,3)
    X_tf_fp16(:,:,ii) = fp16nv(real(double(X_tf(:,:,ii))), SimCtrl.fp16AlgoSel) + 1i*fp16nv(imag(double(X_tf(:,:,ii))), SimCtrl.fp16AlgoSel);
end

ucSamples = reshape(X_tf_fp16, size(X_tf_fp16,1) *  size(X_tf_fp16,2) *  size(X_tf_fp16,3), 1);

for k=1:length(SimCtrl.oranComp.iqWidth)
    iqWidth = SimCtrl.oranComp.iqWidth(k);
    Ref_c = SimCtrl.oranComp.Ref_c(k);
    FSOffset = SimCtrl.oranComp.FSOffset(k);
    Nre_max = SimCtrl.oranComp.Nre_max;
    max_amp_ul = SimCtrl.oranComp.max_amp_ul;
    
    beta = oranCalcBeta(sim_is_uplink, iqWidth, FSOffset, Ref_c, Nre_max, max_amp_ul);
    % hack to force beta
    if SimCtrl.oranCompressBetaForce == 1
       if iqWidth == 9
           beta = 65536;
       elseif iqWidth == 14
           beta = 2097152;
       end           
    end
        
    if (sim_is_uplink)
        beta = 1/beta;
    end

    bypass = 0;
    cSamples_uint8{k} = fh_bfp_compress(ucSamples * beta, iqWidth, bypass);
end

return
