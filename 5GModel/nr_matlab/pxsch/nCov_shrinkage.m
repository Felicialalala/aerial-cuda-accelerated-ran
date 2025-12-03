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

function nCov_shrinked = nCov_shrinkage(Rtmp, T, nCov_shrinkage_method)
    nAnt = size(Rtmp, 1);
    if nCov_shrinkage_method==0 % RBLW 
        rho_adaptive = real( ( (T-2)/T*trace(Rtmp*Rtmp) + (trace(Rtmp))^2 ) / ( (T+2)*(trace(Rtmp*Rtmp)-(trace(Rtmp))^2/nAnt) ) );
    elseif nCov_shrinkage_method==1 % OAS
        rho_adaptive = real( ( (1-2/nAnt)*trace(Rtmp*Rtmp) + (trace(Rtmp))^2 ) / ( (T+1-2/nAnt)*(trace(Rtmp*Rtmp)-(trace(Rtmp))^2/nAnt) ) );
    end
    rho_adaptive = min(rho_adaptive,1);
    nCov_shrinked = (1-rho_adaptive)*Rtmp + rho_adaptive*real(trace(Rtmp))/nAnt*eye(nAnt);
end             