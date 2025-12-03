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

function gUci= uciSegRMSimplexEncode(K,Euci,payload,Qm)

% This function calculates rate matching output sequence lengths for HARQ,
% CSIPART-1 and CSI PART-2 bitstreams, when carried over PUSCH (with or
% without UL-SCH)and payload size is less than 12 bits

% Ref.: TS 38.212 Sec. 5.3.3.2 and 5.3.3.3

% Input parameters


% K:       Number of info bits 
% Euci:    Rate matched sequence length
% payload: UCI payload size
% qam:     Modulation order

% Output 

% gUci:             Rate matched sequence

%% Encoding + rate matching for K<=11 bit information
if K<=2
   gUci = simplexEncode(payload,K,Euci,Qm); 
else % 3<=K<=11
   fecEnc = FecRmObj(1,Euci,K);
   gUci = fecEnc(payload); 
end
return

   
            
            


