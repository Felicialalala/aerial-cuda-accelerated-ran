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

classdef gen_DCIFormat0_0_payload < MessageFormat
            
    properties
        Identifier                  = BitField(1);        
        FrequencyDomainResources    = BitField(); 
        TimeDomainResources         = BitField(4);
        FrequencyHoppingFlag        = BitField(1);
        ModulationCoding            = BitField(5);
        NewDataIndicator            = BitField(1);
        RedundancyVersion           = BitField(2);
        HARQprocessNumber           = BitField(4);
        TPCcommand                  = BitField(2);
%         UL_SUL_indicator            = BitField();    
    end
         
    methods
        
        function obj = gen_DCIFormat0_0_payload(RIV_len)
        % Class constructor
            obj.FrequencyDomainResources    = BitField(RIV_len); 
        end

    end

end