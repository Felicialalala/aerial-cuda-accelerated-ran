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

function [L_RA, delta_f_RA, N_u, N_CP_RA] = readPreambleFormatTable ...
    (preambleFormat, mu, isLastRaSlot)
%
% define preamble length and subcarrier spacing
%

k_const = 64;

% select A1/A2/A3 or B1/B2/B3
switch preambleFormat
    case 'A1/B1'
        if isLastRaSlot
            preambleFormat = 'B1';
        else
            preambleFormat = 'A1';
        end
    case 'A2/B2'
        if isLastRaSlot
            preambleFormat = 'B2';
        else
            preambleFormat = 'A2';
        end        
    case 'A3/B3'
        if isLastRaSlot
            preambleFormat = 'B3';
        else
            preambleFormat = 'A3';
        end
end 

% 3GPP TS 38.211 V15.4.0
% Table 6.3.3.1-1 and Table 6.3.3.1-2
switch preambleFormat
    % Format '0'-'3': long seqNodence
    % Support unrestricted sets, typeA and typeB
    case '0' % similar to LTE
        L_RA = 839;
        delta_f_RA = 1250;
        N_u = 24576*k_const;
        N_CP_RA = 3168*k_const;
    case '1'
        L_RA = 839;
        delta_f_RA = 1250;
        N_u = 2*24576*k_const;
        N_CP_RA = 21024*k_const;
    case '2'
        L_RA = 839;
        delta_f_RA = 1250;
        N_u = 4*24576*k_const;
        N_CP_RA = 4688*k_const;
    case '3' % for high speed channel
        L_RA = 839;
        delta_f_RA = 5000;
        N_u = 4*6144*k_const;
        N_CP_RA = 3168*k_const;
    % Format 'A1'-'C2': short sequence
    % Support unrestricted sets only
    case 'A1'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 2*2048*k_const*2^(-mu);
        N_CP_RA = 288*k_const*2^(-mu);
    case 'A2'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 4*2048*k_const*2^(-mu);
        N_CP_RA = 576*k_const*2^(-mu);
    case 'A3'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 6*2048*k_const*2^(-mu);
        N_CP_RA = 864*k_const*2^(-mu);
    case 'B1'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 2*2048*k_const*2^(-mu);
        N_CP_RA = 216*k_const*2^(-mu);
    case 'B2'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 4*2048*k_const*2^(-mu);
        N_CP_RA = 360*k_const*2^(-mu);
    case 'B3'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 6*2048*k_const*2^(-mu);
        N_CP_RA = 504*k_const*2^(-mu);
    case 'B4'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 12*2048*k_const*2^(-mu);
        N_CP_RA = 936*k_const*2^(-mu);
    case 'C0'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 2048*k_const*2^(-mu);
        N_CP_RA = 1240*k_const*2^(-mu);
    case 'C2'
        L_RA = 139;
        delta_f_RA = 15000*2^mu;
        N_u = 4*2048*k_const*2^(-mu);
        N_CP_RA = 2048*k_const*2^(-mu);
    otherwise
        error('preambleFormat is not supported ... \n')
end    
