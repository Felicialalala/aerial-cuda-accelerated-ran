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

function [u, C_v] = findZcPar(prmbIdx, rootSequenceIndex, ...
    L_RA, restrictedSet, N_CS, logIdx2u_table)
%
% find u and C_v which will be used to generate ZC sequence
%
v = 0;
prmbIdxCounter = 0;
logIdx = rootSequenceIndex;

%% Derive logIdx and v from prmbIdx
d_u_in_range = 1;
switch restrictedSet
    case 0
        if N_CS == 0
            logIdx = mod(logIdx + prmbIdx, L_RA - 1);
        else
            countCS = floor(L_RA/N_CS);
            logIdx = mod(logIdx + floor(prmbIdx/countCS), L_RA - 1);
            v = prmbIdx - floor(prmbIdx/countCS)*countCS;
        end
        u = logIdx2u_table(logIdx+1);
        C_v = v*N_CS;
    % typeA and typeB are used for high speed channel with doppler, which can
    % creates false peak for correlation based preamble detection, by skipping 
    % some cyclic shift positions
    case 1
        for iter = 0:L_RA-1
            u = logIdx2u_table(logIdx+1);
            [d_u, q] = findDu(u, L_RA);
            if (d_u >= N_CS) && (d_u < L_RA/3)
                n_shift_RA = floor(d_u/N_CS);
                d_start = 2*d_u + n_shift_RA*N_CS;
                n_group_RA = floor(L_RA/d_start);
                n1_shift_RA = max(floor((L_RA - 2*d_u - n_group_RA*d_start)...
                    /N_CS),0);
                d_u_in_range = 1;
            elseif (d_u >= L_RA/3) && (d_u <= (L_RA - N_CS)/2)
                n_shift_RA = floor((L_RA - 2*d_u)/N_CS);
                d_start = L_RA - 2*d_u + n_shift_RA*N_CS;
                n_group_RA = floor(d_u/d_start);
                n1_shift_RA = min(max(floor((d_u-n_group_RA*d_start)/N_CS),0), ...
                    n_shift_RA);
                d_u_in_range = 1;
            else
                n_shift_RA = 0;
                d_start = 0;
                n_group_RA = 0;
                n1_shift_RA = 0;
                d_u_in_range = 0;
            end
            if d_u_in_range
                countCS = n_shift_RA*n_group_RA + n1_shift_RA;
                prmbIdxCounter = prmbIdxCounter + countCS;
                if prmbIdxCounter - 1 >= prmbIdx
                    v = prmbIdx - (prmbIdxCounter - countCS);
                    break;
                else
                    logIdx = mod(logIdx + 1,L_RA - 1);
                end 
            else                
%                 logIdx = mod(logIdx + prmbIdx, L_RA - 1);
%                 break;
                prmbIdxCounter = prmbIdxCounter + 1;                
                v = 0;
                if prmbIdxCounter -1 >= prmbIdx
                    break;
                else
                    logIdx = mod(logIdx + 1, L_RA - 1);
                end
            end
        end
    case 2
        for iter = 0:L_RA-1
            u = logIdx2u_table(logIdx+1);
            [d_u, q] = findDu(u, L_RA);        
            if (d_u >= N_CS) && (d_u < L_RA/5)
                n_shift_RA = floor(d_u/N_CS);
                d_start = 4*d_u + n_shift_RA*N_CS;
                n_group_RA = floor(L_RA/d_start);
                n1_shift_RA = max(floor((L_RA - 4*d_u - n_group_RA*d_start)/...
                    N_CS),0);
                n2_shift_RA = 0;
                n3_shift_RA = 0;
                d_u_in_range = 1;
            elseif (d_u >= L_RA/5) && (d_u <= (L_RA-N_CS)/4)
                n_shift_RA = floor((L_RA - 4*d_u)/N_CS);
                d_start = L_RA - 4*d_u + n_shift_RA*N_CS;
                n_group_RA = floor(d_u/d_start);
                n1_shift_RA = min(max(floor((d_u - n_group_RA*d_start)/N_CS),0), ...
                    n_shift_RA);
                n2_shift_RA = 0;
                n3_shift_RA = 0;
                d_u_in_range = 1;
            elseif (d_u >= (L_RA + N_CS)/4) && (d_u < 2*L_RA/7)
                n_shift_RA = floor((4*d_u - L_RA)/N_CS);
                d_start = 4*d_u - L_RA + n_shift_RA*N_CS;
                n_group_RA = floor(d_u/d_start);
                n1_shift_RA = max(floor((L_RA - 3*d_u - n_group_RA*d_start)/...
                    N_CS),0);            
                d2_start = L_RA - 3*d_u + n_group_RA*d_start + n1_shift_RA*N_CS;
                n2_shift_RA = floor(min(d_u-n_group_RA*d_start, 4*d_u - L_RA - ...
                    n1_shift_RA*N_CS)/N_CS);            
                d3_start = L_RA - 2*d_u + n_group_RA*d_start + n2_shift_RA*N_CS;
                n3_shift_RA = floor(((1 - min(1, n1_shift_RA))*(d_u - n_group_RA ...
                    *d_start) + min(1, n1_shift_RA)*(4*d_u - L_RA - n1_shift_RA ...
                    *N_CS))/N_CS) - n2_shift_RA;
                d_u_in_range = 1;
            elseif (d_u >= 2*L_RA/7) && (d_u <= (L_RA - N_CS)/3)
                n_shift_RA = floor((L_RA - 3*d_u)/N_CS);
                d_start = L_RA - 3*d_u + n_shift_RA*N_CS;
                n_group_RA = floor(d_u/d_start);
                n1_shift_RA = max(floor((4*d_u - L_RA - n_group_RA*d_start)/...
                    N_CS),0);
                d2_start = d_u + n_group_RA*d_start + n1_shift_RA*N_CS;
                n2_shift_RA = floor(min(d_u - n_group_RA*d_start, L_RA - 3*d_u - ...
                    n1_shift_RA*N_CS)/N_CS);
                d3_start = 0;
                n3_shift_RA = 0;
                d_u_in_range = 1;
            elseif (d_u >= (L_RA + N_CS)/3) && (d_u < 2*L_RA/5)
                n_shift_RA = floor((3*d_u - L_RA)/N_CS);
                d_start = 3*d_u - L_RA + n_shift_RA*N_CS;
                n_group_RA = floor(d_u/d_start);
                n1_shift_RA = max(floor((L_RA - 2*d_u - n_group_RA*d_start)/...
                    N_CS),0);
                d2_start = 0;
                n2_shift_RA = 0;            
                d3_start = 0; 
                n3_shift_RA = 0;
                d_u_in_range = 1;
            elseif (d_u >= 2*L_RA/5) && (d_u <= (L_RA-N_CS)/2)
                n_shift_RA = floor((L_RA - 2*d_u)/N_CS);
                d_start = 2*(L_RA - 2*d_u) + n_shift_RA*N_CS;
                n_group_RA = floor((L_RA-d_u)/d_start);
                n1_shift_RA = max(floor((3*d_u - L_RA - n_group_RA*d_start)/...
                    N_CS),0);
                d2_start = 0; 
                n2_shift_RA = 0;
                d3_start = 0; 
                n3_shift_RA = 0;      
                d_u_in_range = 1;
            else
                n_shift_RA = 0;
                d_start = 0;
                n_group_RA = 0;
                n1_shift_RA = 0;
                d2_start = 0; 
                n2_shift_RA = 0;
                d3_start = 0; 
                n3_shift_RA = 0;                  
                d_u_in_range = 0;
            end
            if d_u_in_range
                countCS = n_shift_RA*n_group_RA + n1_shift_RA + ...
                    n2_shift_RA + n3_shift_RA;
                prmbIdxCounter = prmbIdxCounter + countCS;
                if prmbIdxCounter - 1 >= prmbIdx
                    v = prmbIdx - (prmbIdxCounter - countCS);
                    break;
                else
                    logIdx = mod(logIdx + 1,L_RA - 1);
                end 
            else
                prmbIdxCounter = prmbIdxCounter + 1;                
                v = 0;
                if prmbIdxCounter -1 >= prmbIdx
                    break;
                else
                    logIdx = mod(logIdx + 1, L_RA - 1);
                end
            end        
        end
    otherwise
        error('restricted set type is not supported ... \n');
end

%% Derive u
u = logIdx2u_table(logIdx+1);

%% Derive C_v
switch restrictedSet
    case 0
        if N_CS == 0
            C_v = 0;
        elseif v <= floor(L_RA/N_CS)-1
            C_v = v*N_CS;
        else
            error('v of unrestricated set is out of range ... \n'); 
        end
    case 1       
        if d_u_in_range == 1
            w = n_shift_RA*n_group_RA + n1_shift_RA;
            if v <= w - 1 
                C_v = d_start*floor(v/n_shift_RA) + mod(v, n_shift_RA)*N_CS;
            else
                error('v of type A is out of range ... \n');
            end         
        else
            C_v = 0;
        end
    case 2
        if d_u_in_range == 1
            w = n_shift_RA*n_group_RA + n1_shift_RA;
            if v <= w - 1 
                C_v = d_start*floor(v/n_shift_RA) + mod(v, n_shift_RA)*N_CS;
            elseif v <= w + n2_shift_RA - 1
                C_v = d2_start + (v - w)*N_CS;
            elseif v <= w + n2_shift_RA + n3_shift_RA - 1
                C_v = d3_start + (v - w - n2_shift_RA)*N_CS;
            else
                error('v of type B is out of range ...\n');
            end                 
        else
            C_v = 0;
        end
    otherwise
        error('restricted set type is not supported ... \n');
end
