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

classdef FecRmObj < matlab.System
    % 5G Reed Muller FEC Encoder/Decoder System Object
    %

    % Pre-computed constants
    properties(Access = private)
        encoderMode = 0; % 1 - encode; 0 - decode
        C; % matrix of all possible codewords (with fixed length 32)
        N = 32; % fixed codeword length before rate-matching
        E = 32; % length of rate-matched code sequence
        K = 11; % number of original information bits
        G = [ ... % codebook
            1	1	0	0	0	0	0	0	0	0	1;
            1	1	1	0	0	0	0	0	0	1	1;
            1	0	0	1	0	0	1	0	1	1	1;
            1	0	1	1	0	0	0	0	1	0	1;
            1	1	1	1	0	0	0	1	0	0	1;
            1	1	0	0	1	0	1	1	1	0	1;
            1	0	1	0	1	0	1	0	1	1	1;
            1	0	0	1	1	0	0	1	1	0	1;
            1	1	0	1	1	0	0	1	0	1	1;
            1	0	1	1	1	0	1	0	0	1	1;
            1	0	1	0	0	1	1	1	0	1	1;
            1	1	1	0	0	1	1	0	1	0	1;
            1	0	0	1	0	1	0	1	1	1	1;
            1	1	0	1	0	1	0	1	0	1	1;
            1	0	0	0	1	1	0	1	0	0	1;
            1	1	0	0	1	1	1	1	0	1	1;
            1	1	1	0	1	1	1	0	0	1	0;
            1	0	0	1	1	1	0	0	1	0	0;
            1	1	0	1	1	1	1	1	0	0	0;
            1	0	0	0	0	1	1	0	0	0	0;
            1	0	1	0	0	0	1	0	0	0	1;
            1	1	0	1	0	0	0	0	0	1	1;
            1	0	0	0	1	0	0	1	1	0	1;
            1	1	1	0	1	0	0	0	1	1	1;
            1	1	1	1	1	0	1	1	1	1	0;
            1	1	0	0	0	1	1	1	0	0	1;
            1	0	1	1	0	1	0	0	1	1	0;
            1	1	1	1	0	1	0	1	1	1	0;
            1	0	1	0	1	1	1	0	1	0	0;
            1	0	1	1	1	1	1	1	1	0	0;
            1	1	1	1	1	1	1	1	1	1	1;
            1	0	0	0	0	0	0	0	0	0	0;
        ];
    end

    methods(Access = protected)
        function setupImpl(obj)
            if obj.E <= 32
                M = obj.E;
            else
                M = 32;
            end
            % Truncate generate matrix if rate-matched code length < 32
            obj.G = obj.G(1:M,1:obj.K);
            
            % Build every codeword, one per row
            C = zeros(2^obj.K,M);
            for k=0:2^obj.K-1
                x = transpose(dec2bin(k,obj.K)); % no need to use Communication_Toolbox
                %x = transpose(de2bi(k,obj.K,'left-msb'));
                c = mod(obj.G * x,2);
                C(k+1,:) = transpose(c);
            end
            obj.C = C;
        end

        function [y, z] = stepImpl(obj,x)
            z = [];
            if (obj.encoderMode)
                y = obj.encode(x);
            else
                [y, z] = obj.decode(x);
            end
        end

        function resetImpl(obj)
            % Unused reset function, must be present per matlab.System API
        end

        function y = encode(obj,x)
            if obj.E <= 32
                % Generate matrix encoding
                y = mod(obj.G * x,2);
            else
                y = zeros(obj.E, 1);
                
                y(1:32) = mod(obj.G * x,2);
                
                %% rate-match
                for bitIdx = 33:obj.E
                    IdxTemp = mod(bitIdx-1, 32)+1;
                    
                    y(bitIdx) = y(IdxTemp);
                end
            end
        end

        function [xhat, confLevel] = decode(obj, r)
            % ML decoding, r inputs assumed to be LLRs
            
            global SimCtrl
            dtxModePf2 = SimCtrl.alg.dtxModePf2;

            if dtxModePf2
                obj.C = 1-2*obj.C;
            end
            
            if obj.E <= 32
                m = obj.C * r;
            else
                rDeRateMatched = zeros(32, 1);
                rDeRateMatched = r(1:32);
                for bitIdx = 33:obj.E
                    IdxTemp = mod(bitIdx-1, 32)+1;
                    
                    rDeRateMatched(IdxTemp) = rDeRateMatched(IdxTemp) + r(bitIdx);
                end
                
                m = obj.C * rDeRateMatched;
            end
            
            if dtxModePf2 == 0
                [min_m, xhat_idx] = min(m);
                confLevel = 1;
            else
                [max_m, xhat_idx] = max(m);
                confLevel = (max_m/obj.E)^2/mean(abs(r).^2);
            end
            
            xhat_idx = xhat_idx - 1; % remove matlab 1-based indexing effect
            xhat = transpose(dec2bin(xhat_idx, obj.K)); % no need to use Communication_Toolbox
            %xhat = transpose(de2bi(xhat_idx,obj.K,'left-msb'));
        end
    end

    methods(Access = public)

        function obj = FecRmObj(encoderMode, E, K)
            % Inputs: 
            %        encoderMode:  indication for encode/decode mode, % 1 - encode; 0 - decode
            %        E:            length of rate-matched code sequence
            %        K:            number of original information bits
            
            if (encoderMode)
                obj.encoderMode = 1;
            end
            obj.E = E;
            obj.K = K;
        end

    end
end
