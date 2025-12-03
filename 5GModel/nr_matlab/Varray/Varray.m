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

classdef Varray
    
    properties%(Access = private)
        value;
        fp;
    end

    methods
        function obj = Varray(val,fp_flag) % constructor
            if nargin < 2
                obj.fp      = 0;%'fp64'; % ieee754
                warning("[Varray.m] No floating point format was specified! Set to 0, i.e., 'fp64matlab' by default.")
            else
                obj.fp      = fp_flag;
            end
            switch obj.fp
                case 0      % FP64
                    obj.value  = double(val);           % obj.value  = util_quant_fp(val, 11, 52); % 11 Exponent bits; 52 Mantissa bits
                case 1      % FP32
                    obj.value  = double(single(val));   % obj.value  = util_quant_fp(val, 8, 23);
                case 2      % TF32
                    obj.value  = util_quant_fp(val, 8, 10);
                case 3      % FP16
                    obj.value  = util_quant_fp(val, 5, 10);
                case 4      % BF16
                    obj.value  = util_quant_fp(val, 8, 7);
                case 5      % FP8 E4M3
                    obj.value  = util_quant_fp(val, 4, 3);
                case 6      % FP8 E5M2
                    obj.value  = util_quant_fp(val, 5, 2);
            end            
        end
        
        function d = getValue(obj)                          % get value, return value in double format
            if isa(obj, 'Varray')
                d   = double(obj.value);
            else
                d = obj;
            end
        end
        
        function r = plus(obj1, obj2)                       % plus operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a + b, obj1.fp);
        end
        
        function r = minus(obj1, obj2)                    % minus operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a - b, obj1.fp);
        end
        
        function r = uminus(obj1)                           % unary minus operator
            a   = getValue(obj1);
            r   = Varray(-a, obj1.fp);
        end
        
        function r = uplus(obj1)                              % unary plus operator
            a   = getValue(obj1);
            r   = Varray(+a, obj1.fp);
        end
        
        function r = times(obj1, obj2)                     % element-wise multiplication operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a .* b, obj1.fp);
        end
        
        function r = mtimes(obj1, obj2)                  % matrix multiplication operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a * b, obj1.fp);
        end
        
        function r = pagemtimes(obj1,obj2,obj3,obj4)
            if nargin == 2
                a = getValue(obj1);
                b = getValue(obj2);
                r = Varray(pagemtimes(a, b),obj1.fp);
            elseif nargin == 4
                a = getValue(obj1);
                b = getValue(obj3);
                r = Varray(pagemtimes(a, obj2, b, obj4),obj1.fp);
            end
        end

        function r = rdivide(obj1, obj2)                   % right element-wise division operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a ./ b, obj1.fp);
        end
        
        function r = ldivide(obj1, obj2)                    % left element-wise division operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a .\ b, obj1.fp);
        end
        
        function r = mrdivide(obj1, obj2)
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a / b, obj1.fp);
        end
        
        function r = mldivide(obj1, obj2)
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a \ b, obj1.fp);
        end
        
        function r = sqrt(obj1)
            a   = getValue(obj1);
            r   = Varray(sqrt(a), obj1.fp);
        end

        function r = log2(obj1)
            a   = getValue(obj1);
            r   = Varray(log2(a), obj1.fp);
        end

        function r = log10(obj1)
            a   = getValue(obj1);
            r   = Varray(log10(a), obj1.fp);
        end

        function r = real(obj1)
            a   = getValue(obj1);
            r   = Varray(real(a), obj1.fp);
        end

        function r = imag(obj1)
            a   = getValue(obj1);
            r   = Varray(imag(a), obj1.fp);
        end

        function r = floor(obj1)
            a   = getValue(obj1);
            r   = Varray(floor(a), obj1.fp);
        end

        function r = ceil(obj1)
            a   = getValue(obj1);
            r   = Varray(ceil(a), obj1.fp);
        end

        function r = round(obj1)
            a   = getValue(obj1);
            r   = Varray(round(a), obj1.fp);
        end

        function r = diag(obj1)
            a   = getValue(obj1);
            r   = Varray(diag(a), obj1.fp);
        end

        function r = chol(obj1, tri_flag, fp_flag)
            r = Varray_util_chol_3d(obj1, tri_flag, fp_flag);
        end

        function r = tril(obj1,val) 
            a = getValue(obj1);
            if nargin==1
                r = Varray(tril(a));
            else
                r = Varray(tril(a,val));
            end            
        end

        function [L,D,U] = ldl(obj1, fp_flag) 
            [L,D,U] = Varray_util_LDL_3d(obj1, fp_flag);
        end

        function r = forward_sub(obj1, obj2, fp_flag) 
            r = Varray_util_forward_sub_3d(obj1, obj2, fp_flag);
        end

        function r = backward_sub(obj1, obj2, fp_flag) 
            r = Varray_util_backward_sub_3d(obj1, obj2, fp_flag);
        end

        function r = inv_tri(obj1, tri_flag, fp_flag) 
            r = Varray_util_inv_tri_3d(obj1,tri_flag, fp_flag);
        end

        function r = power(obj1, obj2)                    % element-wise power operator
            if isa(obj2, 'Varray')
                b   = getValue(obj2);
                fp_flag = obj2.fp;
            else
                b = obj2;
            end
            if isa(obj1, 'Varray')
                a   = getValue(obj1);
                fp_flag = obj1.fp;
            else
                a = obj1;
            end
             
            r   = Varray(a .^ b, fp_flag);
        end       
        
        function r = lt(obj1, obj2)                            % element-wise less-than operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a < b, obj1.fp);         
        end
        
        function r = gt(obj1, obj2)                          % element-wise great-than operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a > b, obj1.fp);         
        end
        
        function r = le(obj1, obj2)                           % element-wise less-than or equal-to operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a <= b, obj1.fp);         
        end
        
        function r = ge(obj1, obj2)                          % element-wise great-than or equal-to operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a >= b, obj1.fp);         
        end
        
        function r = ne(obj1, obj2)                          % element-wise not-equal-to operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a ~= b, obj1.fp);         
        end
        
        function r = eq(obj1, obj2)                          % element-wise equal-to operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray(a == b, obj1.fp);         
        end
        
        function r = and(obj1, obj2)                        % element-wise logical AND operator
            r   = Varray(obj1.value & obj2.value, obj1.fp);         
        end
        
        function r = or(obj1, obj2)                          % element-wise logical OR operator
            r   = Varray(obj1.value | obj2.value, obj1.fp);         
        end
        
        function r = not(obj1)                                 % element-wise logical Not operator
            r   = Varray(~obj1.value, obj1.fp);         
        end
        
        function r = colon(obj1, obj2, obj3)             % colon operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            c   = getValue(obj3);
            if narg == 2
                r   = Varray(a : b); 
            elseif narg == 3
                r   = Varray(a : b : c);   
            else
                error('Undefined!')
            end
        end
        
        function r = ctranspose(obj1)                     % conjugate transpose operator
            a   = getValue(obj1);
            r   = Varray(a', obj1.fp);         
        end

        function r = pagectranspose(obj1)                     % conjugate transpose operator
            a   = getValue(obj1);
            r   = Varray(pagectranspose(a), obj1.fp);         
        end
        
        function r = transpose(obj1)                       % transpose operator
            a   = getValue(obj1);
            r   = Varray(a.', obj1.fp);         
        end
        
        function r = horzcat(obj1, obj2)                  % horizontal concatenation operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray([a,  b], obj1.fp);         
        end
        
        function r = vertcat(obj1, obj2)                   % vertical concatenation operator
            a   = getValue(obj1);
            b   = getValue(obj2);
            r   = Varray([a ; b], obj1.fp);         
        end
        
        function r = subsref(obj1, index)                % subscripted reference operator
            if strcmp(index.type, '.')
                r   = builtin('subsref', obj1, index);  % get class memeber
            else
                a   = getValue(obj1);
                a_sub = a(index.subs{:});
                r = Varray(a_sub, obj1.fp);
%                 r   = Varray(a(index.subs{:}));
%                 r   = Varray(subsref(a,index));                
%                 r   = Varray(builtin('subsref', a, index)); 
            end
        end
        
        function r = subsasgn(obj1, index, asigned_val) % subscripted assignment operator
            if isa(asigned_val, 'Varray')
                a   = getValue(obj1);
                c   = getValue(asigned_val);
                a(index.subs{:}) = c;
                r = Varray(a, obj1.fp);
%                 obj1.value = builtin('subsasgn',a, index, c);
%                 r   = Varray(obj1.value);         
            else
                obj1.value = builtin('subsasgn', obj1.value, index, double(asigned_val));
                r   = Varray(obj1.value, obj1.fp);
            end            
        end
        
        function r = conj(obj1)                                          % conjugate function
            a   = getValue(obj1);
            r = Varray(conj(a), obj1.fp);
        end
        
        function r = sum(obj1,dim)                                          % sum function
            a   = getValue(obj1);
            r = Varray(sum(a,dim), obj1.fp);
        end
        
        function r = mean(obj1, dim)                                          % mean function
%             import matlab.datafun.mean.*
            a   = getValue(obj1);
            r = Varray(mean(a,dim), obj1.fp);
        end
        
%         function r = min(obj1)                                          % min function
%             a   = getValue(obj1);
%             [minValue, idx] = builtin('min', a);
%             r = [minValue, idx] ;
%         end
%         
%         function r = max(obj1)                                          % max function
%             a   = getValue(obj1);
%             [maxValue, idx] = builtin('max', a);
%             r = [maxValue, idx] ;
%         end
        
        function r = len(obj1)                                          % get length function
            a   = getValue(obj1);
            r = Varray(builtin('length', a)); % Need to change!
        end
        
        function r = abs(obj1)                                          % get absolute function
            a   = getValue(obj1);
            r = Varray(abs(a), obj1.fp); % Need to change!
        end
        
        function r = reshape(obj1, dim)                             % reshape function
            a   = getValue(obj1);
            r = Varray(reshape(a, dim), obj1.fp);
        end
        
        function r = permute(obj1, dim)                             % permute function
            a   = getValue(obj1);
            r = Varray(permute(a, dim), obj1.fp);    
        end

        function r = repmat(obj1, dim)                             % reshape function
            a   = getValue(obj1);
            r = Varray(repmat(a, dim), obj1.fp);
        end
        
        function r = circshift(obj, shiftValue)
            a   = getValue(obj);
            r = Varray(circshift(a, shiftValue), obj1.fp);  
        end
    end
    
end




