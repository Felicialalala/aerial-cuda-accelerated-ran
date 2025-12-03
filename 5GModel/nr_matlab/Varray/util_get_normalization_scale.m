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

function scale_val = util_get_normalization_scale(x)

    x_vec = x(:);
    if isa(x(1), 'Varray')
        if isreal(getValue(x_vec(1)))
            x_vec_cat = x_vec;
        else
            x_vec_cat = vertcat(real(x_vec), imag(x_vec));
        end
        x_vec_cat_nonzero = x_vec_cat(x_vec_cat.value~=0);
    else
        if isreal(x_vec(1))
            x_vec_cat = x_vec;
        else
            x_vec_cat = vertcat(real(x_vec), imag(x_vec));
        end
        x_vec_cat_nonzero = x_vec_cat(x_vec_cat~=0);
    end
        
    exponent_val = ceil(log2(abs(x_vec_cat_nonzero)));
    mean_exponent_val = mean(exponent_val);
    scale_val = power(2,(-round(mean_exponent_val)));

end