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

function x = xor_butterfly(x,s)

for j = s : -1 : 1

  for i = 0 : (2^s-1)
      if mod(i,2^j) < 2^(j-1)
        c0 = x(i+1);
        c1 = x(i+2^(j-1)+1);

        x(i+1) = xor(c0,c1);
      end
  end

end