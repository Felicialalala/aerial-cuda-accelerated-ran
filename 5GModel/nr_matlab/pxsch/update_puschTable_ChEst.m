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

function puschTable = update_puschTable_ChEst(puschTable,table)
    
    puschTable.W_lower = table.W_lower;
    puschTable.W_middle = table.W_middle;
    puschTable.W_upper = table.W_upper;
    
    puschTable.W4_upper = table.W4_upper;
    puschTable.W4_middle = table.W4_middle;
    puschTable.W4_lower = table.W4_lower;
    
    puschTable.W3 = table.W3;
    puschTable.W2 = table.W2;
    puschTable.W1 = table.W1;

    puschTable.shiftSeq = table.shiftSeq;
    puschTable.shiftSeq4 = table.shiftSeq4;
    puschTable.unShiftSeq = table.unShiftSeq;
    puschTable.unShiftSeq4 = table.unShiftSeq4;
end