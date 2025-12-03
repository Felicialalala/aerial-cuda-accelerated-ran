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

function SysPar = resetAlloc(SysPar, N_UE)

testAlloc.dl = 0;
testAlloc.ul = 0;
testAlloc.ssb = 0;
testAlloc.pdcch = 0;
testAlloc.pdsch = 0;
testAlloc.csirs = 0;
testAlloc.prach = 0;
testAlloc.pucch = 0;
testAlloc.pusch = 0;
testAlloc.srs = 0;
testAlloc.bfw = 0;

SysPar.testAlloc = testAlloc;
SysPar.SimCtrl.gNB.tx.alloc = [];
SysPar.SimCtrl.gNB.rx.alloc = [];
for idxUE = 1:N_UE
    SysPar.SimCtrl.UE{idxUE}.tx.alloc = [];
    SysPar.SimCtrl.UE{idxUE}.rx.alloc = [];
end

return