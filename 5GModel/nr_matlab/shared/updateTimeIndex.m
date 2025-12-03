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

function carrier = updateTimeIndex(carrier, idxSlot, forceSlotIdxFlag)

idxSubframe = carrier.idxSubframe;
idxFrame = carrier.idxFrame;

if idxSlot == carrier.N_slot_subframe_mu
    idxSlot = 0;
    idxSubframe = idxSubframe + 1;
    if idxSubframe == carrier.N_subframe
        idxSubframe = 0;
        idxFrame = idxFrame + 1;
    end
end
carrier.idxSlot = idxSlot;
carrier.idxSubframe = idxSubframe; 
carrier.idxFrame = idxFrame;
if ~forceSlotIdxFlag
    carrier.idxSlotInFrame = carrier.idxSlot + carrier.idxSubframe * carrier.N_slot_subframe_mu;
end
