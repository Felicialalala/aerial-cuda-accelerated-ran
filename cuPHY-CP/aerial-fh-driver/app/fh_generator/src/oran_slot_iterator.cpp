/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "oran_slot_iterator.hpp"

#include "utils.hpp"

#undef TAG
#define TAG "FHGEN.ORAN_SLOT_ITER"

namespace fh_gen
{
OranSlotIterator::OranSlotIterator(OranSlotNumber start_slot_number) :
    slot_number_{start_slot_number}
{
    NVLOGI_FMT(TAG, "Starting frameId = {}, subframeId = {}, slotId = {}" , +slot_number_.frame_id , +slot_number_.subframe_id , +slot_number_.slot_id);
    slot_number_.SFN = slot_number_.frame_id;
}

OranSlotNumber OranSlotIterator::get_next()
{
    auto current_slot_number = slot_number_;

    slot_number_.slot_id = (slot_number_.slot_id + 1) % ORAN_MAX_SLOT_ID;

    if(slot_number_.slot_id == 0)
    {
        slot_number_.subframe_id = (slot_number_.subframe_id + 1) % ORAN_MAX_SUBFRAME_ID;
    }

    if((slot_number_.slot_id == 0) && (slot_number_.subframe_id == 0))
    {
        slot_number_.frame_id = (slot_number_.frame_id + 1) % ORAN_MAX_FRAME_ID;
        slot_number_.SFN = (slot_number_.SFN + 1) % kMaxSFN;
    }



    return current_slot_number;
}

} // namespace fh_gen
