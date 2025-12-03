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

#ifndef ORAN_SLOT_ITERATOR_HPP__
#define ORAN_SLOT_ITERATOR_HPP__

#include "aerial-fh-driver/oran.hpp"

#include <random>

namespace fh_gen
{

constexpr uint16_t kMaxSFN      = 1024;

struct OranSlotNumber
{
    uint8_t frame_id;
    uint8_t subframe_id;
    uint8_t slot_id;
    int SFN;
};

class OranSlotIterator {
public:
    OranSlotIterator(OranSlotNumber start_slot_number);
    OranSlotNumber get_next();

protected:
    OranSlotNumber slot_number_;
};

} // namespace fh_gen

#endif //ifndef ORAN_SLOT_ITERATOR_HPP__
