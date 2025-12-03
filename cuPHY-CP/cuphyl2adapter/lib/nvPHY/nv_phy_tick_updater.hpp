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

#if !defined(NV_PHY_TICK_UPDATER_HPP_INCLUDED_)
#define NV_PHY_TICK_UPDATER_HPP_INCLUDED_
#include "slot_command/slot_command.hpp"
#include "nv_phy_utils.hpp"

namespace nv
{
    class PHY_module;
    using namespace slot_command_api;
    struct TickUpdater
    {
        explicit TickUpdater(yaml::node& config):
            mu_highest_(config["mu_highest"].as<uint32_t>()),
            slot_info_(),
            prev_slot_info_(),
            slot_advance_(0)
        {
            if (config.has_key("slot_advance"))
            {
                slot_advance_ = config["slot_advance"].as<uint32_t>();
                slot_info_.slot_ += slot_advance_;
                prev_slot_info_.slot_ += slot_advance_ - 1;
            }
        }

        public:
        inline void operator()()
        {
            prev_slot_info_.sfn_ = slot_info_.sfn_;
            prev_slot_info_.slot_ = slot_info_.slot_;

            uint16_t slot_range = nv::mu_to_slot_in_sf(mu_highest_);
            if(slot_info_.slot_ == slot_range - 1)
            {
                slot_info_.sfn_ = (slot_info_.sfn_ + 1) % FAPI_SFN_MAX;
            }
            slot_info_.slot_ = (slot_info_.slot_ + 1) % slot_range;
            slot_info_.tick_++;
        }

        private:
        uint32_t mu_highest_;
        slot_command_api::slot_indication slot_info_;
        slot_command_api::slot_indication prev_slot_info_;
        uint32_t slot_advance_;
        TickUpdater(const TickUpdater&) = delete;
        TickUpdater& operator=(const TickUpdater&) = delete;
        public:
        TickUpdater(TickUpdater&& other):
            slot_info_(std::move(other.slot_info_)),
            prev_slot_info_(std::move(other.prev_slot_info_)),
            mu_highest_(std::move(other.mu_highest_)),
            slot_advance_(std::move(other.slot_advance_))
        {
        }
        friend class PHY_module;
    };
}

#endif /*NV_PHY_TICK_UPDATER_HPP_INCLUDED_*/
