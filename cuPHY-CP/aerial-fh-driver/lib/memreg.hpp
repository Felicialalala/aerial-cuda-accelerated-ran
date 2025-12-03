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

#ifndef AERIAL_FH_MEMREG_HPP__
#define AERIAL_FH_MEMREG_HPP__

#include "aerial-fh-driver/api.hpp"
#include "fronthaul.hpp"

namespace aerial_fh
{
class MemReg {
public:
    MemReg(Fronthaul* fhi, MemRegInfo const* info);
    ~MemReg();
    Fronthaul* get_fronthaul() const;
    static uint32_t get_lkey(Nic* nic, void* addr);
    bool is_nvidia_peermem_loaded() const noexcept;

protected:
    MemRegInfo info_;
    Fronthaul* fhi_;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_MEMREG_HPP__