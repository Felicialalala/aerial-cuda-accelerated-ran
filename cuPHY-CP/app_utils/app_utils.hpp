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

#ifndef APP_UTILS_H
#define APP_UTILS_H

#include <string>
#include <cstdint>

namespace AppUtils
{
    uint64_t get_ns_clock_tai(void);
    void get_phc_clock(const char *ifname);
    void clock_sanity_check();
    void set_tai_offset(int tai_offset);
    std::string get_nic_name(const std::string& pcie_address);
}

#endif // APP_UTILS_H