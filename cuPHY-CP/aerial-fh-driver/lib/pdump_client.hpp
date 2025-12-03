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

#ifndef AERIAL_FH_PDUMP_CLIENT_HPP__
#define AERIAL_FH_PDUMP_CLIENT_HPP__

#include <atomic>
#include <thread>

#include "utils.hpp"

namespace aerial_fh
{
class Fronthaul;

class PdumpClient {
public:
    PdumpClient(Fronthaul* fhi, int32_t cpu_core);
    ~PdumpClient();

    Fronthaul* get_fhi() const;
    bool       signal_exit();

protected:
    Fronthaul*                   fhi_;
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool>            signal_exit_;
};

} // namespace aerial_fh

#endif // AERIAL_FH_PDUMP_CLIENT_HPP__