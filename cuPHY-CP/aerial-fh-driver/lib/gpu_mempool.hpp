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

#ifndef AERIAL_FH_GPU_MEMPOOL__
#define AERIAL_FH_GPU_MEMPOOL__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"

namespace aerial_fh
{
class Gpu;
class Nic;

class GpuMempool {
public:
    GpuMempool(Gpu* gpu, Nic* nic,bool host_pinned);
    ~GpuMempool();
    rte_mempool* get_pool() const;

protected:
    MempoolUnique      mempool_{nullptr, &rte_mempool_free};
    Nic*               nic_;
    Gpu*               gpu_;
    rte_pktmbuf_extmem gpu_mem_;
    bool               host_pinned_;
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_GPU_MEMPOOL__