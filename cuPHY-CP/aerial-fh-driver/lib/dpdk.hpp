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

#ifndef AERIAL_FH_DPDK_HPP__
#define AERIAL_FH_DPDK_HPP__

#pragma nv_diag_suppress 1217 // warning #1217-D: unrecognized format function type "gnu_printf" ignored
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_flow.h>
#include <rte_gpudev.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_pdump.h>
#include <rte_ring.h>
#include <rte_pmd_mlx5.h>
#pragma nv_diag_default 1217

#include "nvlog.hpp"

#include <memory>
#include <vector>

#define ASSERT_MBUF_UNCHAINED(mbuf)                                                                   \
    if(unlikely((mbuf->nb_segs > 1) || (mbuf->next != nullptr) || (mbuf->data_len != mbuf->pkt_len))) \
    {                                                                                                 \
        THROW_FH(EPERM, StringBuilder() << "Chained mbufs are not supported");                        \
    }

using MempoolUnique = std::unique_ptr<struct rte_mempool, decltype(&rte_mempool_free)>;

class RxFlowRule {
public:
    RxFlowRule(uint16_t port_id, rte_flow* flow) :
        port_id_{port_id}, flow_{flow} {}

    ~RxFlowRule()
    {
        rte_flow_error err;
        auto           ret = rte_flow_destroy(port_id_, flow_, &err);
        if(ret)
        {
            NVLOGE_FMT("FH.FLOW", AERIAL_DPDK_API_EVENT, "Failed to destroy flow rule: {}\n", err.message);
        }
    }

protected:
    uint16_t  port_id_;
    rte_flow* flow_;
};

using RxFlowRulesUnique = std::vector<std::unique_ptr<RxFlowRule>>;

inline size_t sum_up_tx_bytes(rte_mbuf** mbufs, size_t mbuf_count)
{
    size_t bytes = 0;
    for(size_t i = 0; i < mbuf_count; i++)
    {
        bytes += mbufs[i]->pkt_len;
    }

    return bytes;
}

inline void attach_extbuf(rte_mbuf* m, void* buf_addr, uint16_t buf_len)
{
    auto shinfo = static_cast<rte_mbuf_ext_shared_info*>(rte_mbuf_to_priv(m));
    rte_mbuf_ext_refcnt_set(shinfo, 1);
    shinfo->free_cb = [](void* __rte_unused, void* __rte_unused) -> void {};

    m->buf_addr = buf_addr;
    m->buf_len  = buf_len;
    m->data_off = 0;
    m->ol_flags |= RTE_MBUF_F_EXTERNAL;
}

#endif //ifndef AERIAL_FH_DPDK_HPP__
