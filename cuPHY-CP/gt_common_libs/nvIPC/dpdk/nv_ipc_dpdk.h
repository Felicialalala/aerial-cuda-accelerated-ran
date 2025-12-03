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

#ifndef NV_IPC_DPDK_H
#define NV_IPC_DPDK_H

#include <stdint.h>
#include <stddef.h>

#include <rte_mbuf.h>
#include <rte_ethdev.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct nvipc_hdr_t nvipc_hdr_t;
struct nvipc_hdr_t
{
    struct rte_ether_hdr eth_hdr;
    struct rte_vlan_hdr  vlan_hdr;

    uint16_t seg_num; // Total segment number
    uint16_t nb_segs; // Left segment number
    uint16_t seg_id;

    // nv_ipc_msg_t
    int32_t  msg_id;    // IPC message ID
    int32_t  cell_id;   // Cell ID
    int32_t  msg_len;   // MSG part length
    int32_t  data_len;  // DATA part length
    int32_t  data_pool; // DATA memory pool ID
    uint8_t* data_buf;  // MSG buffer pointer

    struct rte_mbuf* self;
    struct rte_mbuf* next;
    // nvipc_hdr_t* next;
    uint8_t payload[0];
};

// When MTU=1536, RX packet length(1518) with head-room(128) is 1646
// #define NIC_PORT_MTU 8192 // 1536
// #define MBUF_DATA_ROOM_SIZE (NIC_PORT_MTU - 18 + 128)
// #define MBUF_DATA_LEN_MAX (NIC_PORT_MTU - 128)
// #define MBUF_NVIPC_PAYLOAD_SIZE (MBUF_DATA_LEN_MAX - sizeof(nvipc_hdr_t))

//#define MAX_DATA_BUF_SIZE 576000
//#define MAX_MBUF_CHAIN_LEN (MAX_DATA_BUF_SIZE / MBUF_NVIPC_PAYLOAD_SIZE + 2)

static inline nvipc_hdr_t* get_nvipc_hdr(struct rte_mbuf* mbuf)
{
    if(mbuf == NULL)
    {
        return NULL;
    }
    else
    {
        return rte_pktmbuf_mtod(mbuf, nvipc_hdr_t*);
    }
}

static inline void* get_nvipc_payload(struct rte_mbuf* mbuf)
{
    nvipc_hdr_t* nvipc_hdr = rte_pktmbuf_mtod(mbuf, nvipc_hdr_t*);
    return nvipc_hdr->payload;
}

static inline struct rte_mbuf* get_rte_mbuf(const uint8_t* _payload)
{
    nvipc_hdr_t* nvipc_hdr = container_of(_payload, nvipc_hdr_t, payload[0]);
    return nvipc_hdr->self;
}

static inline void set_mbuf_self(struct rte_mbuf* mbuf)
{
    nvipc_hdr_t* nvipc_hdr = rte_pktmbuf_mtod(mbuf, nvipc_hdr_t*);
    nvipc_hdr->self        = mbuf;
}
#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* NV_IPC_DPDK_H */
