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

#ifndef FH_GENERATOR_DOCA_UTILS_HPP__
#define FH_GENERATOR_DOCA_UTILS_HPP__

#include "utils.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <doca_log.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_eth_rxq.h>
#include <doca_eth_txq.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#pragma GCC diagnostic pop
namespace fh_gen
{

struct order_sem_info
{
    uint32_t pkts;
};

struct doca_rx_items {
	struct doca_gpu *gpu_dev;		/* GPU device */
	struct doca_dev *ddev;			/* Network DOCA device */
	struct doca_ctx *eth_rxq_ctx;		/* DOCA Ethernet receive queue context */
	struct doca_eth_rxq *eth_rxq_cpu;	/* DOCA Ethernet receive queue CPU handler */
	struct doca_gpu_eth_rxq *eth_rxq_gpu;	/* DOCA Ethernet receive queue GPU handler */
	struct doca_mmap *pkt_buff_mmap;	/* DOCA mmap to receive packet with DOCA Ethernet queue */
	void *gpu_pkt_addr;			/* DOCA mmap GPU memory address */
	uint16_t dpdk_queue_idx;

	struct doca_gpu_semaphore *sem_cpu;	/* One semaphore per queue to report stats, CPU handler*/
	struct doca_gpu_semaphore_gpu *sem_gpu;	/* One semaphore per queue to report stats, GPU handler*/
	int nitems;
};

struct RXKernelParams{
    struct doca_gpu_eth_rxq *rxq_info_gpu[fh_gen::kMaxCells];
};

#ifdef __cplusplus
extern "C" {
#endif
doca_error_t kernel_receive_persistent(cudaStream_t stream, int num_cells,
    /* DOCA objects */
    struct doca_gpu_eth_rxq **rxq_info_gpu,
    uint32_t *exit_flag);

struct kernel_receive_slot_params
{
    uint16_t* expected_rx_prbs_d;
    int slot_count;
    int frame_id;
    int subframe_id;
    int slot_id;
    int pattern_slot_id;
    uint64_t slot_t0;
    uint64_t* ta4_min_ns_d;
    uint64_t* ta4_max_ns_d;
};

#ifdef __cplusplus
}
#endif
}

#endif //ifndef FH_GENERATOR_DOCA_UTILS_HPP__
