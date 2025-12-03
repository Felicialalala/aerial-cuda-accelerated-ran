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

#ifndef AERIAL_FH_QUEUE_HPP__
#define AERIAL_FH_QUEUE_HPP__

#include "aerial-fh-driver/api.hpp"
#include "dpdk.hpp"
#include "utils.hpp"
#include "fronthaul.hpp"
#include "doca_obj.hpp"
#include "aerial-fh-driver/fh_mutex.hpp"

namespace aerial_fh
{
class Nic;

/**
 * Base class for NIC packet queues
 *
 * Abstract base providing common functionality for transmit (TX)
 * and receive (RX) queues. Supports both DPDK and DOCA GPUNetIO
 * accelerated packet processing.
 */
class Queue {
public:
    /**
     * Constructor
     * @param[in] nic Associated NIC instance
     * @param[in] id Queue ID
     * @param[in] size Queue size (number of descriptors)
     */
    Queue(Nic* nic, uint16_t id, uint16_t size);

    /**
     * Virtual destructor - pure virtual (abstract class)
     */
    virtual ~Queue() = 0;

    /**
     * Get the queue ID
     * @return Queue identifier
     */
    uint16_t get_id() const;

    /**
     * Get DOCA RX items handle for GPU-accelerated RX
     * @return Pointer to DOCA RX items structure
     */
    doca_rx_items_t* get_doca_rx_items();

    /**
     * Get DOCA TX items handle for GPU-accelerated TX
     * @return Pointer to DOCA TX items structure
     */
    doca_tx_items_t* get_doca_tx_items();

protected:
    Nic*       nic_{};                  //!< Associated NIC instance
    aerial_fh::FHMutex mtx_{};          //!< Mutex for thread-safe access
    uint16_t   id_;                     //!< Queue identifier
    uint16_t   size_;                   //!< Queue size (descriptor count)
    Ns         last_tx_ts_on_txq_{0};   //!< Last TX timestamp on this queue (nanoseconds)

    doca_rx_items_t doca_rx_h;  //!< DOCA RX items handle for GPU acceleration
    doca_tx_items_t doca_tx_h;  //!< DOCA TX items handle for GPU acceleration
};

/**
 * Transmit Queue (TX)
 *
 * Handles packet transmission with optional accurate scheduling and timing.
 * Supports both CPU-only and GPU-accelerated (DOCA GPUNetIO) transmission modes.
 */
class Txq : public Queue {
public:
    /**
     * Constructor
     * @param[in] nic Associated NIC instance
     * @param[in] id Queue ID
     * @param[in] gpu_comm Enable GPU-initiated communication mode (default false)
     */
    Txq(Nic* nic, uint16_t id, bool gpu_comm = false);

    /**
     * Destructor
     */
    ~Txq();

    /**
     * Send packets without locking (caller must ensure thread safety)
     * @param[in] mbufs Array of mbuf pointers to send
     * @param[in] mbuf_count Number of mbufs to send
     * @param[out] timing Optional timing information structure
     * @return Number of packets successfully sent
     */
    size_t send(rte_mbuf** mbufs, size_t mbuf_count, TxqSendTiming* timing = nullptr);

    /**
     * Send packets with mutex locking (thread-safe)
     * @param[in] mbufs Array of mbuf pointers to send
     * @param[in] mbuf_count Number of mbufs to send
     * @param[out] timing Optional timing information structure
     * @return Number of packets successfully sent
     */
    size_t send_lock(rte_mbuf** mbufs, size_t mbuf_count, TxqSendTiming* timing = nullptr);

    /**
     * Send packets with mutex locking and transmit window scheduling
     * @param[in] mbufs Array of mbuf pointers to send
     * @param[in] mbuf_count Number of mbufs to send
     * @param[in] tx_window_start TX window start time in nanoseconds
     * @param[out] timing Optional timing information structure
     * @return Number of packets successfully sent
     */
    size_t send_lock(rte_mbuf** mbufs, size_t mbuf_count, Ns tx_window_start, TxqSendTiming* timing = nullptr);

    /**
     * Poll for TX completion events
     */
    void   poll_complete();

    /**
     * Warm up the TX queue (pre-allocate resources)
     */
    void   warm_up();

    /**
     * Check if queue is configured for GPU-initiated communication
     * @return true if GPU mode, false if CPU mode
     */
    bool   is_gpu() const;

    uint32_t * wqe_pi;  //!< Work Queue Element Producer Index (for GPU-init comm)
    uint32_t * cqe_ci;  //!< Completion Queue Element Consumer Index (for GPU-init comm)

protected:
    bool gpu_m_;  //!< GPU-initiated communication mode flag
};

/**
 * Receive Queue (RX)
 *
 * Handles packet reception with optional timeout support.
 * Supports both CPU and GPU memory for received packets.
 */
class Rxq : public Queue {
public:
    /**
     * Constructor
     * @param[in] nic Associated NIC instance
     * @param[in] id Queue ID
     */
    Rxq(Nic* nic, uint16_t id);

    /**
     * Receive packets (non-blocking)
     * @param[out] info Array to store received message information
     * @param[in,out] num_msgs Input: max messages to receive, Output: actual messages received
     * @return Number of packets successfully received
     */
    size_t receive(MsgReceiveInfo* info, size_t* num_msgs);

    /**
     * Receive packets with timeout
     * @param[out] info Array to store received message information
     * @param[in,out] num_msgs Input: max messages to receive, Output: actual messages received
     * @param[in] timeout Timeout in nanoseconds
     * @return Number of packets successfully received
     */
    size_t receive_until(MsgReceiveInfo* info, size_t* num_msgs, Ns timeout);

protected:
    MempoolUnique    cpu_mbuf_pool_queue_{nullptr, &rte_mempool_free};  //!< CPU mbuf pool for RX

    /**
     * Fill RX message info structures from received mbufs
     * @param[in] mbufs Array of received mbuf pointers
     * @param[out] info Array to fill with message information
     * @param[in] num_msgs Number of messages to process
     * @return Number of messages processed
     */
    size_t fill_rx_msg_info(rte_mbuf** mbufs, MsgReceiveInfo* info, size_t num_msgs);

    doca_rx_items_t doca_rxq_items;  //!< DOCA RX items for this queue
};

/**
 * Receive Queue for PCAP capture
 *
 * Specialized RX queue that receives packets and logs them to PCAP files
 * for debugging and analysis.
 */
class RxqPcap : public Queue {
public:
    /**
     * Constructor
     * @param[in] nic Associated NIC instance
     * @param[in] id Queue ID
     */
    RxqPcap(Nic* nic, uint16_t id);

    /**
     * Receive and log packets to PCAP
     */
    void receive();
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_QUEUE_HPP__
