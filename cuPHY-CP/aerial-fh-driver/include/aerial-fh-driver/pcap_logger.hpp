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

#ifndef PCAP_LOGGER_HPP__
#define PCAP_LOGGER_HPP__

#include <rte_mbuf.h>
#include <thread>
#include <atomic>
#include <string>
#include <memory>
#include <mutex>
#include "shm_logger.h"
#include "aerial-fh-driver/oran.hpp"
#include <cstring>

namespace aerial_fh {

//!< Maximum number of PCAP logger types supported
constexpr int kMaxPcapLoggerTypes = 3;
//!< Logger names for each packet type
constexpr std::array<std::string_view, kMaxPcapLoggerTypes>logger_names = { "dl_uplane", "dl_cplane", "ul_cplane" };

/**
 * PCAP logger packet type classification
 */
enum class PcapLoggerType {
    DL_UPLANE = 0,  //!< Downlink U-plane packets
    DL_CPLANE = 1,  //!< Downlink C-plane packets
    UL_CPLANE = 2,  //!< Uplink C-plane packets
    MAX_TYPES       //!< Maximum number of types (for array sizing)
};

/**
 * PCAP logger configuration
 */
struct PcapLoggerCfg final {
    bool enableDlUplane{};      //!< Enable downlink U-plane packet logging
    bool enableDlCplane{};      //!< Enable downlink C-plane packet logging
    bool enableUlCplane{};      //!< Enable uplink C-plane packet logging
    std::string output_path{};  //!< File-system path where PCAP files are stored
    int threadAffinity{};       //!< CPU core affinity for logger thread
    int threadPriority{};       //!< Thread priority for logger thread
};

static_assert(static_cast<int>(PcapLoggerType::MAX_TYPES) == kMaxPcapLoggerTypes, "Adjust kMaxPcapLoggerTypes to match # of defined logger types");

/**
 * Packet logging information container
 *
 * Holds mbuf pointer and packet type for queuing to logger thread
 */
struct PacketLogInfo final {
    rte_mbuf* mbuf{};           //!< DPDK packet mbuf to log
    PcapLoggerType  pktType{};  //!< Packet type classification (C-plane/U-plane)

    /**
     * Constructor
     * @param[in] m DPDK mbuf pointer
     * @param[in] type Packet type
     */
    PacketLogInfo(rte_mbuf* m, PcapLoggerType type)
        : mbuf (m), pktType(type) {}
};

/**
 * PCAP packet logger (Singleton)
 *
 * Logs DPDK mbufs to PCAP files for packet capture and debugging.
 * Supports multi-producer/single-consumer design:
 * - Multiple threads can enqueue packets via lock-free ring buffer
 * - Single consumer thread dequeues and writes to PCAP files
 *
 * The logger supports separate files for DL U-plane, DL C-plane, and UL C-plane packets.
 * The consumer thread polls the ring buffer periodically (every 5us) to write packet bursts.
 */
class PcapLogger final {
public:
    // Non-copyable and non-movable
    PcapLogger(const PcapLogger&) = delete;
    PcapLogger& operator=(const PcapLogger&) = delete;

    /**
     * Get singleton instance
     * @return Reference to the global PcapLogger instance
     */
    static PcapLogger& instance() {
        static PcapLogger instance;
        return instance;
    };

    /**
     * Initialize the logger with configuration
     *
     * @param[in] cfg Logger configuration (enables, output path, thread settings)
     */
    void init(const struct PcapLoggerCfg &cfg);

    /**
     * Check if downlink C-plane logging is enabled
     * @return true if enabled, false otherwise
     */
    [[nodiscard]] bool isDlCplaneLoggingEnabled() const {
        return logger_cfg_.enableDlCplane;
    }

    /**
     * Check if uplink C-plane logging is enabled
     * @return true if enabled, false otherwise
     */
    [[nodiscard]] bool isUlCplaneLoggingEnabled() const {
        return logger_cfg_.enableUlCplane;
    }

    /**
     * Check if downlink U-plane logging is enabled
     * @return true if enabled, false otherwise
     */
    [[nodiscard]] bool isDlUplaneLoggingEnabled() const {
        return logger_cfg_.enableDlUplane;
    }

    /**
     * Start the consumer thread for packet logging
     */
    void start();

    /**
     * Stop the consumer thread
     */
    void stop();

    /**
     * Enqueue an mbuf for logging (thread-safe)
     *
     * @param[in] mbuf DPDK mbuf to log
     * @param[in] type Packet type classification
     * @return true if successfully enqueued, false if ring buffer is full
     */
    bool enqueue(rte_mbuf* mbuf, PcapLoggerType type);

    /**
     * Get the ring buffer handle for producer access
     * @return Pointer to DPDK ring buffer
     */
    struct rte_ring* get_ring() { return ring_; }

private:
    /**
     * Private constructor for singleton pattern
     */
    PcapLogger();

    /**
     * Private destructor for singleton pattern
     */
    ~PcapLogger();

    friend std::default_delete<PcapLogger>;

    static constexpr size_t RING_SIZE = 4096*4;     //!< Ring buffer size (16K packets)
    static constexpr uint64_t SLEEP_NS = 5000;      //!< Sleep duration between dequeues (5 microseconds)

    /**
     * Consumer thread main loop
     */
    void run();

    /**
     * Process queued packets and write to PCAP files
     *
     * @param[out] numPkts Array to store number of packets processed per type
     * @return 0 on success, negative on error
     */
    int process_queue(uint32_t numPkts [kMaxPcapLoggerTypes]);

    PcapLoggerCfg logger_cfg_;                                  //!< Logger configuration
    rte_ring* ring_;                                            //!< DPDK ring buffer for packet queue
    std::array<shmlogger_t*,kMaxPcapLoggerTypes> logger_objs_; //!< PCAP file loggers (one per packet type)
    std::thread consumer_thread_;                               //!< Consumer thread handle
    std::atomic<bool> running_;                                 //!< Running flag for thread control
    bool initialized_;                                          //!< Initialization flag
};

} // namespace aerial_fh

#endif // #ifndef PCAP_LOGGER_HPP__
