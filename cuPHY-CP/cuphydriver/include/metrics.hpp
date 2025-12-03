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

#ifndef METRICS_CLASS_H
#define METRICS_CLASS_H

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include "locks.hpp"
#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "nvlog.hpp"

#include <chrono>
#include <thread>

#ifdef AERIAL_METRICS
#include "aerial_metrics.hpp"
using namespace aerial_metrics;
#endif

/**
 * @brief Prometheus metrics exporter for cuPHYDriver
 *
 * Manages a dedicated worker thread that exports cuPHYDriver metrics to Prometheus.
 * When AERIAL_METRICS is enabled, this class provides HTTP endpoint for metrics scraping.
 */
class Metrics {

public:
    /**
     * @brief Construct metrics exporter
     *
     * @param _pdh      - Physical driver handle
     * @param _cpu_core - CPU core to pin the metrics worker thread to
     */
    Metrics(phydriver_handle _pdh, uint8_t _cpu_core);
    
    /**
     * @brief Destroy metrics exporter
     */
    ~Metrics();

    /**
     * @brief Get physical driver handle
     *
     * @return Physical driver handle
     */
    phydriver_handle       getPhyDriverHandler(void) const;
    
    /**
     * @brief Start metrics worker thread
     *
     * Launches the Prometheus metrics exporter thread on the configured CPU core.
     *
     * @return 0 on success, negative error code on failure
     */
    int                    start();
    
protected:
    phydriver_handle            pdh;                           ///< Physical driver handle
    uint8_t                     cpu_core;                      ///< CPU core for metrics worker thread affinity
    uint16_t                    id;                            ///< Metrics instance identifier
    phydriverwrk_handle         wProm;                         ///< Worker thread handle for Prometheus exporter
};

/**
 * @brief Per-cell metric types for Prometheus monitoring
 *
 * Defines all performance and statistics metrics collected per cell and exported
 * to Prometheus for monitoring and analysis. Includes counters for throughput,
 * errors, timing, and histograms for processing latency.
 */
enum class CellMetric {
    kDlSlotsTotal           = 0,                               ///< Total number of downlink slots processed (counter)
    kUlSlotsTotal           = 1,                               ///< Total number of uplink slots processed (counter)
    kPuschLostPrbsTotal     = 2,                               ///< Total PUSCH PRBs lost due to timing or errors (counter)
    kPuschRxTbBytesTotal    = 3,                               ///< Total PUSCH received transport block bytes (counter)
    kPuschRxTbTotal         = 4,                               ///< Total PUSCH received transport blocks (counter)
    kPuschRxTbCrcErrorTotal = 5,                               ///< Total PUSCH transport blocks with CRC errors (counter)
    kPrachLostPrbsTotal     = 6,                               ///< Total PRACH PRBs lost due to timing or errors (counter)
    kPrachRxPreamblesTotal  = 7,                               ///< Total PRACH preambles received (counter)
    kPdschTxBytesTotal      = 8,                               ///< Total PDSCH transmitted bytes (counter)
    kPdschTxTbTotal         = 9,                               ///< Total PDSCH transmitted transport blocks (counter)
    kPdschProcessingTime    = 10,                              ///< PDSCH processing time distribution in microseconds (histogram)
    kPucchProcessingTime    = 11,                              ///< PUCCH processing time distribution in microseconds (histogram)
    kPuschProcessingTime    = 12,                              ///< PUSCH processing time distribution in microseconds (histogram)
    kPrachProcessingTime    = 13,                              ///< PRACH processing time distribution in microseconds (histogram)
    kPuschNrOfUesPerSlot    = 14,                              ///< Number of UEs scheduled per PUSCH slot (histogram)
    kPdschNrOfUesPerSlot    = 15,                              ///< Number of UEs scheduled per PDSCH slot (histogram)
    kEarlyUplanePackets     = 16,                              ///< Total early U-plane packets received (before timing window, counter)
    kOnTimeUplanePackets    = 17,                              ///< Total on-time U-plane packets received (within timing window, counter)
    kLateUplanePackets      = 18,                              ///< Total late U-plane packets received (after timing window, counter)
    kSrsLostPrbsTotal       = 19,                              ///< Total SRS PRBs lost due to timing or errors (counter)
    kEarlyUplanePacketsSrs  = 20,                              ///< Total early SRS U-plane packets received (counter)
    kOnTimeUplanePacketsSrs = 21,                              ///< Total on-time SRS U-plane packets received (counter)
    kLateUplanePacketsSrs   = 22,                              ///< Total late SRS U-plane packets received (counter)
};

/**
 * @brief Per-cell metrics collector for Prometheus
 *
 * Manages Prometheus metric objects (counters and histograms) for a single cell.
 * Provides thread-safe updates for all cell-level performance and statistics metrics.
 * When AERIAL_METRICS is disabled, this becomes a no-op class.
 */
class CellMetrics {

public:
    /**
     * @brief Construct cell metrics collector
     *
     * Initializes all Prometheus metric objects for the specified cell.
     * Registers counters for throughput/errors and histograms for timing distributions.
     *
     * @param _cell_id - Physical cell identifier (used as metric label)
     */
    CellMetrics(uint16_t _cell_id);
    
    /**
     * @brief Destroy cell metrics collector
     *
     * Prometheus metric objects are managed by the global registry and are not deleted here.
     */
    ~CellMetrics();

    /**
     * @brief Update a cell metric with a new value
     *
     * Thread-safe method to increment counters or observe histogram values.
     * For counters: adds 'value' to the current count.
     * For histograms: records 'value' as a new observation.
     *
     * @param metric - Metric type to update (from CellMetric enum)
     * @param value  - Value to add (counter) or observe (histogram)
     */
    void update(CellMetric metric, uint64_t value);
    
protected:
    std::string cell_key_name;                                 ///< Cell identifier as string for metric labels

#ifdef AERIAL_METRICS
    // Throughput and volume counters
    prometheus::Counter*   dl_slots_total;                     ///< Total DL slots processed counter
    prometheus::Counter*   ul_slots_total;                     ///< Total UL slots processed counter
    prometheus::Counter*   pusch_lost_prbs_total;              ///< Total PUSCH PRBs lost counter
    prometheus::Counter*   pusch_rx_tb_bytes_total;            ///< Total PUSCH received bytes counter
    prometheus::Counter*   pusch_rx_tb_total;                  ///< Total PUSCH transport blocks counter
    prometheus::Counter*   pusch_rx_tb_crc_error_total;        ///< Total PUSCH CRC errors counter
    prometheus::Counter*   prach_lost_prbs_total;              ///< Total PRACH PRBs lost counter
    prometheus::Counter*   prach_rx_preambles_total;           ///< Total PRACH preambles counter
    prometheus::Counter*   pdsch_tx_bytes_total;               ///< Total PDSCH transmitted bytes counter
    prometheus::Counter*   pdsch_tx_tb_total;                  ///< Total PDSCH transport blocks counter
    
    // Processing time histograms (in microseconds)
    prometheus::Histogram* pdsch_processing_time;              ///< PDSCH processing latency histogram
    prometheus::Histogram* pucch_processing_time;              ///< PUCCH processing latency histogram
    prometheus::Histogram* pusch_processing_time;              ///< PUSCH processing latency histogram
    prometheus::Histogram* prach_processing_time;              ///< PRACH processing latency histogram
    prometheus::Histogram* pusch_nr_of_ues_per_slot;           ///< PUSCH UEs per slot distribution histogram
    prometheus::Histogram* pdsch_nr_of_ues_per_slot;           ///< PDSCH UEs per slot distribution histogram

    // U-plane packet timing counters (PUSCH/PRACH/PUCCH)
    prometheus::Counter*   early_uplane_rx_packets_total;      ///< Early U-plane packets counter (before window)
    prometheus::Counter*   on_time_uplane_rx_packets_total;    ///< On-time U-plane packets counter (within window)
    prometheus::Counter*   late_uplane_rx_packets_total;       ///< Late U-plane packets counter (after window)
    prometheus::Counter*   srs_lost_prbs_total;                ///< Total SRS PRBs lost counter

    // U-plane packet timing counters (SRS)
    prometheus::Counter*   early_uplane_rx_packets_total_srs;  ///< Early SRS U-plane packets counter
    prometheus::Counter*   on_time_uplane_rx_packets_total_srs;///< On-time SRS U-plane packets counter
    prometheus::Counter*   late_uplane_rx_packets_total_srs;   ///< Late SRS U-plane packets counter
#endif
};


#define MAX_LAUNCH_PATTERN_SLOTS    (80)                       ///< Maximum number of slots in launch pattern for packet statistics (supports up to 80-slot cycles)
#define METRICS_TAG 238                                        ///< Log tag 238 for "DRV.UL_PACKET_SUMMARY" (PUSCH/PRACH/PUCCH packet statistics)
#define METRICS_SRS_TAG 243                                    ///< Log tag 243 for "DRV.SRS_PACKET_SUMMARY" (SRS packet statistics)
#define MAX_PRINT_LOG_LENGTH 1024                              ///< Maximum log message length for packet statistics output

/**
 * @brief U-plane packet timing statistics collector
 *
 * Collects and reports statistics on U-plane packet arrival times relative to
 * ORAN timing windows. Tracks early, on-time, and late packet counts per cell
 * and slot for debugging timing issues and validating fronthaul synchronization.
 * Supports both regular uplink channels (PUSCH/PRACH/PUCCH) and SRS.
 */
class Packet_Statistics {
public:
    /**
     * @brief Packet arrival timing classification relative to ORAN timing window
     */
    enum timing_type
    {
        EARLY,                                                     ///< Packet arrived before the timing window (too early)
        ONTIME,                                                    ///< Packet arrived within the timing window (acceptable)
        LATE,                                                      ///< Packet arrived after the timing window (too late)
        MAX_TIMING_TYPES,                                          ///< Number of timing types (for array sizing)
    };

    /**
     * @brief Downlink packet types for statistics collection
     *
     * Currently unused in the implementation but defined for future DL packet statistics.
     */
    enum dl_packet_type
    {
        ULC,                                                       ///< Uplink C-plane packets
        DLC,                                                       ///< Downlink C-plane packets
        DLU,                                                       ///< Downlink U-plane packets
        MAX_DL_PACKET_TYPES,                                       ///< Number of DL packet types (for array sizing)
    };

    /**
     * @brief Construct packet statistics collector
     *
     * Initializes all counters to zero and sets inactive state.
     */
    Packet_Statistics() { reset(); }
    
    /**
     * @brief Increment packet counter for specific cell/timing/slot
     *
     * Thread-safe atomic increment of packet counter. Used by order kernels
     * and RX packet processing to track packet arrival timing statistics.
     *
     * @param cell - Cell index (0 to MAX_CELLS_PER_SLOT-1)
     * @param type - Timing classification (EARLY, ONTIME, or LATE)
     * @param slot - Slot number in launch pattern (0 to MAX_LAUNCH_PATTERN_SLOTS-1)
     * @param inc  - Increment value (default 1, can be used for batch updates)
     */
    void increment_counter(int cell, timing_type type, int slot, int inc = 1);
    
    /**
     * @brief Get packet count for specific cell/timing/slot
     *
     * @param cell - Cell index
     * @param type - Timing classification
     * @param slot - Slot number in launch pattern
     * @return Current packet count (atomic read)
     */
    uint64_t get_stat(int cell, timing_type type, int slot);
    
    /**
     * @brief Clear packet counter for specific cell/timing/slot
     *
     * Atomically resets counter to zero. Used during statistics flush operations.
     *
     * @param cell - Cell index
     * @param type - Timing classification
     * @param slot - Slot number in launch pattern
     */
    void clear_counter(int cell, timing_type type, int slot);
    
    /**
     * @brief Flush all active slot counters to log
     *
     * Prints packet statistics for all cells and active slots, then clears counters.
     * Outputs to log with appropriate tag (METRICS_TAG or METRICS_SRS_TAG).
     *
     * @param num_cells - Number of cells to report statistics for
     * @param isSRS     - True for SRS statistics (uses METRICS_SRS_TAG), false for PUSCH/PRACH/PUCCH (uses METRICS_TAG)
     */
    void flush_counters(int num_cells, bool isSRS = false);
    
    /**
     * @brief Flush all active slot counters to file
     *
     * Writes packet statistics for all cells and active slots to specified file,
     * then clears counters. Used for offline analysis and debugging.
     *
     * @param num_cells - Number of cells to write statistics for
     * @param filename  - Output file path
     */
    void flush_counters_file(int num_cells, std::string filename);
    
    /**
     * @brief Mark a slot as active for statistics collection
     *
     * Enables statistics collection for the specified slot. Only active slots
     * are reported during flush operations.
     *
     * @param slot - Slot number to activate (0 to MAX_LAUNCH_PATTERN_SLOTS-1)
     */
    void set_active_slot(int slot);
    
    /**
     * @brief Check if packet statistics collection is active
     *
     * @return true if any slot is marked as active, false otherwise
     */
    bool active() {return active_;};
    
    /**
     * @brief Reset all counters and deactivate all slots
     *
     * Clears all packet counters to zero and marks all slots as inactive.
     * Used during initialization and after major configuration changes.
     */
    void reset();
    
private:
    bool active_ = false;                                          ///< Flag indicating if any slot is active for statistics collection
    using Statistics_Array = std::array<
        std::array<
            std::array<std::atomic<uint64_t>, MAX_TIMING_TYPES>,
            MAX_LAUNCH_PATTERN_SLOTS>,
        MAX_CELLS_PER_SLOT>;                                       ///< 3D array: [cell][slot][timing_type] of atomic packet counters
    using Slot_Array = std::array<bool, MAX_LAUNCH_PATTERN_SLOTS>; ///< Array of slot active flags
    Statistics_Array stats;                                        ///< Packet statistics storage (thread-safe atomic counters)
    Slot_Array slots = {false};                                    ///< Slot activation flags (true = collect stats for this slot)
};

#endif
