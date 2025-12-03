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

#ifndef AERIAL_FH_DRIVER_PACKET_STATS__
#define AERIAL_FH_DRIVER_PACKET_STATS__

#include <cstdint>
#include <array>
#include <string>
#include <atomic>
#include <mutex>
#include <cstring>
#include "fh_mutex.hpp"
#include "oran.hpp"
#include "api.hpp"

//!< Maximum of two values
#define MAX(a, b) ((a) > (b) ? (a) : (b))
//!< Statistics array size (max of cell limits)
#define STAT_ARRAY_CELL_SIZE MAX(MAX_CELLS_PER_SLOT, API_MAX_NUM_CELLS)

//!< Total slots per 3GPP frame (10 subframes × slots per subframe)
#define SLOT_3GPP (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)
//!< Maximum number of launch pattern cycles
#define MAX_LAUNCH_PATTERN_CYCLES 4
//!< Total launch pattern slots (3GPP slots × cycles)
#define MAX_LAUNCH_PATTERN_SLOTS (SLOT_3GPP * MAX_LAUNCH_PATTERN_CYCLES)
//!< Maximum log message length for printing
#define MAX_PRINT_LOG_LENGTH 1024

#ifndef TAG_LATE_PACKETS
    #define TAG_LATE_PACKETS (NVLOG_TAG_BASE_FH_DRIVER + 21)  //!< Log tag for late packet messages
#endif

#ifndef TAG_SYMBOL_TIMINGS
    #define TAG_SYMBOL_TIMINGS (NVLOG_TAG_BASE_FH_DRIVER + 22)  //!< Log tag for symbol timing messages
#endif

#define TAG_TX_TIMINGS (NVLOG_TAG_BASE_FH_DRIVER + 23)  //!< Log tag for TX timing messages

#ifndef PACKET_SUMMARY_TAG
    #define PACKET_SUMMARY_TAG (NVLOG_TAG_BASE_FH_DRIVER + 24)  //!< Log tag for packet summary messages
#endif


/**
 * Packet arrival timing classification
 */
enum PacketCounterTiming
{
    EARLY,              //!< Packet arrived too early
    ONTIME,             //!< Packet arrived on time
    LATE,               //!< Packet arrived too late
    CounterTimingMax    //!< Maximum timing type (for array sizing)
};

/**
 * Downlink packet type classification
 */
enum DLPacketCounterType
{
    DLC,                 //!< Downlink C-plane
    DLU,                 //!< Downlink U-plane
    ULC,                 //!< Uplink C-plane
    DLCounterTypeMax     //!< Maximum DL counter type (for array sizing)
};

/**
 * Uplink U-plane packet type classification
 */
enum ULUPacketCounterType
{
    ULU_PRACH = 3,             //!< PRACH (Physical Random Access Channel) - starts after DLCounterTypeMax
    ULU_PUCCH = 4,             //!< PUCCH (Physical Uplink Control Channel)
    ULU_PUSCH = 5,             //!< PUSCH (Physical Uplink Shared Channel)
    ULU_SRS = 6,               //!< SRS (Sounding Reference Signal)
    ULUPacketCounterTypeMax = 7  //!< Maximum ULU counter type (for array sizing)
};

/**
 * Convert packet type enum to human-readable string
 *
 * @param[in] type Packet type from DLPacketCounterType
 * @return String representation of packet type
 */
inline const char* packet_type_to_char_string(int type)
{
    if(type == DLC)
        return "DL C";
    if(type == DLU)
        return "DL U";
    if(type == ULC)
        return "UL C";
    return "Unknown";
}

/**
 * Convert packet timing enum to human-readable string
 *
 * @param[in] timing Packet timing from PacketCounterTiming
 * @return String representation of timing classification
 */
inline const char* packet_timing_to_char_string(int timing)
{
    if(timing == EARLY)
        return "EARL";
    if(timing == ONTIME)
        return "ONTI";
    if(timing == LATE)
        return "LATE";
    return "Unknown";
}

/**
 * Uplink packet type classification
 */
enum ULPacketCounterType
{
    ULU,                 //!< Uplink U-plane
    ULCounterTypeMax     //!< Maximum UL counter type (for array sizing)
};

/**
 * Packet arrival statistics tracker
 *
 * Tracks and reports packet arrival timing statistics per cell, slot, and timing classification.
 * Used for monitoring and debugging fronthaul packet timing performance.
 */
class Packet_Statistics {
public:
    /**
     * Counter type for statistics querying
     */
    enum Counter_Type
    {
        TOTAL = 0,    //!< Total accumulated counters
        RUNNING = 1,  //!< Running/current counters
        BOTH = 2      //!< Both total and running counters
    };

    /**
     * Constructor - initializes and resets all statistics
     */
    Packet_Statistics() { reset(); }

    /**
     * Increment packet counters for a specific cell, timing type, and slot
     *
     * @param[in] cell Cell index
     * @param[in] type Timing type (EARLY/ONTIME/LATE)
     * @param[in] slot Slot index in launch pattern
     * @param[in] inc Increment value (default 1)
     */
    void increment_counters(int cell, int type, int slot, int inc = 1);

    /**
     * Get packet count for specific cell, timing type, and slot
     *
     * @param[in] cell Cell index
     * @param[in] type Timing type
     * @param[in] slot Slot index
     * @return Packet count
     */
    uint64_t get_cell_timing_slot_count(int cell, int type, int slot);

    /**
     * Get total packet count for specific cell and timing type across all slots
     *
     * @param[in] cell Cell index
     * @param[in] type Timing type
     * @return Total packet count
     */
    uint64_t get_cell_timing_count(int cell, int type);

    /**
     * Get total packet count for a cell across all timing types and slots
     *
     * @param[in] cell Cell index
     * @return Total packet count
     */
    uint64_t get_cell_total_count(int cell);

    /**
     * Get total packet count for a cell and slot across all timing types
     *
     * @param[in] cell Cell index
     * @param[in] slot Slot index
     * @return Total packet count for slot
     */
    uint64_t get_cell_slot_total_count(int cell, int slot);

    /**
     * Get percentage of packets with specific timing for a cell
     *
     * @param[in] cell Cell index
     * @param[in] timing Timing type
     * @return Percentage (0.0-100.0)
     */
    float get_cell_timing_percentage(int cell, int timing);

    /**
     * Get percentage of packets with specific timing for a cell and slot
     *
     * @param[in] cell Cell index
     * @param[in] timing Timing type
     * @param[in] slot Slot index
     * @return Percentage (0.0-100.0)
     */
    float get_cell_timing_slot_percentage(int cell, int timing, int slot);

    /**
     * Clear counters for specific cell, type, and slot
     *
     * @param[in] cell Cell index
     * @param[in] type Timing type
     * @param[in] slot Slot index
     */
    void clear_counter(int cell, int type, int slot);

    /**
     * Flush (print) counters to log for all cells
     *
     * @param[in] num_cells Number of cells to flush
     * @param[in] packet_type Specific packet type to flush, -1 for all types
     */
    void flush_counters(int num_cells, int packet_type = -1);

    /**
     * Flush (write) counters to file for all cells
     *
     * @param[in] num_cells Number of cells to write
     * @param[in] filename Output filename
     */
    void flush_counters_file(int num_cells, std::string filename);

    /**
     * Mark a slot as active for statistics tracking
     *
     * @param[in] slot Slot index to activate
     */
    void set_active_slot(int slot);

    /**
     * Check if statistics tracking is active
     *
     * @return true if active, false otherwise
     */
    bool active() {return active_;};

    /**
     * Check if cell meets on-time percentage threshold for a slot
     *
     * @param[in] cell Cell index
     * @param[in] threshold Minimum percentage threshold (0.0-100.0)
     * @return true if threshold met, false otherwise
     */
    bool pass_slot_percentage(int cell, float threshold);

    /**
     * Reset all statistics counters to zero
     */
    void reset();

private:
    bool active_ = false;  //!< Statistics tracking active flag
    //!< Per-slot statistics: [cell][slot][timing] -> count
    using Statistics_Array_Slot = std::array<std::array<std::array<std::atomic<uint64_t>, CounterTimingMax>, MAX_LAUNCH_PATTERN_SLOTS>, STAT_ARRAY_CELL_SIZE>;
    //!< Total statistics: [cell][timing] -> count
    using Statistics_Array = std::array<std::array<std::atomic<uint64_t>, CounterTimingMax>, STAT_ARRAY_CELL_SIZE>;
    //!< Active slot flags: [slot] -> active
    using Slot_Array = std::array<bool, MAX_LAUNCH_PATTERN_SLOTS>;
    Statistics_Array_Slot stats;          //!< Per-slot packet statistics
    Statistics_Array total_stats;         //!< Total accumulated statistics
    Slot_Array active_slots = {false};    //!< Active slots tracking array
};

/**
 * Frame/Subframe/Slot ID structure
 */
struct fssId {
    uint8_t frameId;      //!< Frame ID (0-255)
    uint8_t subframeId;   //!< Subframe ID (0-9)
    uint8_t slotId;       //!< Slot ID (depends on numerology)
};

/**
 * Packet timing information per slot
 *
 * Detailed packet arrival timing statistics for a single slot,
 * including per-symbol breakdowns and time-of-arrival measurements.
 */
struct packet_timer_per_slot
{
    aerial_fh::FHMutex mtx;                //!< Mutex for thread-safe access
    uint64_t earliest_packet_per_slot = UINT64_MAX;  //!< Earliest packet arrival time for slot (ns)
    uint64_t latest_packet_per_slot = 0;   //!< Latest packet arrival time for slot (ns)
    int64_t min_toa = INT64_MAX;           //!< Minimum time-of-arrival relative to t0 (ns)
    int64_t max_toa = INT64_MIN;           //!< Maximum time-of-arrival relative to t0 (ns)
    int64_t t0;                            //!< Slot reference time (t0) in nanoseconds
    int earliest_packet_symbol_num;        //!< Symbol number with earliest packet
    int latest_packet_symbol_num;          //!< Symbol number with latest packet
    int packet_count;                      //!< Total packet count for slot
    int early;                             //!< Number of early packets
    int ontime;                            //!< Number of on-time packets
    int late;                              //!< Number of late packets
    struct fssId fss;                      //!< Frame/Subframe/Slot identifier
    bool first_packet = true;              //!< Flag indicating first packet received
    uint16_t late_packets_per_symbol[ORAN_ALL_SYMBOLS];       //!< Late packet count per symbol
    uint64_t earliest_packet_per_symbol[ORAN_ALL_SYMBOLS];    //!< Earliest arrival time per symbol (ns)
    uint64_t latest_packet_per_symbol[ORAN_ALL_SYMBOLS];      //!< Latest arrival time per symbol (ns)
    uint64_t packet_arrive_abs[ORAN_ALL_SYMBOLS][32];         //!< Absolute arrival times (up to 32 packets per symbol)
    int64_t packet_arrive_t0s[ORAN_ALL_SYMBOLS][32];          //!< Arrival times relative to t0 (up to 32 packets per symbol)
    int num_packets_per_symbol[ORAN_ALL_SYMBOLS];             //!< Packet count per symbol

    /**
     * Reset all timing statistics to initial state
     */
    void reset()
    {
        earliest_packet_per_slot = UINT64_MAX;
        latest_packet_per_slot = 0;
        min_toa = INT64_MAX;
        max_toa = INT64_MIN;
        t0 = 0;
        packet_count = 0;
        early = 0;
        ontime = 0;
        late = 0;
        memset(late_packets_per_symbol, 0, ORAN_ALL_SYMBOLS*sizeof(uint16_t));
        for(int i = 0; i < ORAN_ALL_SYMBOLS; ++i)
        {
            earliest_packet_per_symbol[i] = UINT64_MAX;
        }
        memset(latest_packet_per_symbol, 0, ORAN_ALL_SYMBOLS*sizeof(uint64_t));
        memset(num_packets_per_symbol, 0, ORAN_ALL_SYMBOLS*sizeof(int));
    }
};

//!< Array of packet timers for all slots in a cell
using packet_slot_timers_per_cell = std::array<struct packet_timer_per_slot, MAX_LAUNCH_PATTERN_SLOTS>;

/**
 * Global packet timing information
 *
 * Contains timing arrays organized by packet type (DL C/U, UL C), cell, and slot
 */
struct packet_slot_timers
{
    //!< Packet timers: [packet_type][cell][slot]
    std::array<std::array<packet_slot_timers_per_cell, STAT_ARRAY_CELL_SIZE>, DLCounterTypeMax> timers;
};

/**
 * Flush (print) packet timing information to log
 *
 * @param[in] dir Direction (0=DL, 1=UL)
 * @param[in] type Packet type (C-plane/U-plane)
 * @param[in] cell_index Cell index
 * @param[in,out] packet_timer Packet timer structure to flush
 */
void flush_packet_timers(uint8_t dir, uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer);

#endif