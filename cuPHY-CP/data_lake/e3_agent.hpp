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

#ifndef E3_AGENT_HPP
#define E3_AGENT_HPP

#include <string_view>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdint>

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <cuda_fp16.h>

#include "nvlog.hpp"

#define TAG_E3 (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 7) // "CTL.E3"

// Forward declarations
class DataLake;
struct fhInfo_t;
struct puschInfo_t;
struct hestInfo_t;
typedef int16_t fhDataType;
typedef float2 hestDataType;

using json = nlohmann::json;

// E3 Protocol definitions
namespace e3 {

enum class MessageType : uint16_t {
    E3_SETUP_REQUEST = 0x01,
    E3_SETUP_RESPONSE = 0x02,
    E3_SUBSCRIPTION_REQUEST = 0x03,
    E3_SUBSCRIPTION_RESPONSE = 0x04,
    E3_INDICATION = 0x05,
    E3_CONTROL = 0x06,
    PHY_DATA_NOTIFICATION = 0x10  // Custom notification type
};

enum class DataType : uint32_t {
    IQ_SAMPLES = 0x01,
    CHANNEL_ESTIMATES = 0x02,
    CSI_DATA = 0x04,
    BEAM_WEIGHTS = 0x08,
    CQI_REPORTS = 0x10,
    ALL_DATA = 0xFFFFFFFF
};

/**
 * Stream types as bit flags for efficient stream processing
 */
enum class StreamType : uint64_t {
    NONE                  = 0,
    IQ_SAMPLES            = 1ULL << 0,
    PDU_DATA              = 1ULL << 1,
    H_ESTIMATES           = 1ULL << 2,
    SFN                   = 1ULL << 3,
    SLOT                  = 1ULL << 4,
    CELL_ID               = 1ULL << 5,
    N_RX_ANT              = 1ULL << 6,
    N_RX_ANT_SRS          = 1ULL << 7,
    N_CELLS               = 1ULL << 8,
    N_BS_ANTS             = 1ULL << 9,
    N_LAYERS              = 1ULL << 10,
    N_SUBCARRIERS         = 1ULL << 11,
    N_DMRS_ESTIMATES      = 1ULL << 12,
    DMRS_SYMB_POS         = 1ULL << 13,
    TB_CRC_FAIL           = 1ULL << 14,
    CB_ERRORS             = 1ULL << 15,
    RSRP                  = 1ULL << 16,
    CQI                   = 1ULL << 17,
    CB_COUNT              = 1ULL << 18,
    RSSI                  = 1ULL << 19,
    QAM_MOD_ORDER         = 1ULL << 20,
    MCS_INDEX             = 1ULL << 21,
    MCS_TABLE_INDEX       = 1ULL << 22,
    RB_START              = 1ULL << 23,
    RB_SIZE               = 1ULL << 24,
    START_SYMBOL_INDEX    = 1ULL << 25,
    NR_OF_SYMBOLS         = 1ULL << 26
};

/**
 * Converts string stream name to StreamType enum
 * 
 * @param[in] stream_name The stream name as string
 * @return Corresponding StreamType enum value
 */
constexpr StreamType streamNameToType(const std::string_view stream_name) noexcept
{
    if (stream_name == "iq_samples") return StreamType::IQ_SAMPLES;
    if (stream_name == "pdu_data") return StreamType::PDU_DATA;
    if (stream_name == "h_estimates") return StreamType::H_ESTIMATES;
    if (stream_name == "sfn") return StreamType::SFN;
    if (stream_name == "slot") return StreamType::SLOT;
    if (stream_name == "cell_id") return StreamType::CELL_ID;
    if (stream_name == "n_rx_ant") return StreamType::N_RX_ANT;
    if (stream_name == "n_rx_ant_srs") return StreamType::N_RX_ANT_SRS;
    if (stream_name == "n_cells") return StreamType::N_CELLS;
    if (stream_name == "n_bs_ants") return StreamType::N_BS_ANTS;
    if (stream_name == "n_layers") return StreamType::N_LAYERS;
    if (stream_name == "n_subcarriers") return StreamType::N_SUBCARRIERS;
    if (stream_name == "n_dmrs_estimates") return StreamType::N_DMRS_ESTIMATES;
    if (stream_name == "dmrs_symb_pos") return StreamType::DMRS_SYMB_POS;
    if (stream_name == "tb_crc_fail") return StreamType::TB_CRC_FAIL;
    if (stream_name == "cb_errors") return StreamType::CB_ERRORS;
    if (stream_name == "rsrp") return StreamType::RSRP;
    if (stream_name == "cqi") return StreamType::CQI;
    if (stream_name == "cb_count") return StreamType::CB_COUNT;
    if (stream_name == "rssi") return StreamType::RSSI;
    if (stream_name == "qam_mod_order") return StreamType::QAM_MOD_ORDER;
    if (stream_name == "mcs_index") return StreamType::MCS_INDEX;
    if (stream_name == "mcs_table_index") return StreamType::MCS_TABLE_INDEX;
    if (stream_name == "rb_start") return StreamType::RB_START;
    if (stream_name == "rb_size") return StreamType::RB_SIZE;
    if (stream_name == "start_symbol_index") return StreamType::START_SYMBOL_INDEX;
    if (stream_name == "nr_of_symbols") return StreamType::NR_OF_SYMBOLS;
    return StreamType::NONE;
}

/**
 * Bitwise OR operator for StreamType flags
 */
constexpr StreamType operator|(const StreamType lhs, const StreamType rhs) noexcept
{
    return static_cast<StreamType>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}

/**
 * Bitwise OR assignment operator for StreamType flags
 */
constexpr StreamType& operator|=(StreamType& lhs, const StreamType rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

/**
 * Bitwise AND operator for StreamType flags
 */
constexpr StreamType operator&(const StreamType lhs, const StreamType rhs) noexcept
{
    return static_cast<StreamType>(static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs));
}

/**
 * Converts a vector of stream names to a StreamType bitfield
 * 
 * @param[in] stream_names Vector of stream name strings
 * @return Bitfield representing all requested streams
 */
inline StreamType streamVectorToBitfield(const std::vector<std::string>& stream_names) noexcept
{
    StreamType bitfield = StreamType::NONE;
    for (const auto& name : stream_names) {
        bitfield |= streamNameToType(name);
    }
    return bitfield;
}

} // namespace e3

// Shared memory header structure
struct SharedMemoryHeader {
    uint32_t version;
    uint32_t fh_buffer_size;
    uint32_t pusch_buffer_size;
    uint32_t hest_buffer_size;
    uint32_t num_fh_samples;
    uint32_t num_fh_rows;
    uint32_t num_pusch_rows;
    uint32_t num_hest_rows;
    uint32_t max_hest_samples_per_row;
    uint32_t reserved[7];
};

class E3Agent {
public:
    E3Agent(
        DataLake* dataLake,
        const uint16_t e3PubPort,
        const uint16_t e3RepPort,
        const uint16_t e3SubPort,
        const int numRowsToInsertFh,
        const int numRowsToInsertPusch,
        const int numRowsToInsertHest,
        const uint32_t numFhSamples,
        const uint32_t maxHestSamplesPerRow
    );
    ~E3Agent();

    bool init();
    void shutdown();
    
    // Shared memory management
    bool createSharedMemoryBuffers(
        fhInfo_t** pFh,
        fhInfo_t** pInsertFh,
        puschInfo_t** p,
        puschInfo_t** pInsertPusch,
        hestInfo_t** pHest,
        hestInfo_t** pInsertHest
    );

    void notifyDataReady();

private:
    DataLake* dataLake;

    // E3 Agent configuration
    static constexpr std::string_view E3_AGENT_ID = "e3-agent-gh200";
    static constexpr std::string_view E3_AGENT_VERSION = "1.0.0";
    static constexpr std::string_view E3_SHARED_MEMORY_KEY = "/e3_ran_buffers";
    uint16_t e3PubPort;
    uint16_t e3RepPort;
    uint16_t e3SubPort;
    int numRowsToInsertFh;
    int numRowsToInsertPusch;
    int numRowsToInsertHest;
    uint32_t numFhSamples;
    uint32_t maxHestSamplesPerRow;

    // ZMQ components
    zmq::context_t zmq_context;
    zmq::socket_t e3_pub_socket;  // Agent → Manager (indications)
    zmq::socket_t e3_rep_socket;  // Manager → Agent (REQ-REP)
    zmq::socket_t e3_sub_socket;  // Manager → Agent (PUB-SUB commands)

    // Thread management
    std::thread e3_data_thread;
    std::thread e3_reaper_thread;
    std::thread e3_sub_thread;
    std::atomic<bool> e3_running{false};
    std::atomic<bool> e3_reaper_running{false};
    std::atomic<bool> e3_sub_running{false};

    // Active subscriptions
    struct E3Subscription {
        std::string subscription_id;
        uint32_t dapp_id;
        std::vector<std::string> granted_streams;
        e3::StreamType stream_bitfield;  //!< Bitfield representation of granted_streams
        uint32_t interval_ms;
        std::chrono::steady_clock::time_point last_update;
        bool active;
    };
    std::unordered_map<std::string, E3Subscription> e3_subscriptions;
    std::mutex e3_subscriptions_mutex;

    // Connected dApp managers
    struct DAppConnectionInfo {
        std::chrono::steady_clock::time_point last_activity_time;
    };
    std::map<uint32_t, DAppConnectionInfo> e3_connected_dapps;
    std::mutex e3_dapps_mutex;

    // Shared memory buffers for data exchange
    int shm_data_fd{-1};
    void* shm_data_ptr{nullptr};
    size_t shm_data_size{0};

    // Thread functions
    void dataServerThread();
    void reaperThread();
    void reapTimedOutDapps();
    void managerSubscriptionThread();

    // Request handlers
    void handleSetupRequest(const json& request, std::string& response);
    void handleSubscriptionRequest(const json& request, std::string& response);
    void handleUnsubscriptionRequest(const json& request, std::string& response);
    void handleControlMessage(const json& request, std::string& response);
    void handleManagerMessage(const json& message);

    // E3AP Message helpers
    std::string generateResponseMessageId(const std::string& prefix);
    std::string generateTimestamp();
    std::string generateSubscriptionId(const uint32_t dapp_id);
    
    // Stream creation helpers
    json createIndicationPayloadDelivery(const std::string& stream_id) const;
    json createIndicationPayloadStream(
        const std::string& stream_id,
        const std::string& data_type,
        const std::string& description
    ) const;
    json createSharedMemoryStream(
        const std::string& stream_id,
        const std::string& data_type,
        const std::string& description,
        const size_t memory_size_bytes,
        const uint32_t max_elements,
        const json& additional_shm_info = json::object(),
        const json& data_schema = json::object()
    ) const;
};

#endif // E3_AGENT_HPP

