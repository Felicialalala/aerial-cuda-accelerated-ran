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

#include "e3_agent.hpp"
#include "data_lake.hpp"
#include "fmt/format.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iomanip>
#include <complex>
#include <random>
#include <sstream>

// Constructor
E3Agent::E3Agent(
    DataLake* dl,
    const uint16_t pubPort,
    const uint16_t repPort,
    const uint16_t subPort,
    const int rowsFh,
    const int rowsPusch,
    const int rowsHest,
    const uint32_t fhSamples,
    const uint32_t maxHestSamples
) :
    dataLake(dl),
    e3PubPort(pubPort),
    e3RepPort(repPort),
    e3SubPort(subPort),
    numRowsToInsertFh(rowsFh),
    numRowsToInsertPusch(rowsPusch),
    numRowsToInsertHest(rowsHest),
    numFhSamples(fhSamples),
    maxHestSamplesPerRow(maxHestSamples),
    zmq_context(1),
    e3_pub_socket(zmq_context, ZMQ_PUB),
    e3_rep_socket(zmq_context, ZMQ_REP),
    e3_sub_socket(zmq_context, ZMQ_SUB)
{
}

// Destructor
E3Agent::~E3Agent()
{
    shutdown();

    if (shm_data_ptr != nullptr && shm_data_ptr != MAP_FAILED) {
        munmap(shm_data_ptr, shm_data_size);
    }
    if (shm_data_fd != -1) {
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
    }
}

// Initialize E3 agent - bind sockets and start threads
bool E3Agent::init()
{
    if (e3_running.load()) {
        return true;
    }

    NVLOGC_FMT(TAG_E3, "Initializing E3 Agent...");

    try {
        e3_pub_socket.bind("tcp://*:" + std::to_string(e3PubPort));
        e3_rep_socket.bind("tcp://*:" + std::to_string(e3RepPort));
        e3_sub_socket.connect("tcp://localhost:" + std::to_string(e3SubPort));

        e3_rep_socket.set(zmq::sockopt::tcp_keepalive, 1);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_idle, 5);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_intvl, 2);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_cnt, 3);

        // Subscribe to all topics - TODO: implement per-dApp topic filtering
        e3_sub_socket.set(zmq::sockopt::subscribe, "");
        e3_sub_socket.set(zmq::sockopt::linger, 1000);  // 1 second linger to allow graceful shutdown

        NVLOGC_FMT(TAG_E3, "E3 sockets initialized - PUB: {}, REP: {}, SUB: {}", e3PubPort, e3RepPort, e3SubPort);
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Failed to initialize E3 sockets: {}", e.what());
        return false;
    }

    e3_running = true;
    e3_data_thread = std::thread(&E3Agent::dataServerThread, this);

    e3_reaper_running = true;
    e3_reaper_thread = std::thread(&E3Agent::reaperThread, this);

    e3_sub_running = true;
    e3_sub_thread = std::thread(&E3Agent::managerSubscriptionThread, this);

    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        dataLake->e3_buffer_info = {};
    }

    NVLOGC_FMT(TAG_E3, "E3 Agent initialized successfully.");
    return true;
}

// Shutdown E3 agent - stop threads and close sockets
void E3Agent::shutdown()
{
    if (e3_running) {
        e3_running = false;
        if (e3_data_thread.joinable()) {
            e3_data_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 data server thread shutdown");
    }

    if (e3_sub_running) {
        e3_sub_running = false;
        if (e3_sub_thread.joinable()) {
            e3_sub_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 subscription thread shutdown");
    }

    if (e3_reaper_running) {
        e3_reaper_running = false;
        if (e3_reaper_thread.joinable()) {
            e3_reaper_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 reaper thread shutdown");
    }

    try {
        e3_pub_socket.close();
        e3_rep_socket.close();
        e3_sub_socket.close();
        NVLOGC_FMT(TAG_E3, "E3 sockets closed successfully");
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Error closing E3 sockets: {}", e.what());
    }
}

// Create shared memory buffers for data exchange
bool E3Agent::createSharedMemoryBuffers(
    fhInfo_t** pFh,
    fhInfo_t** pInsertFh,
    puschInfo_t** p,
    puschInfo_t** pInsertPusch,
    hestInfo_t** pHest,
    hestInfo_t** pInsertHest
)
{
    NVLOGC_FMT(TAG_E3, "Creating shared memory buffers for E3");

    const size_t fh_buffer_size = numFhSamples * numRowsToInsertFh * sizeof(fhDataType);
    const size_t pusch_buffer_size = 80000 * numRowsToInsertPusch;
    const size_t hest_buffer_size = maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType);

    const size_t total_size = sizeof(SharedMemoryHeader) +
                             (2 * fh_buffer_size) +
                             (2 * pusch_buffer_size) +
                             (2 * hest_buffer_size);

    shm_data_fd = shm_open(E3_SHARED_MEMORY_KEY.data(), O_CREAT | O_RDWR, 0666);
    if (shm_data_fd == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to create shared memory, errno: {}", errno);
        return false;
    }

    if (ftruncate(shm_data_fd, total_size) == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to set shared memory size, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_data_fd, 0);
    if (shm_data_ptr == MAP_FAILED) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to map shared memory, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_size = total_size;

    SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(shm_data_ptr);
    memset(header, 0, sizeof(SharedMemoryHeader));
    header->version = 1;
    header->fh_buffer_size = fh_buffer_size;
    header->pusch_buffer_size = pusch_buffer_size;
    header->hest_buffer_size = hest_buffer_size;
    header->num_fh_samples = numFhSamples;
    header->num_fh_rows = numRowsToInsertFh;
    header->num_pusch_rows = numRowsToInsertPusch;
    header->num_hest_rows = numRowsToInsertHest;
    header->max_hest_samples_per_row = maxHestSamplesPerRow;

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(header + 1);

    (*pFh)->pDataAlloc = reinterpret_cast<fhDataType*>(base_ptr);
    (*pInsertFh)->pDataAlloc = reinterpret_cast<fhDataType*>(base_ptr + fh_buffer_size);

    (*p)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + 2 * fh_buffer_size);
    (*pInsertPusch)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + 2 * fh_buffer_size + pusch_buffer_size);

    (*pHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + 2 * fh_buffer_size + 2 * pusch_buffer_size);
    (*pInsertHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + 2 * fh_buffer_size + 2 * pusch_buffer_size + hest_buffer_size);

    NVLOGC_FMT(TAG_E3, "Shared memory buffers created successfully");
    NVLOGC_FMT(TAG_E3, "  Total size: {} bytes", total_size);
    NVLOGC_FMT(TAG_E3, "  FH buffers: {} bytes each", fh_buffer_size);
    NVLOGC_FMT(TAG_E3, "  PUSCH buffers: {} bytes each", pusch_buffer_size);
    NVLOGC_FMT(TAG_E3, "  H estimates buffers: {} bytes each", hest_buffer_size);

    return true;
}


// Notify subscribers that data is ready
void E3Agent::notifyDataReady()
{
    if (!e3_running) {
        return;
    }

    NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: e3NotifyDataReady (Op #4) entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());

    E3BufferInfo buffer_info;
    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        buffer_info = dataLake->e3_buffer_info;
    }

    std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
    for (auto& [sub_id, sub] : e3_subscriptions) {
        if (!sub.active) {
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - sub.last_update).count() >= sub.interval_ms) {
            json notif_json;
            notif_json["type"] = "e3_indication";
            notif_json["dapp_id"] = sub.dapp_id;
            notif_json["subscription_id"] = sub.subscription_id;
            notif_json["timestamp_ns"] = buffer_info.timestamp_ns;

            json indication_payload;
            uint64_t remaining_streams = static_cast<uint64_t>(sub.stream_bitfield);
            while (remaining_streams != 0) {
                // Find the lowest set bit
                const uint64_t lowest_bit = remaining_streams & (~remaining_streams + 1);
                const e3::StreamType stream_flag = static_cast<e3::StreamType>(lowest_bit);

                switch (stream_flag) {
                    case e3::StreamType::IQ_SAMPLES: {
                        json iq_shm_data;
                        iq_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        iq_shm_data["fh_buffer_index"] = static_cast<int>(buffer_info.current_fh_buffer);
                        iq_shm_data["fh_write_index"] = buffer_info.fh_write_index;
                        indication_payload["iq_samples"] = iq_shm_data;
                        break;
                    }
                    case e3::StreamType::PDU_DATA: {
                        json pdu_shm_data;
                        pdu_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        pdu_shm_data["pusch_buffer_index"] = static_cast<int>(buffer_info.current_pusch_buffer);
                        pdu_shm_data["pusch_write_index"] = buffer_info.pusch_write_index;
                        indication_payload["pdu_data"] = pdu_shm_data;
                        break;
                    }
                    case e3::StreamType::H_ESTIMATES: {
                        json hest_shm_data;
                        hest_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        hest_shm_data["hest_buffer_index"] = static_cast<int>(buffer_info.current_hest_buffer);
                        hest_shm_data["hest_write_index"] = buffer_info.hest_write_index;
                        indication_payload["h_estimates"] = hest_shm_data;
                        break;
                    }
                    case e3::StreamType::SFN: {
                        indication_payload["sfn"] = buffer_info.sfn;
                        break;
                    }
                    case e3::StreamType::SLOT: {
                        indication_payload["slot"] = buffer_info.slot;
                        break;
                    }
                    case e3::StreamType::CELL_ID: {
                        indication_payload["cell_id"] = buffer_info.cell_id;
                        break;
                    }
                    case e3::StreamType::N_RX_ANT: {
                        indication_payload["n_rx_ant"] = buffer_info.n_rx_ant;
                        break;
                    }
                    case e3::StreamType::N_RX_ANT_SRS: {
                        indication_payload["n_rx_ant_srs"] = buffer_info.n_rx_ant_srs;
                        break;
                    }
                    case e3::StreamType::N_CELLS: {
                        indication_payload["n_cells"] = buffer_info.n_cells;
                        break;
                    }
                    case e3::StreamType::N_BS_ANTS: {
                        indication_payload["n_bs_ants"] = buffer_info.n_bs_ants;
                        break;
                    }
                    case e3::StreamType::N_LAYERS: {
                        indication_payload["n_layers"] = buffer_info.n_layers;
                        break;
                    }
                    case e3::StreamType::N_SUBCARRIERS: {
                        indication_payload["n_subcarriers"] = buffer_info.n_subcarriers;
                        break;
                    }
                    case e3::StreamType::N_DMRS_ESTIMATES: {
                        indication_payload["n_dmrs_estimates"] = buffer_info.n_dmrs_estimates;
                        break;
                    }
                    case e3::StreamType::DMRS_SYMB_POS: {
                        indication_payload["dmrs_symb_pos"] = buffer_info.dmrs_symb_pos;
                        break;
                    }
                    case e3::StreamType::TB_CRC_FAIL: {
                        indication_payload["tb_crc_fail"] = buffer_info.tb_crc_fail;
                        break;
                    }
                    case e3::StreamType::CB_ERRORS: {
                        indication_payload["cb_errors"] = buffer_info.cb_errors;
                        break;
                    }
                    case e3::StreamType::RSRP: {
                        indication_payload["rsrp"] = buffer_info.rsrp;
                        break;
                    }
                    case e3::StreamType::CQI: {
                        indication_payload["cqi"] = buffer_info.cqi;
                        break;
                    }
                    case e3::StreamType::CB_COUNT: {
                        indication_payload["cb_count"] = buffer_info.cb_count;
                        break;
                    }
                    case e3::StreamType::RSSI: {
                        indication_payload["rssi"] = buffer_info.rssi;
                        break;
                    }
                    case e3::StreamType::QAM_MOD_ORDER: {
                        indication_payload["qam_mod_order"] = buffer_info.qam_mod_order;
                        break;
                    }
                    case e3::StreamType::MCS_INDEX: {
                        indication_payload["mcs_index"] = buffer_info.mcs_index;
                        break;
                    }
                    case e3::StreamType::MCS_TABLE_INDEX: {
                        indication_payload["mcs_table_index"] = buffer_info.mcs_table_index;
                        break;
                    }
                    case e3::StreamType::RB_START: {
                        indication_payload["rb_start"] = buffer_info.rb_start;
                        break;
                    }
                    case e3::StreamType::RB_SIZE: {
                        indication_payload["rb_size"] = buffer_info.rb_size;
                        break;
                    }
                    case e3::StreamType::START_SYMBOL_INDEX: {
                        indication_payload["start_symbol_index"] = buffer_info.start_symbol_index;
                        break;
                    }
                    case e3::StreamType::NR_OF_SYMBOLS: {
                        indication_payload["nr_of_symbols"] = buffer_info.nr_of_symbols;
                        break;
                    }
                    case e3::StreamType::NONE:
                    default: {
                        // Ignore unrecognized or unset stream types
                        break;
                    }
                }
                remaining_streams &= ~lowest_bit;
            }

            if (!indication_payload.empty()) {
                notif_json["indication_payload"] = indication_payload;
            }

            try {
                NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: Before ZMQ send (Op #4) at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
                const std::string topic = std::to_string(sub.dapp_id) + ":" + sub.subscription_id + "|";
                const std::string message = topic + notif_json.dump();
                e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
                NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: After ZMQ send (Op #4) at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
                NVLOGD_FMT(TAG_E3, "Sent E3 indication to dApp {} for subscription {} with topic {}", sub.dapp_id, sub.subscription_id, topic);
            } catch (const zmq::error_t& e) {
                NVLOGD_FMT(TAG_E3, "No subscribers for E3 indication");
            }
            sub.last_update = now;
        }
    }
}

// Thread functions

// E3 data server thread - handles ZMQ request/reply
void E3Agent::dataServerThread()
{
    e3_rep_socket.set(zmq::sockopt::rcvtimeo, 1000);

    NVLOGC_FMT(TAG_E3, "E3 data server thread started");

    while (e3_running) {
        zmq::message_t request;
        if (e3_rep_socket.recv(request, zmq::recv_flags::none)) {
            std::string response;
            try {
                const json req_json = json::parse(std::string(static_cast<char*>(request.data()), request.size()));
                const std::string type = req_json.value("type", "");

                if (type != "e3_control_request") {
                    NVLOGC_FMT(TAG_E3, "Received E3 request: {}", req_json.dump());
                }

                if (type == "e3_setup_request") {
                    handleSetupRequest(req_json, response);
                } else if (type == "e3_subscription_request") {
                    handleSubscriptionRequest(req_json, response);
                } else if (type == "e3_unsubscription_request") {
                    handleUnsubscriptionRequest(req_json, response);
                } else if (type == "e3_control_request") {
                    handleControlMessage(req_json, response);
                } else {
                    json error_resp;
                    error_resp["status"] = "error";
                    error_resp["message"] = "unknown request type";
                    response = error_resp.dump();
                }
            } catch (const json::parse_error& e) {
                json error_resp;
                error_resp["status"] = "error";
                error_resp["message"] = "invalid JSON format";
                response = error_resp.dump();
                NVLOGC_FMT(TAG_E3, "Failed to parse request: {}", e.what());
            }
            e3_rep_socket.send(zmq::buffer(response));
        }
    }
    NVLOGC_FMT(TAG_E3, "E3 data server thread stopped");
}

// E3 reaper thread - cleanup disconnected dApps
void E3Agent::reaperThread()
{
    NVLOGC_FMT(TAG_E3, "E3 reaper thread started");

    while (e3_reaper_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        reapTimedOutDapps();
    }
    NVLOGC_FMT(TAG_E3, "E3 reaper thread stopped");
}

// Remove timed-out dApps
void E3Agent::reapTimedOutDapps()
{
    constexpr auto ACTIVITY_TIMEOUT_SECONDS = 1800;

    const auto now = std::chrono::steady_clock::now();
    std::vector<uint32_t> timed_out_dapps;

    {
        std::lock_guard<std::mutex> lock(e3_dapps_mutex);
        for (auto const& [dapp_id, conn_info] : e3_connected_dapps) {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - conn_info.last_activity_time).count() > ACTIVITY_TIMEOUT_SECONDS) {
                bool has_active_subscriptions = false;
                {
                    std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);
                    for (const auto& [sub_id, sub] : e3_subscriptions) {
                        if (sub.dapp_id == dapp_id && sub.active) {
                            has_active_subscriptions = true;
                            break;
                        }
                    }
                }

                if (!has_active_subscriptions) {
                    timed_out_dapps.push_back(dapp_id);
                }
            }
        }
    }

    if (!timed_out_dapps.empty()) {
        std::lock_guard<std::mutex> dapps_lock(e3_dapps_mutex);
        std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);
        for (const uint32_t dapp_id : timed_out_dapps) {
            NVLOGC_FMT(TAG_E3, "dApp {} timed out after {} seconds of inactivity. Removing.", dapp_id, ACTIVITY_TIMEOUT_SECONDS);
            e3_connected_dapps.erase(dapp_id);
            for (auto it = e3_subscriptions.begin(); it != e3_subscriptions.end(); ) {
                if (it->second.dapp_id == dapp_id) {
                    NVLOGC_FMT(TAG_E3, "Removing subscription {}", it->first);
                    it = e3_subscriptions.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

// Manager subscription thread - receives commands from E3 Manager
void E3Agent::managerSubscriptionThread()
{
    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread started");

    while (e3_sub_running) {
        try {
            zmq::message_t msg;
            const auto result = e3_sub_socket.recv(msg, zmq::recv_flags::dontwait);

            if (result) {
                const std::string received = msg.to_string();

                // Parse "topic|message" format
                const size_t delimiter = received.find('|');
                if (delimiter != std::string::npos) {
                    const std::string topic = received.substr(0, delimiter);
                    const std::string message = received.substr(delimiter + 1);

                    NVLOGC_FMT(TAG_E3, "Received Manager command - Topic: '{}', Message: '{}'", topic, message);

                    // Parse JSON message
                    try {
                        const json msg_json = json::parse(message);
                        handleManagerMessage(msg_json);
                    } catch (const json::exception& e) {
                        NVLOGC_FMT(TAG_E3, "Failed to parse Manager message JSON: {}", e.what());
                    }
                } else {
                    NVLOGC_FMT(TAG_E3, "Received malformed Manager message (no topic separator): '{}'", received);
                }
            }
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                // Context terminated during shutdown - exit gracefully
                break;
            } else if (e.num() != EAGAIN) {
                NVLOGC_FMT(TAG_E3, "Error receiving from Manager: {}", e.what());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread stopped");
}

// Handle messages asynchronously received from E3 Manager
void E3Agent::handleManagerMessage(const json& message)
{
    NVLOGC_FMT(TAG_E3, "Handling Manager message: {}", message.dump());
}

// E3AP Message helpers

std::string E3Agent::generateResponseMessageId(const std::string& prefix)
{
    static std::atomic<uint32_t> response_counter{1};
    const uint32_t seq = response_counter.fetch_add(1);

    return fmt::format("{}_{}_{}",prefix, generateTimestamp(), seq);
}

std::string E3Agent::generateTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds);
    const std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm tm_buf{};
    gmtime_r(&time, &tm_buf);

    return fmt::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z",
                       tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                       tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                       ms.count());
}

std::string E3Agent::generateSubscriptionId(const uint32_t dapp_id)
{
    static std::atomic<uint32_t> sub_counter{1};
    const uint32_t seq = sub_counter.fetch_add(1);

    return fmt::format("sub_{}_{:03d}", dapp_id, seq);
}

// Stream creation helpers

json E3Agent::createIndicationPayloadDelivery(const std::string& stream_id) const
{
    json delivery;
    delivery["transport_type"] = "indication_payload";
    delivery["keyword"] = stream_id;
    delivery["encoding"] = "json";
    return delivery;
}

json E3Agent::createIndicationPayloadStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description
) const
{
    json stream;
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";
    stream["delivery_method"] = createIndicationPayloadDelivery(stream_id);
    return stream;
}

json E3Agent::createSharedMemoryStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description,
    const size_t memory_size_bytes,
    const uint32_t max_elements,
    const json& additional_shm_info,
    const json& data_schema
) const
{
    json stream;
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";

    json delivery;
    delivery["transport_type"] = "shared_memory";

    json shm_info;
    shm_info["memory_key"] = E3_SHARED_MEMORY_KEY;
    shm_info["memory_size_bytes"] = memory_size_bytes;
    shm_info["access_pattern"] = "double_buffer";
    shm_info["max_elements"] = max_elements;

    if (!additional_shm_info.empty()) {
        shm_info.update(additional_shm_info);
    }

    delivery["shared_memory_info"] = shm_info;
    stream["delivery_method"] = delivery;

    if (!data_schema.empty()) {
        stream["data_schema"] = data_schema;
    }

    return stream;
}

// Request handlers

void E3Agent::handleSetupRequest(const json& request, std::string& response) {
	json response_json;
	json e3_setup_response;
	
	try {
		const json& e3_setup_req = request;
		
		// Extract transaction ID for response correlation
		uint32_t transaction_id = e3_setup_req.value("transaction_id", 0u);
		std::string e3sm_version = e3_setup_req.value("e3sm_version", "unknown");
		
		// Extract dApp information
		json dapp_info = e3_setup_req.value("dapp_info", json::object());
		std::string dapp_name = dapp_info.value("dapp_name", "unknown");
		std::string dapp_version = dapp_info.value("dapp_version", "unknown");
		std::string vendor = dapp_info.value("vendor", "unknown");
		std::string description = dapp_info.value("description", "");
		
		NVLOGC_FMT(TAG_E3, "E3 Setup Request from dApp '{}' v{} by {} (E3SM v{})",
				   dapp_name, dapp_version, vendor, e3sm_version);
		NVLOGC_FMT(TAG_E3, "dApp description: {}", description);
		
		// Generate dApp ID during setup phase
		uint32_t dapp_id;
		{
			std::lock_guard<std::mutex> lock(e3_dapps_mutex);
			
			// Always generate a new unique dApp ID during setup
			do {
				dapp_id = std::random_device{}();
			} while (e3_connected_dapps.find(dapp_id) != e3_connected_dapps.end());
			
			e3_connected_dapps[dapp_id] = {std::chrono::steady_clock::now()};
		}
		
		// Create E3AP Setup Response
		e3_setup_response["type"] = "e3_setup_response";
		e3_setup_response["message_id"] = generateResponseMessageId("resp");
		e3_setup_response["timestamp"] = generateTimestamp();
		e3_setup_response["transaction_id"] = transaction_id;
		e3_setup_response["status"] = "success";
		e3_setup_response["dapp_id"] = dapp_id;
		e3_setup_response["e3_agent_id"] = E3_AGENT_ID;
		e3_setup_response["e3sm_version"] = "1.0.0";
		
		// Agent information
		json agent_info;
		agent_info["agent_version"] = E3_AGENT_VERSION;
		agent_info["vendor"] = "NVIDIA";
		e3_setup_response["agent_info"] = agent_info;
		
		// Indication info
		json indication_info;
		indication_info["link_protocol"] = "ZMQ";
		indication_info["transport_protocol"] = "TCP";
		indication_info["socket_type"] = "PUB";
		indication_info["socket_port"] = e3PubPort;
		e3_setup_response["indication_info"] = indication_info;
		
		// Available data streams
		json available_data_streams = json::array();
		
		// IQ Samples stream (FH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"iq_samples",
			"array(int16)",
			"Raw IQ samples (Fronthaul data)",
			numFhSamples * numRowsToInsertFh * sizeof(fhDataType),
			numRowsToInsertFh
		));

		// PDU Data stream (PUSCH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"pdu_data",
			"array(uint8)",
			"PUSCH PDU data",
			80000 * numRowsToInsertPusch,
			numRowsToInsertPusch
		));
		
		// H Estimates stream
		json hest_shm_info;
		hest_shm_info["max_samples_per_row"] = maxHestSamplesPerRow;
		
		json hest_schema;
		hest_schema["dimensions"] = "Variable: (N_BS_ANTS, N_LAYERS, NF, NH) for first UE group";
		hest_schema["N_BS_ANTS"] = "Number of base station antennas (limited to 4)";
		hest_schema["N_LAYERS"] = "Number of spatial layers";
		hest_schema["NF"] = "Number of subcarriers (PRBs * 12)";
		hest_schema["NH"] = "Number of DMRS estimates";
		
		available_data_streams.push_back(createSharedMemoryStream(
			"h_estimates",
			"array(complex64)",
			"PUSCH H matrix estimates (first UE group only)",
			maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType),
			numRowsToInsertHest,
			hest_shm_info,
			hest_schema
		));
		
		// Timing and Cell information streams
		available_data_streams.push_back(createIndicationPayloadStream("sfn", "uint16", "Network frame timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("slot", "uint16", "Network slot timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("cell_id", "uint16", "Physical Cell ID"));
		
		// Antenna and cell configuration streams
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant", "uint16", "Number of receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant_srs", "uint16", "Number of SRS receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_cells", "uint16", "Number of cells"));
		
		// H Estimates metadata streams
		available_data_streams.push_back(createIndicationPayloadStream("n_bs_ants", "uint8", "Number of base station antennas in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_layers", "uint8", "Number of spatial layers in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_subcarriers", "uint16", "Number of subcarriers (PRBs * 12) in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_dmrs_estimates", "uint8", "Number of DMRS estimates in H matrix"));
		available_data_streams.push_back(createIndicationPayloadStream("dmrs_symb_pos", "uint16", "DMRS symbol positions bitmap"));
		
		// Quality and Error metrics streams
		available_data_streams.push_back(createIndicationPayloadStream("tb_crc_fail", "uint8", "Transport Block CRC aggregated failure indicator (0=success,1=failure)"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_errors", "uint32", "Code Block CRC error count per UE"));
		available_data_streams.push_back(createIndicationPayloadStream("rsrp", "float32", "Reference Signal Received Power per UE in dB"));
		
		// Modulation and Coding Scheme streams
		available_data_streams.push_back(createIndicationPayloadStream("qam_mod_order", "uint8", "QAM modulation order (2,4,6,8 if transform precoding disabled; 1,2,4,6,8 if enabled)"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_index", "uint8", "MCS index (should match value sent in DCI, range 0-31)"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_table_index", "uint8", "MCS-Table-PUSCH index (0=notqam256, 1=qam256, 2=qam64LowSE, etc.)"));
		
		// Resource allocation streams
		available_data_streams.push_back(createIndicationPayloadStream("rb_start", "uint16", "Starting resource block within the BWP for this PUSCH (resource allocation type 1)"));
		available_data_streams.push_back(createIndicationPayloadStream("rb_size", "uint16", "Number of resource blocks for this PUSCH (resource allocation type 1)"));
		available_data_streams.push_back(createIndicationPayloadStream("start_symbol_index", "uint8", "Start symbol index of PUSCH mapping from the start of the slot (range 0-13)"));
		available_data_streams.push_back(createIndicationPayloadStream("nr_of_symbols", "uint8", "PUSCH duration in symbols (range 1-14)"));
		
		// Additional quality metrics streams
		available_data_streams.push_back(createIndicationPayloadStream("cqi", "float32", "SINR post-equalization per UE in dB (also known as CQI)"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_count", "uint16", "Number of Code Blocks per UE transport block"));
		available_data_streams.push_back(createIndicationPayloadStream("rssi", "float32", "Received Signal Strength Indicator per UE group in dB"));
		
		e3_setup_response["available_data_streams"] = available_data_streams;
		
		// Available control interfaces (placeholder)
		e3_setup_response["available_control_interfaces"] = json::array();
		
		// Subscription info (placeholder)
		e3_setup_response["subscription_info"] = json::object();
		
		// Error handling (placeholder)
		e3_setup_response["error_handling"] = json::object();
		
		response_json = e3_setup_response;
		
		NVLOGC_FMT(TAG_E3, "E3 Setup successful for dApp '{}' assigned ID: {}", dapp_name, dapp_id);
		
	} catch (const json::exception& e) {
		NVLOGC_FMT(TAG_E3, "Error processing E3 Setup Request: {}", e.what());
		json error_resp;
		error_resp["status"] = "error";
		error_resp["message"] = "invalid setup request format";
		response = error_resp.dump();
		return;
	}
	
	response = response_json.dump();
}

void E3Agent::handleSubscriptionRequest(const json& request, std::string& response)
{
    json response_json;
    json e3_sub_response;
    uint32_t dapp_id = 0;
    uint32_t transaction_id = 0;

    try {
        const json& e3_sub_req = request;
        dapp_id = e3_sub_req.at("dapp_id").get<uint32_t>();
        transaction_id = e3_sub_req.value("transaction_id", 0u);

        bool dapp_is_connected;
        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            auto it = e3_connected_dapps.find(dapp_id);
            if (it != e3_connected_dapps.end()) {
                dapp_is_connected = true;
                it->second.last_activity_time = std::chrono::steady_clock::now();
            } else {
                dapp_is_connected = false;
            }
        }

        if (!dapp_is_connected) {
            NVLOGC_FMT(TAG_E3, "Subscription rejected for non-connected dApp {}", dapp_id);
            e3_sub_response["status"] = "error";
            e3_sub_response["message"] = "dApp not connected or timed out";
        } else {
            const json& sub_details = e3_sub_req.at("subscription_details");
            std::vector<std::string> requested_streams = sub_details.at("requested_streams").get<std::vector<std::string>>();
            uint32_t interval_ms = sub_details.at("interval_ms").get<uint32_t>();

            std::vector<std::string> granted_streams = requested_streams;
            e3::StreamType stream_bitfield = e3::streamVectorToBitfield(granted_streams);
            std::string sub_id = generateSubscriptionId(dapp_id);

            {
                std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
                e3_subscriptions[sub_id] = {sub_id, dapp_id, granted_streams, stream_bitfield, interval_ms, std::chrono::steady_clock::now(), true};
            }

            NVLOGC_FMT(TAG_E3, "E3 Subscription '{}' created for dApp {}", sub_id, dapp_id);

            e3_sub_response["status"] = "success";
            e3_sub_response["subscription_id"] = sub_id;

            json sub_info;
            sub_info["granted_streams"] = granted_streams;
            sub_info["interval_ms"] = interval_ms;
            e3_sub_response["subscription_info"] = sub_info;
        }

        e3_sub_response["type"] = "e3_subscription_response";
        e3_sub_response["message_id"] = generateResponseMessageId("sub_resp");
        e3_sub_response["timestamp"] = generateTimestamp();
        e3_sub_response["transaction_id"] = transaction_id;
        e3_sub_response["dapp_id"] = dapp_id;
        response_json = e3_sub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Subscription Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "e3_subscription_response";
        e3_err_resp["status"] = "error";
        e3_err_resp["message"] = "missing or invalid parameters in subscription request";
        if (dapp_id != 0) e3_err_resp["dapp_id"] = dapp_id;
        if (transaction_id != 0) e3_err_resp["transaction_id"] = transaction_id;
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}

void E3Agent::handleUnsubscriptionRequest(const json& request, std::string& response)
{
    json response_json;
    json e3_unsub_response;

    try {
        const json& e3_unsub_req = request;
        uint32_t dapp_id = e3_unsub_req.at("dapp_id").get<uint32_t>();
        std::string sub_id = e3_unsub_req.at("subscription_id").get<std::string>();
        uint32_t transaction_id = e3_unsub_req.value("transaction_id", 0u);

        bool found = false;
        {
            std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
            auto it = e3_subscriptions.find(sub_id);
            if (it != e3_subscriptions.end() && it->second.dapp_id == dapp_id) {
                e3_subscriptions.erase(it);
                found = true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            auto it = e3_connected_dapps.find(dapp_id);
            if (it != e3_connected_dapps.end()) {
                it->second.last_activity_time = std::chrono::steady_clock::now();
            }
        }

        if (found) {
            NVLOGC_FMT(TAG_E3, "E3 Unsubscription successful for subscription {}", sub_id);
            e3_unsub_response["status"] = "success";
        } else {
            NVLOGC_FMT(TAG_E3, "Unsubscription failed for sub_id {}, dApp_id {}", sub_id, dapp_id);
            e3_unsub_response["status"] = "error";
            e3_unsub_response["message"] = "subscription not found or dApp ID mismatch";
        }

        e3_unsub_response["subscription_id"] = sub_id;
        e3_unsub_response["type"] = "e3_unsubscription_response";
        e3_unsub_response["message_id"] = generateResponseMessageId("unsub_resp");
        e3_unsub_response["timestamp"] = generateTimestamp();
        e3_unsub_response["transaction_id"] = transaction_id;
        e3_unsub_response["dapp_id"] = dapp_id;
        response_json = e3_unsub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Unsubscription Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "e3_unsubscription_response";
        e3_err_resp["status"] = "error";
        e3_err_resp["message"] = "missing or invalid parameters in unsubscription request";
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}

void E3Agent::handleControlMessage(const json& request, std::string& response)
{
    json response_json;
    json e3_control_response;
    uint32_t dapp_id = 0;
    uint32_t transaction_id = 0;

    try {
        const json& e3_control_req = request;
        dapp_id = e3_control_req.at("dapp_id");
        transaction_id = e3_control_req.value("transaction_id", 0u);

        const json& control_message = e3_control_req.at("control_message");
        const uint16_t sfn = control_message.at("sfn");
        const uint16_t slot = control_message.at("slot");

        {
            std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
            dataLake->e3_buffer_info.sfn = sfn;
            dataLake->e3_buffer_info.slot = slot;
        }

        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            auto it = e3_connected_dapps.find(dapp_id);
            if (it != e3_connected_dapps.end()) {
                it->second.last_activity_time = std::chrono::steady_clock::now();
            }
        }

        NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: handleControlMessage (Op #10) from dApp {} sfn = {}, slot = {} at {}", dapp_id, sfn, slot, std::chrono::high_resolution_clock::now().time_since_epoch().count());
        e3_control_response["status"] = "success";

    } catch (const json::exception& e) {
        e3_control_response["status"] = "error";
        e3_control_response["message"] = "invalid data format";
        NVLOGC_FMT(TAG_E3, "Error processing E3 Control Message: {}. Request: {}", e.what(), request.dump());
    }

    e3_control_response["type"] = "e3_control_response";
    e3_control_response["message_id"] = generateResponseMessageId("ctrl_resp");
    e3_control_response["timestamp"] = generateTimestamp();
    e3_control_response["transaction_id"] = transaction_id;
    e3_control_response["dapp_id"] = dapp_id;

    response_json = e3_control_response;
    response = response_json.dump();
}
