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

#include "context.hpp"
#include "shm_logger.h"
#include "oran.hpp"
#define UL_PCAP_MAX_FILENAME_LEN 512

void write_pcap_file(uint8_t* pcap_buffer, uint8_t* pcap_buffer_ts, shmlogger_t* logger, uint16_t mtu)
{
    shmlogger_reset(logger);
    for(int i = 0; i < MAX_PKTS_PER_PCAP_BUFFER; i++)
    {
        int offset = i * mtu;
        auto pkt = &pcap_buffer[offset];
        auto pkt_ts = (reinterpret_cast<uint64_t*>(pcap_buffer_ts))[i];
        auto ecpri_len = oran_umsg_get_ecpri_payload(pkt);
        shmlogger_save_fh_buffer(logger, (const char*)pkt, ecpri_len + ORAN_ETH_HDR_SIZE + 4, 0, pkt_ts); // +4 for padding
    }
}

void generate_pcap_filename(char* output_filename, size_t max_len,
                          uint8_t frame_id, uint8_t subframe_id,
                          uint8_t slot_id, uint8_t cell_id)
{
    // Get current time
    struct timeval tv;
    struct tm* timeinfo;
    char timestamp[32];

    gettimeofday(&tv, NULL);
    timeinfo = localtime(&tv.tv_sec);

    // Format timestamp YYYYMMDD_HHMMSS
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);

    // Generate filename with format: YYYYMMDD_HHMMSS_F000_SF02_SL1_C0.pcap
    snprintf(output_filename, max_len,
             "%s_F%03d_SF%02d_SL%d_C%d",
             timestamp, frame_id, subframe_id, slot_id, cell_id);
}

void* ul_pcap_capture_thread_func(void* arg)
{
    nvlog_fmtlog_thread_init();
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    ul_pcap_capture_context_info_t& ctx_info = pdctx->ul_pcap_capture_context_info;
    Cell* c;
    uint8_t write_idx, read_idx = 0;
    shmlogger_config_t shm_cfg;
    shm_cfg.save_to_file = 1;               // Start a background thread to save SHM cache to file before overflow
    shm_cfg.shm_cache_size = (1L << 27);    // 128MB, shared memory size, a SHM file will be created at /dev/shm/${name}_pcap
    shm_cfg.max_file_size = (1L << 28);     // 256MB Max file size, a disk file will be created at /var/log/aerial/${name}_pcap
    shm_cfg.file_saving_core = -1;          // CPU core ID for the background file saving if enabled.
    shm_cfg.shm_caching_core = -1;          // CPU core ID for the background copying to shared memory if enabled.
    shm_cfg.max_data_size = 8000;
    auto shmlogger = shmlogger_open(1, "ul_crc_pcap", &shm_cfg);

    uint8_t* pcap_buffer = nullptr;
    uint8_t* pcap_buffer_ts = nullptr;
    uint16_t sfn = 0;
    uint16_t slot = 0;
    uint8_t cell_id = 0;
    uint8_t frameId = 0;
    uint8_t subframeId = 0;
    uint8_t slotId = 0;
    char output_filename[UL_PCAP_MAX_FILENAME_LEN];
    char timeofday[32];

    while(!pdctx->stop_ul_pcap_thread)
    {
        write_idx = ctx_info.ul_pcap_capture_write_idx;
        read_idx = ctx_info.ul_pcap_capture_read_idx;
        if(read_idx != write_idx)
        {
            // Get pcap capture context info
            auto& capture_info = ctx_info.ul_pcap_capture_info[read_idx];
            pcap_buffer = capture_info.buffer_pointer;
            pcap_buffer_ts = capture_info.buffer_pointer_ts;
            sfn = capture_info.sfn;
            slot = capture_info.slot;
            cell_id = capture_info.cell_id;

            write_pcap_file(pcap_buffer, pcap_buffer_ts, shmlogger, capture_info.mtu);
            frameId = sfn % ORAN_MAX_FRAME_ID;
            subframeId = slot / ORAN_MAX_SLOT_ID;
            slotId = slot % ORAN_MAX_SLOT_ID;
            generate_pcap_filename(output_filename, sizeof(output_filename), frameId, subframeId, slotId, cell_id);

            shmlog_collect_params_t params = {
                .prefix = "ul_crc",
                .type = "pcap",
                .path = ".",
                .fh_collect = 1,
                .output_filename = output_filename
            };

            shmlogger_collect_ex(&params);
            // Free pcap buffer
            ctx_info.ul_pcap_capture_read_idx=(ctx_info.ul_pcap_capture_read_idx+1)%UL_MAX_CELLS_PER_SLOT;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::nanoseconds(5000));
        }
    }
    return nullptr;
}

int PhyDriverCtx::launch_ul_capture_thread()
{
    //Create h2d copy prepone thread
    std::thread t;
    t = std::thread(&ul_pcap_capture_thread_func, (void*)this);
    ul_pcap_capture_thread.swap(t);

    int name_st = pthread_setname_np(ul_pcap_capture_thread.native_handle(), "ulpcap_thread");

    if (name_st != 0 )
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT ,"ul_pcap_capture_thread Thread pthread_setname_np failed with status: {}",std::strerror(name_st));
        return -1;
    }

    sched_param sch;
    int         policy;
    int         status = 0;
    //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    // Set thread CPU affinity
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ul_pcap_capture_thread_cpu_affinity, &cpuset);
    status = pthread_setaffinity_np(ul_pcap_capture_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if(status)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "ul_pcap_capture_thread setaffinity_np  failed with status : {}" , std::strerror(status));
        return -1;
    }

    if(ul_pcap_capture_thread_sched_priority>0)
    {
        // Set thread priority
        status = pthread_getschedparam(ul_pcap_capture_thread.native_handle(), &policy, &sch);
        if(status != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "ul_pcap_capture_thread pthread_getschedparam failed with status : {}", std::strerror(status));
            return -1;
        }
        sch.sched_priority = ul_pcap_capture_thread_sched_priority;

#ifdef ENABLE_SCHED_FIFO_ALL_RT
        status = pthread_setschedparam(ul_pcap_capture_thread.native_handle(), SCHED_FIFO, &sch);
        if(status != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "ul_pcap_capture_thread setschedparam failed with status : {}" , std::strerror(status));
            return -1;
        }
#endif
    }
    return 0;
}
