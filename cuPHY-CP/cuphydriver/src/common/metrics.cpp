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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 28) // "DRV.METRICS"

#include "metrics.hpp"
#include "context.hpp"

Metrics::Metrics(phydriver_handle _pdh, uint8_t _cpu_core) : pdh(_pdh), cpu_core(_cpu_core)
{
    id=0;
    wProm=nullptr;
}

Metrics::~Metrics() {
}

phydriver_handle Metrics::getPhyDriverHandler(void) const
{
    return pdh;
}

static int metrics_thread_wrapper(phydriverwrk_handle wh, void* arg)
{   
    auto pdctx = static_cast<PhyDriverCtx*>(arg);
    auto fhproxy = pdctx->getFhProxy();
    
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        fhproxy->updateMetrics();
    }

    return 0;
}

int Metrics::start()
{
#ifdef AERIAL_METRICS
    auto pdctx   = StaticConversion<PhyDriverCtx>(pdh).get();
    
    int ret = l1_worker_start_generic(pdh, &wProm, "metrics", cpu_core, 5, metrics_thread_wrapper, pdctx);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "metrics_thread_wrapper couldn't start");
        return -1;
    }
#endif

    return 0;
}

CellMetrics::CellMetrics(uint16_t _cell_id) : cell_key_name(std::to_string(_cell_id))
{
#ifdef AERIAL_METRICS
    auto &metrics_manager = AerialMetricsRegistrationManager::getInstance();

    dl_slots_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_slots_total", "Aerial cuPHY-CP total number of processed Downlink slots", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_DIRECTION_KEY, "DL"}});
    ul_slots_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_slots_total", "Aerial cuPHY-CP total number of processed Uplink slots", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_DIRECTION_KEY, "UL"}});
    pusch_lost_prbs_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_lost_prb_total", "Aerial cuPHY-CP total number of PRBs expected but not received in the PUSCH channel", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pusch"}});
    pusch_rx_tb_bytes_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_pusch_rx_tb_bytes_total", "Aerial cuPHY-CP total number of transport block bytes received in the PUSCH channel", {{METRIC_CELL_KEY, cell_key_name}});
    pusch_rx_tb_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_pusch_rx_tb_total", "Aerial cuPHY-CP total number of transport blocks received in the PUSCH channel", {{METRIC_CELL_KEY, cell_key_name}});
    pusch_rx_tb_crc_error_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_pusch_rx_tb_crc_error_total", "Aerial cuPHY-CP total number of transport blocks received with CRC errors in the PUSCH channel", {{METRIC_CELL_KEY, cell_key_name}});
    prach_lost_prbs_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_lost_prb_total", "Aerial cuPHY-CP total number of PRBs expected but not received in the PRACH channel", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "prach"}});
    prach_rx_preambles_total =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphy_prach_rx_preambles_total", "Aerial cuPHY-CP PRACH RX preambles", {{METRIC_CELL_KEY, cell_key_name}});
    pdsch_tx_bytes_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphy_pdsch_tx_tb_bytes_total", "Aerial cuPHY-CP total number of transport block bytes transmitted in the PDSCH channel", {{METRIC_CELL_KEY, cell_key_name}});
    pdsch_tx_tb_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphy_pdsch_tx_tb_total", "Aerial cuPHY-CP total number of transport blocks transmitted in the PDSCH channel", {{METRIC_CELL_KEY, cell_key_name}});
    srs_lost_prbs_total = &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_uplane_lost_prb_total", "Aerial cuPHY-CP total number of PRBs expected but not received in the SRS channel", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "srs"}});

    prometheus::Histogram::BucketBoundaries processing_time_buckets;
    for (uint64_t bucket_boundary = METRIC_PROCESSING_TIME_BIN_SIZE_US; bucket_boundary <= METRIC_PROCESSING_TIME_MAX_BIN_US; bucket_boundary += METRIC_PROCESSING_TIME_BIN_SIZE_US)
    {
        processing_time_buckets.push_back(bucket_boundary);
    }

    pdsch_processing_time = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_slot_processing_duration_us", "Aerial cuPHY-CP total number of PDSCH slots with GPU processing duration in each histogram bin", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pdsch"}}, processing_time_buckets);
    pucch_processing_time = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_slot_processing_duration_us", "Aerial cuPHY-CP total number of PUCCH slots with GPU processing duration in each histogram bin", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pucch"}}, processing_time_buckets);
    pusch_processing_time = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_slot_processing_duration_us", "Aerial cuPHY-CP total number of PUSCH slots with GPU processing duration in each histogram bin", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pusch"}}, processing_time_buckets);
    prach_processing_time = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_slot_processing_duration_us", "Aerial cuPHY-CP total number of PRACH slots with GPU processing duration in each histogram bin", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "prach"}}, processing_time_buckets);

    prometheus::Histogram::BucketBoundaries ue_per_slot_buckets;
    for (uint64_t bucket_boundary = METRIC_UE_PER_SLOT_BIN_SIZE; bucket_boundary <= METRIC_UE_PER_SLOT_MAX_BIN; bucket_boundary += METRIC_UE_PER_SLOT_BIN_SIZE)
    {
        ue_per_slot_buckets.push_back(bucket_boundary);
    }

    pusch_nr_of_ues_per_slot = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_nrofuesperslot", "Aerial cuPHY-CP total number of UEs processed in each slot per histogram bin in the PUSCH channel", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pusch"}}, ue_per_slot_buckets);
    pdsch_nr_of_ues_per_slot = &metrics_manager.addMetric<prometheus::Histogram>("aerial_cuphycp_nrofuesperslot", "Aerial cuPHY-CP total number of UEs processed in each slot per histogram bin in the PDSCH channel", {{METRIC_CELL_KEY, cell_key_name}, {METRIC_CHANNEL_KEY, "pdsch"}}, ue_per_slot_buckets);

    early_uplane_rx_packets_total =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_early_uplane_rx_packets_total", "Aerial cuPHY-CP Uplink U-plane packets which arrived before their receive windows", {{METRIC_CELL_KEY, cell_key_name}});
    on_time_uplane_rx_packets_total =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_on_time_uplane_rx_packets_total", "Aerial cuPHY-CP Uplink U-plane packets which arrived within their receive windows", {{METRIC_CELL_KEY, cell_key_name}});
    late_uplane_rx_packets_total =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_late_uplane_rx_packets_total", "Aerial cuPHY-CP Uplink U-plane packets which arrived after their receive windows", {{METRIC_CELL_KEY, cell_key_name}});

    early_uplane_rx_packets_total_srs =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_early_uplane_rx_packets_total_srs", "Aerial cuPHY-CP SRS U-plane packets which arrived before their receive windows", {{METRIC_CELL_KEY, cell_key_name}});
    on_time_uplane_rx_packets_total_srs =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_on_time_uplane_rx_packets_total_srs", "Aerial cuPHY-CP SRS U-plane packets which arrived within their receive windows", {{METRIC_CELL_KEY, cell_key_name}});
    late_uplane_rx_packets_total_srs =  &metrics_manager.addMetric<prometheus::Counter>("aerial_cuphycp_late_uplane_rx_packets_total_srs", "Aerial cuPHY-CP SRS U-plane packets which arrived after their receive windows", {{METRIC_CELL_KEY, cell_key_name}});
#endif
}

CellMetrics::~CellMetrics() {
}

// NOTE: For counters ('Total' suffix) the value is the increment
void CellMetrics::update(CellMetric metric, uint64_t value)
{
#ifdef AERIAL_METRICS
    switch(metric) {
        case CellMetric::kDlSlotsTotal: dl_slots_total->Increment(value); break;
        case CellMetric::kUlSlotsTotal: ul_slots_total->Increment(value); break;
        case CellMetric::kPuschLostPrbsTotal:  pusch_lost_prbs_total->Increment(value); break;
        case CellMetric::kPuschRxTbBytesTotal: pusch_rx_tb_bytes_total->Increment(value); break;
        case CellMetric::kPuschRxTbTotal: pusch_rx_tb_total->Increment(value); break;
        case CellMetric::kPuschRxTbCrcErrorTotal: pusch_rx_tb_crc_error_total->Increment(value); break;
        case CellMetric::kPrachLostPrbsTotal: prach_lost_prbs_total->Increment(value); break;
        case CellMetric::kPrachRxPreamblesTotal: prach_rx_preambles_total->Increment(value); break;
        case CellMetric::kPdschTxBytesTotal: pdsch_tx_bytes_total->Increment(value); break;
        case CellMetric::kPdschTxTbTotal: pdsch_tx_tb_total->Increment(value); break;
        case CellMetric::kPdschProcessingTime: pdsch_processing_time->Observe(value); break;
        case CellMetric::kPucchProcessingTime: pucch_processing_time->Observe(value); break;
        case CellMetric::kPuschProcessingTime: pusch_processing_time->Observe(value); break;
        case CellMetric::kPrachProcessingTime: prach_processing_time->Observe(value); break;
        case CellMetric::kPuschNrOfUesPerSlot: pusch_nr_of_ues_per_slot->Observe(value); break;
        case CellMetric::kPdschNrOfUesPerSlot: pdsch_nr_of_ues_per_slot->Observe(value); break;
        case CellMetric::kEarlyUplanePackets: early_uplane_rx_packets_total->Increment(value); break;
        case CellMetric::kOnTimeUplanePackets: on_time_uplane_rx_packets_total->Increment(value); break;
        case CellMetric::kLateUplanePackets: late_uplane_rx_packets_total->Increment(value); break;
        case CellMetric::kSrsLostPrbsTotal:  srs_lost_prbs_total->Increment(value); break;
        case CellMetric::kEarlyUplanePacketsSrs: early_uplane_rx_packets_total_srs->Increment(value); break;
        case CellMetric::kOnTimeUplanePacketsSrs: on_time_uplane_rx_packets_total_srs->Increment(value); break;
        case CellMetric::kLateUplanePacketsSrs: late_uplane_rx_packets_total_srs->Increment(value); break;
    }
#endif
}

void Packet_Statistics::increment_counter(int cell, timing_type type, int slot, int inc)
{
    if(cell >= MAX_CELLS_PER_SLOT || slot >= MAX_LAUNCH_PATTERN_SLOTS || type >= MAX_TIMING_TYPES)
    {
        // Invalid parameters
        return;
    }
    stats[cell][slot][type] += inc;
}

uint64_t Packet_Statistics::get_stat(int cell, timing_type type, int slot)
{
    if(cell >= MAX_CELLS_PER_SLOT || slot >= MAX_LAUNCH_PATTERN_SLOTS || type >= MAX_TIMING_TYPES)
    {
        // Invalid parameters
        return 0;
    }
    return stats[cell][slot][type].load();
}

void Packet_Statistics::clear_counter(int cell, timing_type type, int slot)
{
    if(cell >= MAX_CELLS_PER_SLOT || slot >= MAX_LAUNCH_PATTERN_SLOTS || type >= MAX_TIMING_TYPES)
    {
        // Invalid parameters
        return;
    }
    stats[cell][slot][type].store(0);
}

void Packet_Statistics::set_active_slot(int slot)
{
    slots[slot] = true;
    active_ = true;
}

void Packet_Statistics::reset()
{
    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
            for(int cell = 0; cell < MAX_CELLS_PER_SLOT; ++cell)
            {
                for(int timing = 0; timing < MAX_TIMING_TYPES; ++timing)
                {
                    stats[cell][slot][timing].store(0);
                }
            }
    }
}

void Packet_Statistics::flush_counters(int num_cells, bool isSRS)
{
    if(num_cells > MAX_CELLS_PER_SLOT)
    {
        // Invalid parameters
        return;
    }

    int buffer_index = 0;
    char buffer[MAX_PRINT_LOG_LENGTH];

    buffer_index = 0;
    buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "slot,");
    for(int cell = 0; cell < num_cells; ++cell)
    {
        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "cell_%d_early,cell_%d_ontime,cell_%d_late,", cell, cell, cell);
    }

    if(isSRS)
    {
        NVLOGC_FMT(METRICS_SRS_TAG,"{}",buffer);
    }
    else
    {
        NVLOGC_FMT(METRICS_TAG, "{}",buffer);
    }

    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(slots[slot])
        {
            buffer_index = 0;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "Slot %d |", slot);
            
            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " %lu,%lu,%lu |", early, ontime, late);
            }
            if(isSRS)
            {
                NVLOGC_FMT(METRICS_SRS_TAG, "{}",buffer);
            }
            else
            {
                NVLOGC_FMT(METRICS_TAG, "{}",buffer);
            }
        }
    }

    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(slots[slot])
        {
            buffer_index = 0;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "Slot %d |", slot);
            
            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                total = (total == 0) ? 1 : total;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " %.2f,%.2f,%.2f |", (float)early / total * 100.0, (float)ontime / total * 100.0, (float)late / total * 100.0);
            }
            if(isSRS)
            {
                NVLOGC_FMT(METRICS_SRS_TAG, "{}",buffer);
            }
            else
            {
                NVLOGC_FMT(METRICS_TAG, "{}",buffer);
            }
        }
    }
}


void Packet_Statistics::flush_counters_file(int num_cells, std::string filename)
{
    if(num_cells > MAX_CELLS_PER_SLOT)
    {
        // Invalid parameters
        return;
    }

    int buffer_index = 0;
    char buffer[MAX_PRINT_LOG_LENGTH];

    FILE *fp;

    fp = fopen(filename.c_str(), "w+");

    buffer_index = 0;
    buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "slot,");
    for(int cell = 0; cell < num_cells; ++cell)
    {
        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "cell_%d_early,cell_%d_ontime,cell_%d_late,", cell, cell, cell);
    }

    fprintf(fp, "%s\n", buffer);


    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(slots[slot])
        {
            buffer_index = 0;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "%d,", slot);
            
            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "%lu,%lu,%lu,", early, ontime, late);
            }
            fprintf(fp, "%s\n", buffer);
        }
    }
   fclose(fp);
}


