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

#include "util.hpp"
#include "srs_tx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include <list>
#include <fstream>
#include "test_config.hpp"
#include "memtrace.h"

#define SRSTX_STREAM_PRIORITY 0

using namespace cuphy;

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector (first iteration only).\n");
    printf("    -g                          Execute all UEs in a slot on the same SRSTX object with batching too.\n");
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
}

int main(int argc, char* argv[])
{
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "srs_tx_multiue.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations  = 1;
    bool        ref_check_srstx = false;
    uint32_t    delayUs         = 10000;
    bool        group_ues       = false;
    int         time_setup_mode = 0;     // default mode: only time run, not setup
    std::string setup_modes[3]  = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};
    std::string proc_modes[2]  = {"in Stream mode", "in Graphs mode"};

    CUDA_CHECK(cudaSetDevice(0));

    while(iArg < argc)
    {
        if('-' == argv[iArg][0])
        {
            switch(argv[iArg][1])
            {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i':
                if(++iArg >= argc)
                {
                    NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'd':
                    delayUs = std::stoi(argv[++iArg]);
                    ++iArg;
                    break;
            case 'g':
                group_ues = true;
                ++iArg;
                break;
            case 'k':
                ref_check_srstx = true; // Note: enabled by default when input file is single HDF5 TV and not a yaml
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    std::unique_ptr<cuphy::test_config> testCfg;

    // Default value of 1 for num_ues and num_slots used when input file is HDF5 file and not a yaml.
    int num_ues   = 1;
    int num_slots = 1;

    const std:: string channelName = "SRSTX";
    std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
    //NVLOGC_FMT(NVLOG_SRSTX, "File extension: {}", inFileExtn.c_str());
    if(inFileExtn == "yaml")
    {
        testCfg = std::make_unique<cuphy::test_config>(inputFileName.c_str());
        testCfg->print_channel(channelName); // print only SRSTX channel related parts of the input YAML file
        //testCfg->print(); // print entire YAML file contents, incl. other channels not run with this example
        num_ues = testCfg->num_cells(); 
        num_slots = testCfg->num_slots();
    } else {
        NVLOGC_FMT(NVLOG_SRSTX, "SRS TX Pipeline single-ue example with TV {} and ref. checks enabled",  inputFileName.c_str());
        ref_check_srstx = true; 
    }

    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device 0

    std::vector<std::vector<cuphy::srs_tx>>         m_srsTxPipes;
    std::vector<std::vector<srsTxStaticApiDataset>> m_srsTxStaticApiDataSets;
    std::vector<std::vector<srsTxDynApiDataset>>    m_srsTxDynamicApiDataSets;

    std::vector<stream>        streams;
    cudaEvent_t                start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_ues);
    const int num_srstx_objects = group_ues ? 1 : num_ues;
    m_srsTxPipes.resize(num_slots);
    m_srsTxStaticApiDataSets.resize(num_slots);
    m_srsTxDynamicApiDataSets.resize(num_slots);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        if (testCfg) {
            int ues_in_slot = testCfg->slots()[idxSlot].at(channelName).size();
            if (ues_in_slot != num_ues) {
                NVLOGE_FMT(NVLOG_SRSTX, AERIAL_CUPHY_EVENT, "Slot {} error: expected {} ues but got {} ues", idxSlot, num_ues, ues_in_slot);
                exit(1);
            }
        }
        
        m_srsTxStaticApiDataSets[idxSlot].reserve(num_srstx_objects);
        m_srsTxDynamicApiDataSets[idxSlot].reserve(num_srstx_objects);

        for(int i = 0; i < num_ues; i += 1)
        {
            std::string tv_filename = (testCfg) ? testCfg->slots()[idxSlot].at(channelName)[i] : inputFileName;
            if((idxSlot == 0) && (i < num_srstx_objects)) {
                streams.emplace_back(cudaStreamNonBlocking, SRSTX_STREAM_PRIORITY);
            }
            if (group_ues) {
                if (i == 0) {
                    m_srsTxStaticApiDataSets[idxSlot].emplace_back(tv_filename, num_ues);
                    m_srsTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_srsTxStaticApiDataSets[idxSlot][i].srsTxStatPrms.nMaxSrsUes, streams[i].handle());
                    m_srsTxPipes[idxSlot].emplace_back(m_srsTxStaticApiDataSets[idxSlot][i].srsTxStatPrms);
                } else {
                    // Update the dynamic parameters
                    m_srsTxDynamicApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle());
                }
            } else {
                m_srsTxStaticApiDataSets[idxSlot].emplace_back(tv_filename);
                m_srsTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_srsTxStaticApiDataSets[idxSlot][i].srsTxStatPrms.nMaxSrsUes, streams[i].handle());
                m_srsTxPipes[idxSlot].emplace_back(m_srsTxStaticApiDataSets[idxSlot][i].srsTxStatPrms);
            }
        }
    }
    if (group_ues)
    {
        NVLOGC_FMT(NVLOG_SRSTX, "Grouping all {} ues in a slot.", num_ues);
    } else {
        NVLOGC_FMT(NVLOG_SRSTX, "");
    }
    if (time_setup_mode== 0) {
        NVLOGC_FMT(NVLOG_SRSTX, "- NB: Allocations, setup processing not included.");
        NVLOGC_FMT(NVLOG_SRSTX, "");
    }

    for(int i = 0; i < num_srstx_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
    float total_time_single_ue_slot[num_slots][num_srstx_objects];
    //NVLOGC_FMT(NVLOG_SRSTX, "num slots {}, num_srstx_objects {}", num_slots, num_srstx_objects);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_ue(num_srstx_objects, 0);
        std::vector<event_timer> cuphy_timer_single_ue(num_srstx_objects);

        if (time_setup_mode == 0){ // In this mode, setup is not timed
            for(int i = 0; i < num_srstx_objects; i++)
            {
                m_srsTxPipes[idxSlot][i].setup(m_srsTxDynamicApiDataSets[idxSlot][i].srstx_dyn_params);
            }
        }
#if 0
        if (idxSlot == 0) { // why warmup?
            m_srsTxPipes[idxSlot][0].setup(m_srsTxDynamicApiDataSets[idxSlot][0].srstx_dyn_params);
            m_srsTxPipes[idxSlot][0].run();
        }
#endif
        if (testCfg) {
            gpu_us_delay(delayUs, 0, streams[0].handle());
        }
        CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

        // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

        for(int iter = 0; iter < num_iterations; iter++)
        {
            event_timer cuphy_timer;
            cuphy_timer.record_begin(streams[0].handle());

            for(int i = 0; i < num_srstx_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }

                cuphy_timer_single_ue[i].record_begin(streams[i].handle());
                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    m_srsTxPipes[idxSlot][i].setup(m_srsTxDynamicApiDataSets[idxSlot][i].srstx_dyn_params);
                }
                if (time_setup_mode != 1)
                {
                    m_srsTxPipes[idxSlot][i].run();
                }
                cuphy_timer_single_ue[i].record_end(streams[i].handle());

                /* Record a stop event on all streams but the streams[0].handle() stream. */
                /* Have streams[0].handle() stream wait for all other streams to complete their work before stopping the timer. */
                if(i != 0)
                {
                    CUDA_CHECK(cudaEventRecord(stop_streams_events[i], strm_handle));
                    CUDA_CHECK(cudaStreamWaitEvent(streams[0].handle(), stop_streams_events[i], 0));
                }
            }

            cuphy_timer.record_end(streams[0].handle());
            cuphy_timer.synchronize();
            total_time += cuphy_timer.elapsed_time_ms();

            for(int i = 0; i < num_srstx_objects; i++)
            {
                cuphy_timer_single_ue[i].synchronize(); // To be safe
                total_time_single_ue[i] += cuphy_timer_single_ue[i].elapsed_time_ms();
            }

            if (ref_check_srstx && (iter == 0)) {
               for(int i = 0; i < num_srstx_objects; i++)
               {
                  int errors = m_srsTxDynamicApiDataSets[idxSlot][i].refCheck(true);
                  if (errors != 0) {
                      exit(1);
                  }
               }
            }
            if (testCfg) {
                gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            }
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;
        memtrace_set_config(0); // disable memory allocation tracing beyond this point

        for(int i = 0; i < num_srstx_objects; i++)
        {
            total_time_single_ue_slot[idxSlot][i] = total_time_single_ue[i] / num_iterations;
        }
    }
    
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_SRSTX, "Slot # {}, SRS TX pipeline(s) {} {}: {:.2f} us (avg. over {} iterations).", idxSlot, setup_modes[time_setup_mode].c_str(), proc_modes[1].c_str(), total_time_slot[idxSlot] * 1000, num_iterations);
        for(int i = 0; i < num_srstx_objects; i++)
        {
            if (group_ues) {
                NVLOGC_FMT(NVLOG_SRSTX, "--> SRS TX object # {} with {} ues: {:.2f} us (avg over {} iterations)", i, num_ues, total_time_single_ue_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_SRSTX, "--> UE # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_ue_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}
