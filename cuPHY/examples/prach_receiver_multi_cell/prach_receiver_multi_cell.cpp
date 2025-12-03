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

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdio>

#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "util.hpp"
#include "datasets.hpp"
#include "test_config.hpp"

#include "CLI/CLI.hpp"

//#define CUPHY_MEMTRACE // uncomment to exercise in cuPHY local run

#ifdef CUPHY_MEMTRACE
#include "memtrace.h"
#endif
#define NUM_PREAMBLE 64

using namespace std;
using namespace cuphy;

int main(int argc, char* argv[])
{

    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "prach.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;

    //------------------------------------------------------------------
    // Parse command line arguments
    constexpr uint32_t N_MAX_INST = 36;
    std::string inputFileName;
    bool        enableOutputFileLog = false;
    std::string outputFileName;
    uint32_t    num_iterations      = 1;
    uint32_t    nInst               = 1;
    bool        ref_check_prach     = false;
    uint32_t    delayUs             = 10000;
    bool        group_cells         = false; // Group cells in the same cell-group.
    uint64_t    procModeBmsk        = PRACH_PROC_MODE_NO_GRAPH;
    uint32_t    SMsPerGreenCtx      = 0;

    try
    {
        // Parse command line arguments using CLI11
        CLI::App app{"prach_receiver_multi_cell"};
        std::ignore = app.add_option("-l,--log", log_name, "Filename to save log output");
        std::ignore = app.add_option("-i,--input", inputFileName, "Input yaml or HDF5 file for slot/cell config")
                            ->required()
                            ->check(CLI::ExistingFile);
        std::ignore = app.add_option("-n,--cells", nInst, fmt::format("Number of cells to run [1,{}] (used only for HDF5 input)", N_MAX_INST))
                            ->check(CLI::Range(1, static_cast<int>(N_MAX_INST)));
        std::ignore = app.add_option("-r,--iterations", num_iterations, "Number of run iterations to run")
                            ->check(CLI::PositiveNumber);
        std::ignore = app.add_option("-d,--delay-us", delayUs, "Delay kernel duration in microseconds")
                            ->check(CLI::NonNegativeNumber);
        std::ignore = app.add_option("-m,--mode", procModeBmsk, "Proc mode: streams(0x0), graphs(0x1) (default = 0x0)");
        std::ignore = app.add_flag("-k,--ref-check", ref_check_prach, "Enable reference check. Compare GPU output with test vector.");
        std::ignore = app.add_flag("-g,--group-cells", group_cells, "Execute all cells in a slot on the same PRACH object.");
        auto* out_opt = app.add_option("-o,--output", outputFileName, "Output HDF5 debug filename (enables debug write)");
        std::ignore = app.add_option("--G", SMsPerGreenCtx, "Use green contexts (GC) with specified SM count per context (default = 0; GC disabled)")
                            ->check(CLI::NonNegativeNumber);
        CLI11_PARSE(app, argc, argv)

        enableOutputFileLog = out_opt->count() > 0;
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
        if (procModeBmsk != PRACH_PROC_MODE_NO_GRAPH && procModeBmsk != PRACH_PROC_MODE_WITH_GRAPH) {
            const auto err = fmt::format("Invalid processing mode: 0x{:x}", procModeBmsk);
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "{}", err);
            return app.exit({"Parse error", err});
        }

        int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
        CUdevResource split_result[2] = {{}, {}};
        cuphy::cudaGreenContext prach_green_ctx;
        unsigned int split_groups = 1;

        if(SMsPerGreenCtx > 0)
        {
            // Best to ensure that MPS service is not running
            int mpsEnabled = 0;
            CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, current_device));
            if (mpsEnabled == 1) {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
                //exit(1);
            } else {
                NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "MPS service is not running.");
            }

            //Check SMsPerGreenCtxs value is in valid range
            int32_t gpuMaxSmCount = 0;
            CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_device));
            if (SMsPerGreenCtx > (uint32_t)gpuMaxSmCount)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid --G argument {}. It is greater than {} (GPU's max SMs).", SMsPerGreenCtx, gpuMaxSmCount);
                exit(1);
            }

            CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
            CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, SMsPerGreenCtx));
            prach_green_ctx.create(gpuId, &split_result[0]);
            prach_green_ctx.bind();
            NVLOGC_FMT(NVLOG_PRACH, "PRACH green context will have access to {} SMs ({} SMs requested).", prach_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        std::unique_ptr<cuphy::test_config> testCfg;
        int num_cells = 1;
        int num_slots = 1;

        // Debug output file
        bool autoGeneratedOutputFile = false;
        // If no output file was specified, auto-generate a timestamped one based on the input file.
        if (outputFileName.empty() && !inputFileName.empty())
        {
            std::string baseName = inputFileName;
            const std::size_t slashPos = baseName.find_last_of("/\\");
            if (slashPos != std::string::npos)
            {
                baseName = baseName.substr(slashPos + 1);
            }
            const std::size_t dotPos = baseName.find_last_of('.');
            if (dotPos != std::string::npos)
            {
                baseName = baseName.substr(0, dotPos);
            }

            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm{};
            localtime_r(&tt, &tm);
            char ts[32]{};
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm);
            outputFileName = fmt::format("{}_{}.h5", baseName, ts);
            autoGeneratedOutputFile = true;
        }
        std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
        if(!outputFileName.empty())
        {
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFileName.c_str())));
        }
        
        std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        NVLOGC_FMT(NVLOG_PRACH, "File extension: {}", inFileExtn);
        if(inFileExtn == "yaml")
        {
            testCfg = make_unique<cuphy::test_config>(inputFileName.c_str());
            testCfg->print();

            num_cells = testCfg->num_cells(); // The same number of cells is present across all slots.
            num_slots = testCfg->num_slots();
        }
        else
        {
            num_cells = nInst;
            NVLOGC_FMT(NVLOG_PRACH, "number of cells: {}", nInst);
            NVLOGC_FMT(NVLOG_PRACH, "File: {}", inputFileName);
        }
        
        const std:: string channelName = "PRACH";

        std::vector<std::vector<prach_rx>>          m_prachRxPipes;
        std::vector<std::vector<PrachApiDataset>>   m_prachApiDataSets;

        std::vector<stream>        streams;
        cudaEvent_t                start_streams_event;
        std::vector<cudaEvent_t> stop_streams_events(num_cells);

        const int num_prach_objects = group_cells ? 1 : num_cells;

        m_prachRxPipes.resize(num_slots);
        m_prachApiDataSets.resize(num_slots);

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
            // Reminder num_prach_objects is the number of cells, if group_cells is false, or 1 otherwise
            m_prachApiDataSets[idxSlot].reserve(num_prach_objects);

            // Loop over all cells (not num_prach_objects) to populate the datasets
            for(int i = 0; i < num_cells; i += 1)
            {
                std::string tv_filename;
                if(testCfg)
                {
                    tv_filename = testCfg->slots()[idxSlot].at(channelName)[i];
                }
                else
                {
                    tv_filename = inputFileName;
                }

                if((idxSlot == 0) && (i < num_prach_objects)) {
                    streams.emplace_back(cudaStreamNonBlocking);
                }
                if (group_cells) {
                    if (i == 0) {
                        m_prachApiDataSets[idxSlot].emplace_back(tv_filename, streams[i].handle(), procModeBmsk, ref_check_prach, outputFileName);
                    } else {
                        // Nothing to cumulative update for the static parameters.
                        // Update the dyanmic parameters
                        m_prachApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, outputFileName, streams[0].handle());
                    }
                } else {
                    m_prachApiDataSets[idxSlot].emplace_back(tv_filename, streams[i].handle(), procModeBmsk, ref_check_prach, outputFileName);
                    m_prachApiDataSets[idxSlot][i].finalize(streams[i].handle());
                    m_prachRxPipes[idxSlot].emplace_back(m_prachApiDataSets[idxSlot][i].prachStatPrms);
                }
            }

            if (group_cells) {
                m_prachApiDataSets[idxSlot][0].finalize(streams[0].handle());
                m_prachRxPipes[idxSlot].emplace_back(m_prachApiDataSets[idxSlot][0].prachStatPrms);
            }
        }

        NVLOGC_FMT(NVLOG_PRACH, "Timing {} PRACH pipeline(s) Run(). ",num_cells);
        if (group_cells)
        {
            NVLOGC_FMT(NVLOG_PRACH, "Grouping all cells in a slot.");
        }
        NVLOGC_FMT(NVLOG_PRACH, "- NB: Allocations, setup processing not included.\n");

        /* Prach::Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
        Have that stream wait for all other streams to complete their work too. */

        for(int i = 0; i < num_prach_objects; i++)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

        float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
        float total_time_single_cell_slot[num_slots][num_prach_objects];

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            float                    total_time = 0;
            std::vector<float>       total_time_single_cell(num_prach_objects, 0);
            std::vector<event_timer> cuphy_timer_single_cell(num_prach_objects);
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

            for(int i = 0; i < num_prach_objects; i++)
            {
                m_prachRxPipes[idxSlot][i].setup(m_prachApiDataSets[idxSlot][i].prachDynPrms);
            }
            gpu_us_delay(delayUs, 0, streams[0].handle());
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(0);
#endif
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

            for(int iter = 0; iter < num_iterations; iter++)
            {
                event_timer cuphy_timer;
                cuphy_timer.record_begin(streams[0].handle());

                for(int i = 0; i < num_prach_objects; i++)
                {
                    cudaStream_t strm_handle = streams[i].handle();

                    if(i != 0)
                    {
                        CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                    }
                    cuphy_timer_single_cell[i].record_begin(streams[i].handle());

#ifdef CUPHY_MEMTRACE
                    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
                    m_prachRxPipes[idxSlot][i].run();
                    memtrace_set_config(0);
#else
                    m_prachRxPipes[idxSlot][i].run();
#endif

                    cuphy_timer_single_cell[i].record_end(streams[i].handle());

                    if(enableOutputFileLog)
                    {
                        m_prachRxPipes[idxSlot][i].writeDbgSynch(streams[i].handle());
                    }

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

                for(int i = 0; i < num_prach_objects; i++)
                {
                    cuphy_timer_single_cell[i].synchronize(); // To be safe
                    total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
                }

                if (ref_check_prach && (iter == 0)) {
                    int errors = 0;
                    for(int i = 0; i < num_prach_objects; i++)
                    {
                        errors += m_prachApiDataSets[idxSlot][i].evaluateOutput();
                    }

                    if (errors != 0) {
                        for (int i = 0; i < num_prach_objects; i++)
                        {
                            m_prachRxPipes[idxSlot][i].writeDbgSynch(streams[i].handle());
                        }
                        CUDA_CHECK(cudaDeviceSynchronize());
                        exit(1);
                    }
                }
                gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
                CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
            }
            total_time_slot[idxSlot] = total_time / num_iterations;

            for(int i = 0; i < num_prach_objects; i++)
            {
                total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
            }
        }

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            NVLOGC_FMT(NVLOG_PRACH, "Slot # {},  PRACH pipeline(s): {:.2f} us (avg. over {} iterations)", idxSlot, total_time_slot[idxSlot] * 1000, num_iterations);
            for(int i = 0; i < num_prach_objects; i++)
            {
                if (group_cells) {
                    NVLOGC_FMT(NVLOG_PRACH, "--> PRACH object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
                } else {
                    NVLOGC_FMT(NVLOG_PRACH, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
                }
            }
        }

        // If we auto-generated a debug file for potential failures but no failure occurred, close any local handle and delete the file
        if (autoGeneratedOutputFile)
        {
            // Destroy pipelines to release any HDF5 handles they may be holding
            for (auto& v : m_prachRxPipes) { v.clear(); }
            m_prachRxPipes.clear();
            for (auto& v : m_prachApiDataSets) { v.clear(); }
            m_prachApiDataSets.clear();
            debugFile.reset();
            std::remove(outputFileName.c_str());
        }
    }
    catch(std::exception& e)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;

}
