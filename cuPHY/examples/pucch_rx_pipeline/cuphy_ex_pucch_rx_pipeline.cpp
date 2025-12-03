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

#include "cuphy_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "pucch_rx.hpp"
#include "test_config.hpp"
#include "nvlog.hpp"

//#define CUPHY_MEMTRACE // uncomment to exercise in cuPHY local run
#ifdef CUPHY_MEMTRACE
#include "memtrace.h"
#endif

#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;


////////////////////////////////////////////////////////////////////////
// usage()
void usage(char* argv[])
{
    printf("%s [options]\n", argv[0]);
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -l  log_filename           filename to save log output\n");
    printf("    -m  processing mode        PUCCH proc mode: streams (0x0), graphs (0x1)\n");
    printf("    -o  output_filename        Output HDFS debug file\n");
    printf("    -r  num                    Number of iterations to run\n");
    printf("    --G SM count               Use green contexts with specified SM count per context.\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "pucch.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;
    bool     useGreenCtxs       = false;
    uint32_t SMsPerGreenCtx     = 0;

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();
        std::string outputFilename = std::string();
        uint64_t    procModeBmsk   = PUCCH_PROC_MODE_FULL_SLOT;
        int         totalIters     = 1;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                    case 'i':
                        if(++iArg >= argc)
                        {
                            throw std::invalid_argument("No valid filename provided");
                        }
                        inputFilename.assign(argv[iArg++]);
                        break;
                    case 'l':
                        if(++iArg < argc)
                        {
                            log_name.assign(argv[iArg++]);
                        }
                        break;
                    case 'o':
                        if(++iArg < argc)
                        {
                            outputFilename.assign(argv[iArg++]);
                        }
                        break;
                    case 'm':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT) && (procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)))
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({})", procModeBmsk);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'r':
                        if(++iArg < argc)
                        {
                            totalIters = std::stoi(argv[iArg++]);
                        }
                        break;
                    case '-':
                        switch(argv[iArg][2])
                        {
                           case 'G':
                               useGreenCtxs = true;
                               if(((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &SMsPerGreenCtx))) || (SMsPerGreenCtx == 0))
                               {
                                   // Will later check that SMsPerGreenCtx does not exceed the SMs of the GPU. This will also capture if a negative number was provided.
                                   NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid or missing useGreenCtxs argument (--G)");
                                   exit(1);
                               }
                               ++iArg;
                               break;
                           default:
                               NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                               usage(argv);
                               exit(1);
                               break;
                        }
                        break;
                    default:
                        usage(argv);
                        throw std::invalid_argument(fmt::format("Unknown option: {}", argv[iArg]));
                }
            }
            else
            {
                throw std::invalid_argument(fmt::format("Invalid command line argument: {}", argv[iArg]));
            }
        }
        if(inputFilename.empty())
        {
            usage(argv);
            throw std::invalid_argument("No valid filename provided");
        }
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
        if(procModeBmsk == PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)
        {
            NVLOGI_FMT(NVLOG_PUCCH, "CUDA graph enabled!");
        } else {
            NVLOGI_FMT(NVLOG_PUCCH, "CUDA stream mode");
        }

        int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
        CUdevResource split_result[2] = {{}, {}};
        cuphy::cudaGreenContext pucch_green_ctx;
        unsigned int split_groups = 1;

        if(useGreenCtxs)
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
            pucch_green_ctx.create(gpuId, &split_result[0]);
            pucch_green_ctx.bind();
            NVLOGC_FMT(NVLOG_PUCCH, "PUCCH green context will have access to {} SMs ({} SMs requested).", pucch_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        // Initialize debug instrumentation
        // CUDA Timers
        cuphy::event_timer evtTmrSetup;
        cuphy::event_timer evtTmrRun;
        // CPU Start/Stop times
        TimePoint timePtStartSetup, timePtStopSetup;
        TimePoint timePtStartRun, timePtStopRun;

        duration<float, std::micro> elpasedTimeDurationUs;
        typedef enum _elapsedTypes
        {
            ELAPSED_CPU_SETUP = 0,
            ELAPSED_EVT_SETUP = 1,
            ELAPSED_CPU_RUN   = 2,
            ELAPSED_EVT_RUN   = 3,
            ELAPSED_TYPES_MAX
        } elapsedTypes_t;
        std::array<std::vector<float>,ELAPSED_TYPES_MAX> m_elapsedTimes;
        for(auto& timeVec : m_elapsedTimes)
        {
            timeVec.resize(totalIters);
        }

        //-----------------------------------------------------------------
        // Open debug file

        std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
        if(!outputFilename.empty())
        {
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
        }

        //------------------------------------------------------------------
        // input files
        std::vector<std::string> inputFileNameVec;
        std::string              inFileExtn = inputFilename.substr(inputFilename.find_last_of(".") + 1);
        if(inFileExtn == "yaml")
        {
            // yaml parsing
            cuphy::test_config testCfg(inputFilename.c_str());
            int                nCells           = testCfg.num_cells();
            int                nSlots           = testCfg.num_slots();
            const std::string  pucchChannelName = "PUCCH";

            try
            {
                for(size_t idxSlot = 0; idxSlot < nSlots; idxSlot++)
                {
                    for(int idxCell = 0; idxCell < nCells; idxCell++)
                    {
                        auto fname = testCfg.slots()[idxSlot].at(pucchChannelName)[idxCell];
                        inputFileNameVec.emplace_back(fname);
                    }
                }
            }
            catch(...)
            {
                throw std::runtime_error("PUCCH channel name not found in the input file");
            }
            assert(inputFileNameVec.size() == nCells);
        }
        else
        {
            inputFileNameVec.emplace_back(inputFilename);
        }

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain;
        cudaStream_t  cuStrm           = cuStrmMain.handle();

        pucchStaticApiDataset  statPucchApiDataset(inputFileNameVec, cuStrm, outputFilename);
        pucchDynApiDataset     dynPucchApiDataset (inputFileNameVec, cuStrm, procModeBmsk);
        EvalPucchDataset       evalPucchDataset   (inputFileNameVec, cuStrm);
        cuStrmMain.synchronize(); // synch to ensure data copied

        cuphyPucchDynPrms_t&  pucchDynPrm   = dynPucchApiDataset.pucchDynPrm;
        cuphyPucchStatPrms_t& pucchStatPrms =  statPucchApiDataset.pucchStatPrms;

        //------------------------------------------------------------------
        // allocate output buffers

        size_t MAX_N_F234_UCI  = CUPHY_PUCCH_F2_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI;
        
        //------------------------------------------------------------------
        // Finish setting dynamic parameters

        dynPucchApiDataset.pucchDynPrm.cuStream                       = cuStrm; // save stream in dynamic parameters
        dynPucchApiDataset.pucchDynPrm.cpuCopyOn                      = 1;      // option to copy uci output to CPU immediately after run
        //------------------------------------------------------------------
        // Create pucch reciever object

        cuphyPucchRxHndl_t pucchRxHndl;
        
        cuphyStatus_t statusCreate = cuphyCreatePucchRx(&pucchRxHndl, &pucchStatPrms, cuStrm);
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
#ifdef CUPHY_MEMTRACE
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

        for(int iterIdx=0; iterIdx<totalIters;iterIdx++)
        {
            
            auto& elapsedTimeUsSetup      = m_elapsedTimes[ELAPSED_CPU_SETUP][iterIdx];
            auto& elapsedTimeUsRun        = m_elapsedTimes[ELAPSED_CPU_RUN][iterIdx];
            auto& elapsedEvtTimeUsSetup   = m_elapsedTimes[ELAPSED_EVT_SETUP][iterIdx];
            auto& elapsedEvtTimeUsRun     = m_elapsedTimes[ELAPSED_EVT_RUN][iterIdx];

            //------------------------------------------------------------------
            // Setup pucch reciever object

            cuphyPucchBatchPrmHndl_t const batchPrmHndl = nullptr;  // batchPrms currently un-used
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

            // Record GPU & CPU time before setup
            evtTmrSetup.record_begin(cuStrm);
            timePtStartSetup = Clock::now();
            cuphyStatus_t statusSetup  = cuphySetupPucchRx(pucchRxHndl, &pucchDynPrm, batchPrmHndl);
            // Record GPU & CPU time after setup
            timePtStopSetup = Clock::now();
            evtTmrSetup.record_end(cuStrm);

            if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);

            //------------------------------------------------------------------
            // Run pucch reciever object

            uint64_t procModeBmsk = 0; // procModeBmsk currently un-used 

            // Record GPU & CPU time before run
            evtTmrRun.record_begin(cuStrm);
            timePtStartRun = Clock::now();
            cuphyStatus_t statusRun = cuphyRunPucchRx(pucchRxHndl, procModeBmsk);
            // Record GPU & CPU time after run
            timePtStopRun = Clock::now();
            evtTmrRun.record_end(cuStrm);
            
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            //------------------------------------------------------------------

            // Process timing data
            evtTmrSetup.synchronize();
            evtTmrRun.synchronize();
            cuStrmMain.synchronize();

            // Compare cuphy UCI output to reference
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(0); // Disable for evalPucchRxPipeline() call; not in the critical path
#endif
            if (iterIdx == 0)
            {
                evalPucchDataset.evalPucchRxPipeline(pucchDynPrm);
            }

            elpasedTimeDurationUs = timePtStopSetup - timePtStartSetup;
            elapsedTimeUsSetup    = elpasedTimeDurationUs.count();
            elpasedTimeDurationUs = timePtStopRun - timePtStartRun;
            elapsedTimeUsRun      = elpasedTimeDurationUs.count();
            elapsedEvtTimeUsSetup = evtTmrSetup.elapsed_time_ms()*1000;
            elapsedEvtTimeUsRun   = evtTmrRun.elapsed_time_ms()*1000;
        }
#ifdef CUPHY_MEMTRACE
        memtrace_set_config(0); // disable memory allocation tracing beyond this point
#endif
        //------------------------------------------------------------------
        // Save debug output

        if(!outputFilename.empty())
        {
            cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchPucch(pucchRxHndl, cuStrm);
            cuStrmMain.synchronize();
            if(CUPHY_STATUS_SUCCESS != statusDebugWrite) throw cuphy::cuphy_exception(statusDebugWrite);
        }
        // Timing Debug
        float avgElapsedTimesUs[ELAPSED_TYPES_MAX];
        float minElapsedTimesUs[ELAPSED_TYPES_MAX];
        float maxElapsedTimesUs[ELAPSED_TYPES_MAX];
        for(int i=0;i<ELAPSED_TYPES_MAX;i++)
        {
            const auto minmax_pair = std::minmax_element(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]));
            float mean = std::accumulate(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]),0.0)/m_elapsedTimes[i].size();
            avgElapsedTimesUs[i] = mean;
            minElapsedTimesUs[i] = *minmax_pair.first;
            maxElapsedTimesUs[i] = *minmax_pair.second;

        }

        NVLOGC_FMT(NVLOG_PUCCH,"Timing results {}, format: avg (min, max) ",
            procModeBmsk == PUCCH_PROC_MODE_FULL_SLOT ? "in stream mode" : "in graph mode");
        
        NVLOGC_FMT(NVLOG_PUCCH,"{} Pipeline[{:02d}]: Metric - GPU Time usec (using CUDA events, over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "PucchRx",
               0,
               m_elapsedTimes[0].size(),
               avgElapsedTimesUs[ELAPSED_EVT_RUN],
               minElapsedTimesUs[ELAPSED_EVT_RUN],
               maxElapsedTimesUs[ELAPSED_EVT_RUN],
               avgElapsedTimesUs[ELAPSED_EVT_SETUP],
               minElapsedTimesUs[ELAPSED_EVT_SETUP],
               maxElapsedTimesUs[ELAPSED_EVT_SETUP],
               avgElapsedTimesUs[ELAPSED_EVT_RUN] + avgElapsedTimesUs[ELAPSED_EVT_SETUP]);

        NVLOGC_FMT(NVLOG_PUCCH,"{} Pipeline[{:02d}]: Metric - CPU Time usec (using wall clock,  over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "PucchRx",
               0,
               m_elapsedTimes[0].size(),
               avgElapsedTimesUs[ELAPSED_CPU_RUN],
               minElapsedTimesUs[ELAPSED_CPU_RUN],
               maxElapsedTimesUs[ELAPSED_CPU_RUN],
               avgElapsedTimesUs[ELAPSED_CPU_SETUP],
               minElapsedTimesUs[ELAPSED_CPU_SETUP],
               maxElapsedTimesUs[ELAPSED_CPU_SETUP],
               avgElapsedTimesUs[ELAPSED_CPU_RUN] + avgElapsedTimesUs[ELAPSED_CPU_SETUP]);


        // --------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPucchRx(pucchRxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
}
