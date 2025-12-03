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

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <numeric>
#include <algorithm>
#include <cctype>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "util.hpp"
#include "test_config.hpp"
#include "datasets.hpp"
#include "cuphy_channels.hpp"
#include "CLI/CLI.hpp"

#include <chrono>
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("bfw.log");
    try
    {
        // Parse command line arguments using CLI11
        std::string inputFileName;
        std::string outputFileName;
        int32_t nIterations = 1000;
        int32_t delayMs = 5;
        int gpuId = 0;
        bool enableOutputFileLog = false;
        int nCells = 0;
        uint32_t nSlots = 1;
        float refCheckSnrThd = 30.0f;
        float inRefCheckSnrThd = 0.0f;
        bool enableRefChecks = false;
        OutputBufferMode outputBufferMode = OutputBufferMode::Device;
        bool useHostBuffer = true;
        int beamIdStart = -1;
        uint64_t procModeBmsk = 0;
        std::string copyMode = "batched"; // D2H copy of BFW weights: "batched" memcpy (default) or "kernel" copy

        CLI::App app{"cuphy_ex_bfc - BFW Example"};
        app.add_option("-i,--input", inputFileName, "Input HDF5 or yaml filename")->required();
        app.add_option("-o,--output", outputFileName, "Write pipeline tensors to an HDF5 output file")->capture_default_str();
        app.add_option("-r,--iterations", nIterations, "Number of run iterations to run (default = 1000)")->default_val(1000);
        app.add_option("-d,--delay", delayMs, "CPU latency hiding delay in milliseconds (default: 5)")->default_val(5);
        app.add_option("-m,--mode", procModeBmsk, "PUSCH proc mode: streams(0x0), graphs (0x1) (default = 0x0)")->default_val(0x0);
        app.add_option("-b,--beamIdStart", beamIdStart, "Starting offset for calculating beam IDs, negative value means no beam ID (default: -1)")->default_val(-1);
        app.add_option("-H,--host-buff", useHostBuffer, "Host output buffer (default: true)")->default_val(true);
        app.add_option("-c,--refcheck", refCheckSnrThd, "Enable reference check with optional SNR threshold (default: 30dB)")->expected(0, 1);
        app.add_option("--gpu-id", gpuId, "GPU ID (default: 0)")->default_val(0);
        app.add_option("--copy-mode", copyMode, "D2H copy of BFW weights: 'batched' for batched memcpy (default), 'kernel' for kernel-based copy")->check(CLI::IsMember({"batched", "kernel"}, CLI::ignore_case))->default_val("batched");
        CLI11_PARSE(app, argc, argv);
        // If refCheckSnrThd was set, enableRefChecks should be true
        if(app.count("-c") || app.count("--refcheck")) {
            enableRefChecks = true;
        }
        // If outputFileName is set, enableOutputFileLog should be true
        if(!outputFileName.empty()) {
            enableOutputFileLog = true;
        }

        //-------------------------------------------------------------------------------
        // Parse inputs
        std::vector<std::vector<std::string>> inFileNamesBfc; 
        std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        NVLOGC_FMT(NVLOG_BFW, "File extension: {}", inFileExtn);
        if(inFileExtn == "yaml")
        {
            cuphy::test_config testCfg(inputFileName.c_str());
            testCfg.print();
            nCells = testCfg.num_cells();
            nSlots = testCfg.num_slots();
            const std:: string bfcChannelName = "BFC";
            
            std::vector<std::string> bfcHdf5Filenames;
    
            inFileNamesBfc.resize(nSlots);
            for (int iSlot = 0; iSlot < nSlots; iSlot++) 
            {
                inFileNamesBfc[iSlot].resize(nCells);
                for (int iCell = 0; iCell < nCells; iCell += 1) 
                {
                    std::string bfcTvFilename = testCfg.slots()[iSlot].at(bfcChannelName)[iCell];        
                    inFileNamesBfc[iSlot][iCell] = bfcTvFilename;
                }
            }
            
        }
        else
        {
            // Only single slot, single cell supported in vanilla HDF5 mode
            nCells = 1;
            inFileNamesBfc.resize(nCells);
            inFileNamesBfc[0].emplace_back(inputFileName);            
        }

        if(useHostBuffer)
        {
            outputBufferMode = OutputBufferMode::Host;
        }

        //-------------------------------------------------------------------------------

        // Stream for workload submission
        cuphy::stream cuphyStrm(cudaStreamNonBlocking);

        // Events for syncrhonization across streams
        std::vector<float> elapsedTimeUsCuphyEvts(nIterations, 0.0f);
        std::vector<cuphy::event_timer> cuphyEvtTimers(nIterations);
    
        //-------------------------------------------------------------------------------
        // Pipeline creation and setup
        // Note: test bench consumes the lambda value from the first file. lambda value assumed to be same across all TVs 
        bfwStaticApiDataset staticApiDataset(inFileNamesBfc[0], cuphyStrm.handle(), outputFileName);

        // Configure D2H copy mode of BFW weights in static parameters before pipeline creation
        {
            std::string modeLower = copyMode;
            std::transform(modeLower.begin(), modeLower.end(), modeLower.begin(),
                           [](unsigned char character){ return static_cast<char>(std::tolower(character)); });
            const bool useKernelCopy = (modeLower == "kernel");
            staticApiDataset.bfwStatPrms.useKernelCopy = useKernelCopy ? 1 : 0;
            // If using kernel copy, disable batched memcpy; otherwise enable it
            staticApiDataset.bfwStatPrms.enableBatchedMemcpy = useKernelCopy ? 0 : 1;
            NVLOGC_FMT(NVLOG_BFW, "BFW weights D2H copy mode: {}", useKernelCopy ? "kernel" : "batched");
        }

#if 1  
        //------------------------------------------------------------------
        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(enableOutputFileLog)
        {
            std::string inFileName = inFileNamesBfc[0][0];//[cellIdx];

            // If possible use input file name as suffix for output
            size_t pos = inFileName.rfind('/', inFileName.length());
            if(pos != std::string::npos)
            {
//                outputFileName = "gpu_out_cell_" + std::to_string(cellIdx) + "_" + inFileName.substr(pos + 1, inFileName.length() - pos);
                printf("outputFileName: %s\n", outputFileName.c_str());
            }
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFileName.c_str())));
    
            // Write channel equalizer outputs
//            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef, "Coef");
//            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
            
            // Wait for writes to complete
//            cudaDeviceSynchronize();
        }
#endif        

        cuphyStrm.synchronize();

        std::vector<bfwDynApiDataset> dynApiDatasets; 
        std::vector<bfwEvalDataset> evalDatasets;
        std::vector<cuphy::bfw_tx> bfwTxPipelines;

        dynApiDatasets.reserve(nSlots);
        evalDatasets.reserve(nSlots);
        bfwTxPipelines.reserve(nSlots);
        for(int32_t iSlot = 0; iSlot < nSlots; ++iSlot)
        {
            dynApiDatasets.emplace_back(inFileNamesBfc[iSlot], cuphyStrm.handle(), procModeBmsk, outputBufferMode, beamIdStart);
            evalDatasets.emplace_back(inFileNamesBfc[iSlot], cuphyStrm.handle());
            bfwTxPipelines.emplace_back(staticApiDataset.bfwStatPrms, cuphyStrm.handle());
            bfwTxPipelines[iSlot].setup(dynApiDatasets[iSlot].bfwDynPrms);
        }
        cuphyStrm.synchronize();

#if 0
        //------------------------------------------------------------------
        // 1. Vectors above invoke tensor copy constructor, tensor copy constructor's invocation of convert function uses default stream        
        // 2. Ensure all prior work on the GPU is completed before launching delay kernel (free up the internal FIFOs to accomodate as much
        // of the workload burst that follows the delay kernel)        
        cudaDeviceSynchronize();
#endif

        //-------------------------------------------------------------------------------
        // Execute pipeline
        for(int32_t iSlot = 0; iSlot < nSlots; ++iSlot)
        {
            // Insert short delay kernel
            gpu_ms_delay(delayMs, gpuId, cuphyStrm.handle());

            auto startWallClock = Clock::now();
            for(int32_t i = 0; i < nIterations; ++i)
            {
                // run launches on the same CUDA stream setup uses
                cuphyEvtTimers[i].record_begin(cuphyStrm.handle());
                bfwTxPipelines[iSlot].run(procModeBmsk);
                cuphyEvtTimers[i].record_end(cuphyStrm.handle());
            }

            // Wait for work to complete
            float totalElapsedTimeUsCuphyEvt = 0.0f;
            for(int32_t i = 0; i < nIterations; ++i)
            {
                cuphyEvtTimers[i].synchronize();
                elapsedTimeUsCuphyEvts[i] = cuphyEvtTimers[i].elapsed_time_ms()*1000;
                totalElapsedTimeUsCuphyEvt += elapsedTimeUsCuphyEvts[i];
            }
            cuphyStrm.synchronize();
            auto stopWallClock = Clock::now();
            duration<float, std::micro> diff = stopWallClock - startWallClock;            
            float elapsedTimeUsWallClk = diff.count();

            NVLOGC_FMT(NVLOG_BFW, "---------------------------------------------------------------");
            NVLOGC_FMT(NVLOG_BFW, "Slot[{}]: Average ({} runs) elapsed time in usec (CUDA event w/ {} ms delay kernel) = {:07.4f}",
                   iSlot,
                   nIterations,
                   delayMs,
                   totalElapsedTimeUsCuphyEvt / nIterations);
            //NVLOGC_FMT(NVLOG_BFW, "Average elapsed time wall clock {:07.4f}", elapsedTimeUsWallClk/nIterations); 

            //-------------------------------------------------------------------------------
            // Evaluate results
            if(enableRefChecks)
            {
                evalDatasets[iSlot].bfwEvalCoefs(staticApiDataset, dynApiDatasets[iSlot], cuphyStrm.handle(), refCheckSnrThd, enableRefChecks);
            }
            bfwTxPipelines[0].writeDbgSynch();
            cuphyStrm.synchronize();
            NVLOGC_FMT(NVLOG_BFW, "---------------------------------------------------------------");
        }

#if 0    
        //------------------------------------------------------------------
        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(enableOutputFileLog)
        {
            std::string inFileName = inFileNamesBfc[cellIdx];

            // If possible use input file name as suffix for output
            size_t pos = inFileName.rfind('/', inFileName.length());
            if(pos != std::string::npos)
            {
                outputFileName = "gpu_out_cell_" + std::to_string(cellIdx) + "_" + inFileName.substr(pos + 1, inFileName.length() - pos);
                NVLOGC_FMT(NVLOG_BFW, "outputFileName: {}", outputFileName);
            }
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFileName.c_str())));
    
            // Write channel equalizer outputs
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef, "Coef");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
            
            // Wait for writes to complete
            cudaDeviceSynchronize();
        }
#endif        
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
