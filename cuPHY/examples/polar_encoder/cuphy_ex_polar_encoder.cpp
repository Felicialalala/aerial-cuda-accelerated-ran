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

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include "cuda_profiler_api.h"
#include "cuphy.h"
#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "CLI/CLI.hpp"

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

using namespace cuphy;

int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("polar_encoder.log");
    try
    {
        CLI::App app{"Polar Encoder Example"};
        
        // Command line options
        std::string inputFilename;
        std::string outputFilename;
        uint32_t nInfoBits{0};
        uint32_t nTxBits{0};
        bool enNvprof{false};
        uint32_t verboseMode{1};
        uint32_t nIter{1000};

        // Add options
        app.add_option("-i,--input", inputFilename, "Input HDF5 filename")->required();
        app.add_option("-o,--output", outputFilename, "Output HDF5 filename for pipeline tensors");
        app.add_option("-r,--iterations", nIter, "Number of iterations to run")
           ->check(CLI::PositiveNumber);
        app.add_option("--I", nInfoBits, "Number of info bits")
           ->check(CLI::Range(1, CUPHY_POLAR_ENC_MAX_INFO_BITS));
        app.add_option("--T", nTxBits, "Number of transmit bits")
           ->check(CLI::Range(1, CUPHY_POLAR_ENC_MAX_TX_BITS));
        app.add_option("--V", verboseMode, "Verbose logging level")
           ->check(CLI::Range(0, 2));
        app.add_flag("--P", enNvprof, "Enable nvprof (sets iterations to 1)");

        // Parse options
        CLI11_PARSE(app, argc, argv);

        // Post-parse processing
        if(enNvprof) {
            nIter = 1;
        }

        cudaStream_t cuStream;
        cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);

        bool verboseErrsOnly = false;
        bool verbose = false;
        if(1 == verboseMode) verboseErrsOnly = true;
        if(2 == verboseMode) { verboseErrsOnly = true; verbose = true; }

        uint8_t procModeBmsk = 0; // Downlink

        cudaEvent_t eStart, eStop;
        CUDA_CHECK(cudaEventCreateWithFlags(&eStart, cudaEventBlockingSync));
        CUDA_CHECK(cudaEventCreateWithFlags(&eStop, cudaEventBlockingSync));

        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());
        using tensor_pinned_R_8U  = typed_tensor<CUPHY_R_8U, pinned_alloc>;

        cuphy::tensor_device tGpuInfoBits  = cuphy::tensor_from_dataset(fInput.open_dataset("InfoBits"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStream);
        tensor_pinned_R_8U   tCpuCodedBits = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("CodedBits"), cuphy::tensor_flags::align_tight, cuStream);
        tensor_pinned_R_8U   tCpuTxBits    = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("TxBits"), cuphy::tensor_flags::align_tight, cuStream);

        uint32_t nExpectedCodedBits = 0;

        cuphy::disable_hdf5_error_print(); // Disable HDF5 stderr printing
        try
        {
            cuphy::cuphyHDF5_struct encPrms = cuphy::get_HDF5_struct(fInput, "encPrms");
            nInfoBits                       = encPrms.get_value_as<uint32_t>("nInfoBits");
            nExpectedCodedBits              = encPrms.get_value_as<uint32_t>("nCodedBits");
            nTxBits                         = encPrms.get_value_as<uint32_t>("nTxBits");
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have an encPrms struct.

            nExpectedCodedBits = round_up_to_next(nInfoBits, 32U);
        }
        cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        // Allocate output tensors
        // For coded bits provide the worst case storage
        cuphy::tensor_device tGpuCodedBits(CUPHY_R_8U,
                                           div_round_up(CUPHY_POLAR_ENC_MAX_CODED_BITS, 8),
                                           cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tGpuTxBits(CUPHY_R_8U,
                                        round_up_to_next(CUPHY_POLAR_ENC_MAX_TX_BITS, 32) / 8, // roundup to nearest 32b boundary (multiple of words)
                                        cuphy::tensor_flags::align_tight);

        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize(); // Needed because typed_tensor does not support non-default streams

        //------------------------------------------------------------------
        // Run the test
        if(enNvprof) cudaProfilerStart();

        TimePoint startTime = Clock::now();
        CUDA_CHECK(cudaEventRecord(eStart, cuStream));
        uint32_t nCodedBits = 0;

        for(uint32_t i = 0; i < nIter; ++i)
        {
            cuphyStatus_t polarEncStat = cuphyPolarEncRateMatch(nInfoBits,
                                                                nTxBits,
                                                                static_cast<uint8_t const*>(tGpuInfoBits.addr()),
                                                                &nCodedBits,
                                                                static_cast<uint8_t*>(tGpuCodedBits.addr()),
                                                                static_cast<uint8_t*>(tGpuTxBits.addr()),
                                                                procModeBmsk,
                                                                cuStream);
            if(CUPHY_STATUS_SUCCESS != polarEncStat) throw cuphy::cuphy_exception(polarEncStat);
        }

        CUDA_CHECK(cudaEventRecord(eStop, cuStream));
        CUDA_CHECK(cudaEventSynchronize(eStop));

        cudaStreamSynchronize(cuStream);

        TimePoint stopTime = Clock::now();

        if(enNvprof) cudaProfilerStop();

        //------------------------------------------------------------------
        // Display execution times
        float elapsedMs = 0.0f;
        cudaEventElapsedTime(&elapsedMs, eStart, eStop);

        printf("Execution time: Polar encoding + Rate matching \n");
        printf("---------------------------------------------------------------\n");
        printf("Average (over %d runs) elapsed time in usec (CUDA event) = %.0f\n",
               nIter,
               elapsedMs * 1000 / nIter);

        duration<float, std::milli> diff = stopTime - startTime;
        printf("Average (over %d runs) elapsed time in usec (wall clock) w/ 1s delay kernel = %.0f\n",
               nIter,
               diff.count() * 1000 / nIter);

        //------------------------------------------------------------------
        // Verify results
        // Coded bits are always a multiple of 32
        tensor_pinned_R_8U tCpuCpyCodedBits(tGpuCodedBits.layout(), cuphy::tensor_flags::align_tight);
        // typed_tensor<CUPHY_BIT, pinned_alloc> tCpuCpyCodedBits(tGpuCodedBits.layout(), cuphy::tensor_flags::align_tight);
#if 1     
        tCpuCpyCodedBits = tGpuCodedBits;
#endif
        tensor_pinned_R_8U tCpuCpyTxBits(tGpuTxBits.layout(), cuphy::tensor_flags::align_tight);
        // typed_tensor<CUPHY_BIT, pinned_alloc> tCpuCpyInfoBits(tGpuInfoBits.layout(), cuphy::tensor_flags::align_tight);
#if 1    
        tCpuCpyTxBits = tGpuTxBits;
#endif
        // Wait for copy to complete
        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize(); // Needed becase typed_tensor does not support non-default streams

        // Compare expected vs observed
        printf("nInfoBits: %d nExpectedCodedBits: %d nComputedCodedBits: %d nTxBits: %d\n", nInfoBits, nExpectedCodedBits, nCodedBits, nTxBits);
        // Coded bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing coded bits\n");
        uint32_t nCodedByteErrs = 0;
        uint32_t nCodedBytes    = nExpectedCodedBits / 8; // nExpectedCodedBits is a multiple of 32
        for(int n = 0; n < nCodedBytes; ++n)
        {
            uint32_t expectedCodedByte = tCpuCodedBits({n});
            uint32_t observedCodedByte = tCpuCpyCodedBits({n});
            if(expectedCodedByte != observedCodedByte)
            {
                if(verboseErrsOnly) printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
                nCodedByteErrs++;
            }
        }

        if(0 == nCodedByteErrs)
        {
            printf("No errors detected in coded bits\n");
        }
        else
        {
            if(! verboseErrsOnly) printf("Errors detected in coded bits\n");
        }        

        // Transmit bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing transmit bits\n");
        uint32_t nTxByteErrs = 0;
        uint32_t nTxBytes    = (nTxBits + 7) / 8; // nTxBits is not a multiple of 8, needs rounding
        for(int n = 0; n < nTxBytes; ++n)
        {
            uint32_t expectedTxByte = tCpuTxBits({n});
            uint32_t observedTxByte = tCpuCpyTxBits({n});
            if(expectedTxByte != observedTxByte)
            {
                if(verboseErrsOnly) printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
                nTxByteErrs++;
            }
        }

        if(0 == nTxByteErrs)
        {
            printf("No errors detected in transmit bits\n");
        }
        else
        {
            if(! verboseErrsOnly) printf("Errors detected in transmit bits\n");
        }

        if(verbose)
        {
            // Coded bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping coded bits (formatted as %d bytes)\n", nCodedBytes);
            for(int n = 0; n < nCodedBytes; ++n)
            {
                uint32_t expectedCodedByte = tCpuCodedBits({n});
                uint32_t observedCodedByte = tCpuCpyCodedBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
            }

            // Transmit bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping transmit bits (formatted as %d bytes)\n", nTxBytes);
            for(int n = 0; n < nTxBytes; ++n)
            {
                uint32_t expectedTxByte = tCpuTxBits({n});
                uint32_t observedTxByte = tCpuCpyTxBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
            }
        }

        //------------------------------------------------------------------
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(eStart));
        CUDA_CHECK(cudaEventDestroy(eStop));

        cudaStreamSynchronize(cuStream);

        cudaDeviceSynchronize();
        cudaStreamDestroy(cuStream);
        
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}

