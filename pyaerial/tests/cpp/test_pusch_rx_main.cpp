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

#include <iostream>
#include <string>
#include <vector>
#include "test_pusch_rx.hpp"
#include "pycuphy_params.hpp"
#include "fmtlog.h"
#include "datasets.hpp"


////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("test_pusch_rx_main [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -i  input_filenames    Input filename\n");
    printf("    -l  log_filename       filename to save log output\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    // Number of GPUs
    int32_t nGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));

    // Read arguments
    int iArg = 1;
    std::vector<std::string> inputFileNames;
    int32_t gpuId = 0;


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
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileNames.push_back(std::string(argv[iArg++]));
                break;
            case 'g':
                if((++iArg >= argc) ||
                    (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                    ((gpuId < 0) || (gpuId >= nGPUs)))
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid GPU Id (should be within [0,{}])", nGPUs - 1);
                    exit(1);
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else // if('-' == argv[iArg][0])
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
            exit(1);
        }
    } // while (iArg < argc)

    if(inputFileNames.empty())
    {
        usage();
        exit(1);
    }

    cudaStream_t cuStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Read the input file
    StaticApiDataset staticApiDataset(inputFileNames, cuStream);
    DynApiDataset dynApiDataset(inputFileNames, cuStream);

    dynApiDataset.puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;

    // Set parameters.
    pycuphy::PuschParams puschParams;
    puschParams.setStatPrms(staticApiDataset.puschStatPrms);
    puschParams.setDynPrms(dynApiDataset.puschDynPrm);

    // Create and run the test
    std::string errMsg = "\033[1;32mPASSED\033[0m";
    pycuphy::TestPuschRxPipeline test_pipeline(puschParams, cuStream);
    bool status = test_pipeline.runTest(puschParams, errMsg);
    std::cout << inputFileNames[0] << ": ";
    if(status)
        std::cout << errMsg << std::endl;
    else {
        std::cout << "\033[1;31mFAILED\033[0m: " << errMsg << std::endl;
        return 1;
    }
    return 0;
}