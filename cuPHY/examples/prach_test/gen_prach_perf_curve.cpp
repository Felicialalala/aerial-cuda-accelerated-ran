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

#include <cuda_runtime.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "cuphy_channels.hpp"
#include "gen_prach_perf_curve.hpp"
#include <math.h>
#include <fstream>
#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir
#include "fading_chan.cuh"
using namespace std;

struct DetectSummary {
    int false_detection_count;
    int miss_detection_count;
    int total_count;
}; 

void check_results(void * num_detectedPrmb,
                   void * prmbIndex_estimates,
                   void * prmbDelay_estimates,
                   hdf5hpp::hdf5_file & prach_file,
                   float  chanDelay,
                   float  delay_error_limit,
                   DetectSummary * detect_summary) 
{
    detect_summary->total_count += 1;

    // Copy output from GPU to CPU for reference comparison   
    std::vector<uint32_t> gpu_num_detectedPrmb(1);
    CUDA_CHECK(cudaMemcpy(gpu_num_detectedPrmb.data(), num_detectedPrmb, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    int numPrmb = gpu_num_detectedPrmb[0];

    std::vector<uint32_t> gpu_prmbIndex_estimates(numPrmb);
    std::vector<float> gpu_prmbDelay_estimates(numPrmb);

    CUDA_CHECK(cudaMemcpy(gpu_prmbIndex_estimates.data(), prmbIndex_estimates, numPrmb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_prmbDelay_estimates.data(), prmbDelay_estimates, numPrmb * sizeof(float), cudaMemcpyDeviceToHost));        
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read results from matlab test vector
    using tensor_pinned_R_32U = cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>;

    // load ground-truth info at UE
    tensor_pinned_R_32U matlab_prachFalseAlarmTest = cuphy::typed_tensor_from_dataset<CUPHY_R_32U,
        cuphy::pinned_alloc>(prach_file.open_dataset("prachFalseAlarmTest_0"));
    int matlab_prachFalseAlarmTest_val = *(matlab_prachFalseAlarmTest.addr());
    
    tensor_pinned_R_32U matlab_txPrmbIdx = cuphy::typed_tensor_from_dataset<CUPHY_R_32U,
        cuphy::pinned_alloc>(prach_file.open_dataset("UE_prmbIdx_0"));
    int matlab_txPrmbIdx_val = *(matlab_txPrmbIdx.addr());


    // the following error calculations are corresponding to 5GModel/nr_matlab/collectData.m
    if (matlab_prachFalseAlarmTest_val) {
        if (numPrmb > 0) {
            detect_summary->false_detection_count += 1;
        }
    } else {
        if (numPrmb < 1) {
            detect_summary->miss_detection_count += 1;
        } else {
            int prmbFound = 0;

            for (int rxpIdx = 0; rxpIdx < numPrmb; rxpIdx++) {
                bool txPrmbFound = false;
                if (gpu_prmbIndex_estimates[rxpIdx] == matlab_txPrmbIdx_val) {
                    txPrmbFound = true;
                    prmbFound++;
                }

                if (txPrmbFound) {
                    float timingError = abs(gpu_prmbDelay_estimates[rxpIdx] - chanDelay);
                    if (timingError > delay_error_limit) {
                        detect_summary->miss_detection_count += 1;
                    }
                //} else {
                //    detect_summary->false_detection_count += 1;
                }
            }

            if (prmbFound == 0) {
                detect_summary->miss_detection_count += 1;
            }
        }
    }
}

void gen_prach_perf_curve(std::string prachInputFilename, std::string perfOutFilename, std::vector<float> snrVec, uint32_t nItrPerSnr, uint8_t seed)
{
    // Initialize main stream
    cuphy::stream cuStrmMain;

    // Input file
    hdf5hpp::hdf5_file inputFile            = hdf5hpp::hdf5_file::open(prachInputFilename.c_str());
    
    // load parameters from H5 TV
    float TTIlen = 0.0005f; // length of each TTI, assuming numerology 1
    hdf5hpp::hdf5_dataset dset_carrier              = inputFile.open_dataset("carrier_pars");
    hdf5hpp::hdf5_dataset_elem dset_carrier_elem    = dset_carrier[0];
    uint16_t N_bsAnt                                = dset_carrier_elem["N_bsAnt"].as<uint16_t>();
    uint16_t nSubcarriers                           = dset_carrier_elem["N_sc"].as<uint16_t>();
    uint32_t N_samp_slot                            = dset_carrier_elem["N_samp_slot"].as<uint32_t>();
    uint32_t delta_f                                = 15e3 * static_cast<uint32_t>(pow(2, dset_carrier_elem["mu"].as<uint32_t>()));
    hdf5hpp::hdf5_dataset prachParams               = inputFile.open_dataset("prachParams_0");
    hdf5hpp::hdf5_dataset_elem prachParams_elem     = prachParams[0];
    uint32_t N_rep                                  = prachParams_elem["N_rep"].as<uint32_t>();
    uint32_t L_RA                                   = prachParams_elem["L_RA"].as<uint32_t>();
    uint32_t Nsamp_oran = L_RA == 139? 144 : 864;
    uint32_t K                                      = prachParams_elem["K"].as<uint32_t>();

    //************** Validate input arguments *******************
    // assume the number of UEs/PRACH occasions is 1
    // validate K value: currently only support K == 1
    assert (K == 1);

    hdf5hpp::hdf5_dataset dset_chan                 = inputFile.open_dataset("chan_pars");
    hdf5hpp::hdf5_dataset_elem dset_chan_elem       = dset_chan[0];
    // chanType: 0 - AWGN, 1 - TDL, 2 - CDL, 3 - P2P 
    uint8_t chanType                               = dset_chan_elem["chanType"].as<uint8_t>();
    float chanDelay                                = dset_chan_elem["delay"].as<float>();

    // validate chanType
    // currently only AWGN, TDL, and CDL are implemented
    assert (chanType == 0 || chanType == 1 || chanType == 2);
    uint16_t N_ueLayer = 0;
    if (chanType == 2)  // CDL, need to check antenna size
    {
        std::vector<uint8_t> ueAntSize = dset_chan_elem["ueAntSize"].as<std::vector<uint8_t>>();
        N_ueLayer = std::accumulate(ueAntSize.begin(), ueAntSize.end(), 1, std::multiplies<uint8_t>());
    }
    else
    {
        N_ueLayer = dset_chan_elem["numUeAnt"].as<uint16_t>();
    }
    // validate the number of Tx layers: currently only verified cases with a single Tx layer (major concern is the order of dimensions of the loaded Xt signal)
    assert (N_ueLayer == 1);
    // -------------------------------------------------------------------
    
    // perf curve file
    std::unique_ptr<hdf5hpp::hdf5_file> perfFile;
    bool                     perfOutputFlag = false;
    hdf5hpp::hdf5_file       perfHdf5File;
    if(!perfOutFilename.empty())
    {
        perfOutputFlag = true;
        perfFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(perfOutFilename.c_str())));
        perfHdf5File = hdf5hpp::hdf5_file::open(perfOutFilename.c_str());
    }

    // -------------------------------------------------------------------
    // Load PRACH dataset
    PrachApiDataset dataset(prachInputFilename, cuStrmMain.handle(), 0, true);
    dataset.finalize(cuStrmMain.handle());

    // initialize randomness
    cuphy::rng rng(seed, 0, cuStrmMain.handle());

    // initialize signal buffers
    cuphy::tensor_device tNoiseFreeTxSlot(CUPHY_C_16F, N_samp_slot, N_ueLayer, cuphy::tensor_flags::align_tight);
    cuphy::tensor_ref tNoisyRxSlot;
    tNoisyRxSlot.desc().set(CUPHY_C_16F, Nsamp_oran*N_rep, N_bsAnt, cuphy::tensor_flags::align_tight);
    tNoisyRxSlot.set_addr( dataset.dataRxTensor[0].addr());

    // initialize fading channel
    uint8_t phychanType = 2;
    fadingChan<__half2>* fadeChanPtr = new fadingChan<__half2>(static_cast<__half2*>(tNoiseFreeTxSlot.addr()), static_cast<__half2*>(tNoisyRxSlot.addr()), cuStrmMain.handle(), chanType, static_cast<uint16_t>(seed), phychanType);
    fadeChanPtr->setup(inputFile, 1/*enableSwapTxRx*/);

    // Allocate PRACH handle
    std::unique_ptr<cuphyPrachRxHndl_t> prach_handle = std::make_unique<cuphyPrachRxHndl_t>();

    // Create receiver
    cuphyStatus_t status = cuphyCreatePrachRx(prach_handle.get(),  &(dataset.prachStatPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphyCreatePrachRx(): {}", cuphyGetErrorString(status));
        exit(1);
    }
    cudaStreamSynchronize(cuStrmMain.handle()); 

    // determine threshold for timing error. Refer to 5GModel/nr_matlab/collectData.m, PRACH section
    float timingErrorThreshold;

    if (chanType == 0) { // AWGN
        if (L_RA == 839) {
            timingErrorThreshold = 1.04e-6;
        } else if (delta_f == 15e3) {
            timingErrorThreshold = 0.52e-6;
        } else {
            timingErrorThreshold = 0.26e-6;
        }     
    } else if (chanType == 1 || chanType == 2) { // TDL, CDL
        if (L_RA == 839) {
            timingErrorThreshold = 2.55e-6;
        } else if (delta_f == 15e3) {
            timingErrorThreshold = 2.03e-6;
        } else {
            timingErrorThreshold = 1.77e-6;
        }
    }

    // detection result data structure
    DetectSummary detect_summary;

    // initialize perf curves
    int nSnrSteps = snrVec.size();
    cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tSnrCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumMissCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumFalseCpu(nSnrSteps);
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tNumTotCntCpu(nSnrSteps);

    for (int snrStepIdx = 0; snrStepIdx < nSnrSteps; snrStepIdx++)
    {
        tNumMissCpu(snrStepIdx)              = 0;
        tNumFalseCpu(snrStepIdx)             = 0;
        tNumTotCntCpu(snrStepIdx)            = 0;
        float snr                            = snrVec[snrStepIdx];

        detect_summary.miss_detection_count  = 0;
        detect_summary.false_detection_count = 0;
        detect_summary.total_count           = 0;

        for (int itrIdx = 0; itrIdx < nItrPerSnr; ++itrIdx)
        {
            // channel fading/noise
            fadeChanPtr -> run(TTIlen * itrIdx, snr, 1/*enableSwapTxRx*/);
            cudaStreamSynchronize(cuStrmMain.handle()); 

            // setup PRACH receiver
            status = cuphySetupPrachRx(*prach_handle, &(dataset.prachDynPrms));
            if(status != CUPHY_STATUS_SUCCESS)
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphySetupPrachRx(): {}", cuphyGetErrorString(status));
                exit(1);
            }
            cudaStreamSynchronize(cuStrmMain.handle());

            // run PRACH receiver
            status = cuphyRunPrachRx(*prach_handle);
            if (status != CUPHY_STATUS_SUCCESS) 
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphyRunPrachRx(): {}", cuphyGetErrorString(status));
                exit(1);
            }
            cudaStreamSynchronize(cuStrmMain.handle());

            // check detection results
            check_results(dataset.num_detectedPrmb.addr(),
                        dataset.prmbIndex_estimates.addr(), dataset.prmbDelay_estimates.addr(), dataset.prach_file,
                        chanDelay, timingErrorThreshold, &detect_summary); 
            cudaStreamSynchronize(cuStrmMain.handle());
        }
        tSnrCpu(snrStepIdx) = snr;
        tNumMissCpu(snrStepIdx)        = detect_summary.miss_detection_count;
        tNumFalseCpu(snrStepIdx)       = detect_summary.false_detection_count;
        tNumTotCntCpu(snrStepIdx)      = detect_summary.total_count;
    }

    // clean up
    // destroy PRACH receiver
    cuphyDestroyPrachRx(*prach_handle);

    if (fadeChanPtr) {
        delete fadeChanPtr;
    }

    // -----------------------------------------------------------------
    // Save tber to H5
    if(perfOutputFlag)
    {
        cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> tSnrGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumMissGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumFalseGpu(nSnrSteps);
        cuphy::typed_tensor<CUPHY_R_32U, cuphy::device_alloc> tNumTotCntGpu(nSnrSteps);

        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tSnrGpu.addr()) , static_cast<void*>(tSnrCpu.addr()) , nSnrSteps * sizeof(float), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumMissGpu.addr()), static_cast<void*>(tNumMissCpu.addr()), nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumFalseGpu.addr()), static_cast<void*>(tNumFalseCpu.addr()), nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(tNumTotCntGpu.addr()), static_cast<void*>(tNumTotCntCpu.addr()), nSnrSteps * sizeof(uint32_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        cuphy::write_HDF5_dataset(perfHdf5File, tSnrGpu,            "snr", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(perfHdf5File, tNumMissGpu,        "numMiss", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(perfHdf5File, tNumFalseGpu,       "numFalseDet", cuStrmMain.handle());
        cuphy::write_HDF5_dataset(perfHdf5File, tNumTotCntGpu,      "numTotCnt", cuStrmMain.handle());
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        perfHdf5File.close();
    }

}