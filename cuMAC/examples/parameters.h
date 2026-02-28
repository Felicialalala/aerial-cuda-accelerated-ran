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

 #pragma once

 // debug parameters
 // #define OUTPUT_SOLUTION_
 // #define LIMIT_NUM_SM_TIME_MEASURE_
 // #define MCSCHEDULER_DEBUG_
 // #define SCSCHEDULER_DEBUG_
 // #define CHANN_INPUT_DEBUG_
 // #define CELLASSOCIATION_PRINT_SAMPLE_

 // GPU index
 #define gpuDeviceIdx          0

 // simulation duration
 #define numSimChnRlz          400
 
 // randomness
 #define seedConst          42

 // system parameters
 //#define mu                     0 // OFDM numerology: 0, 1, 2, 3, 4
 #define slotDurationConst          0.5e-3
 #define scsConst          30000.0
 #define numMcsLevels           28
 #define cellRadiusConst          500
 #define numCellConst          19
 #define numCoorCellConst          7
 #define numUePerCellConst          8
 #define numUeForGrpConst          8
 // assumption's that numUePerCellConst <= numUeForGrpConst
 #define numActiveUePerCellConst          8
 #define totNumUesConst          numCoorCellConst*numUePerCellConst
 #define totNumActiveUesConst          numCoorCellConst*numActiveUePerCellConst
 // antenna configurations
 // *AntSize, *AntSpacing, *AntPolarAngles, *AntPattern, vDirection are only used in CDL channel model and n*Ant must be equal to prod(*AntSize)
 // for other channel models, *AntSize, *AntSpacing, *AntPolarAngles, *AntPattern, vDirection are not used
 // by default, UE uses isotropic antennas, BS uses directional antennas
 #define nBsAntConst          4
 #define bsAntSizeConst         {1,2,2,1,1} // {M_g,N_g,P,M,N} 3GPP TR 38.901 Section 7.3. Assuming one panel, fix M=N=1. P can be 1 or 2
 #define bsAntSpacingConst      {0.5f, 0.5f, 1.0f, 1.0f} // BS antenna spacing w.r.t. wavelength. Assuming one panel, fix last two elements to 1.0f
 #define bsAntPolarAnglesConst  {45.0f, -45.0f} // BS antenna polarization angles
 #define bsAntPatternConst      1 // 0: isotropic; 1: 38.901
 #define nUeAntConst          4
 #define ueAntSizeConst         {2,2,1,1,1} // {M_g,N_g,P,M,N} 3GPP TR 38.901 Section 7.3. Assuming one panel, fix M=N=1. P can be 1 or 2
 #define ueAntSpacingConst      {0.5f, 0.5f, 1.0f, 1.0f} // UE antenna spacing w.r.t. wavelength. Assuming one panel, fix last two elements to 1.0f
 #define ueAntPolarAnglesConst  {0.0f, 90.0f} // UE antenna polarization angles
 #define ueAntPatternConst      0  // 0: isotropic; 1: 38.901
 #define vDirectionConst        {90, 0} // moving direction, [RxA; RxZ] — RxA and RxZ specify the azimuth and zenith of the direction of travel of the moving UE; moving speed is converted to maxDopplerShift in cdlCfg
 #define nPrbsPerGrpConst          4
 #define nPrbGrpsConst          68
 #define WConst                 12.0*scsConst*nPrbsPerGrpConst
 #define totWConst              WConst*nPrbGrpsConst
 #define PtConst                79.4328 // Macrocell - 49.0 dBm (79.4328 W), Microcell - 23 dBm (0.1995 W)
 #define PtRbgConst             PtConst/nPrbGrpsConst
 #define PtRbgAntConst          PtRbgConst/nBsAntConst
 #define bandwidthRBConst       12*scsConst
 #define bandwidthRBGConst      nPrbsPerGrpConst*bandwidthRBConst
 #define noiseFigureConst       9 // dB
 // For testing need to adjust noise variance based on channel gain
 #define sigmaSqrdDBmConst      -174 + noiseFigureConst+ 10*log10(bandwidthRBGConst)
 #define sigmaSqrdConst         pow(10.0, ((sigmaSqrdDBmConst - 30.0)/10.0))
 #define gpuAllocTypeConst          0
 #define cpuAllocTypeConst          0
 #define prdSchemeConst         0 // 0 - no precoding, 1 - SVD precoding
 #define rxSchemeConst          1 // 1 - MMSE-IRC
 #define heteroUeSelCellsConst  0 // 0 - homogeneous UE selection config. across cells, 1 - heterogeneous UE selection config. across cells
 // heterogeneous UE selection config. currently not supported for performance benchmarking vs. RR scheduler

 // max dimentions
 #define maxNumCoorCellConst    21
 #define maxNumBsAntConst       16
 #define maxNumUeAntConst       16
 #define maxNumPrbGrpsConst     100

 // buffer size
 #define estHfrSizeCOnst        nPrbGrpsConst*totNumUesConst*numCoorCellConst*nBsAntConst*nUeAntConst
 
 // PDSCH parameters
 #define pdschNrOfSymbols       12
 #define pdschNrOfDmrsSymb      1
 #define pdschNrOfDataSymb      pdschNrOfSymbols-pdschNrOfDmrsSymb
 #define pdschNrOfLayers        1

 // PF scheduling
 #define initAvgRateConst       1.0
 #define pfAvgRateUpdConst      0.001
 #define betaCoeffConst         1.0
 #define sinValThrConst         0.1
 #define prioWeightStepConst    100
 // power scaling
 #define AFTER_SCALING_SIGMA_CONST 1.0 // noise std after scaling to improve precision
 // 1.0 for 49.0 dBm BS Tx power

 #define cpuGpuPerfGapPerUeConst 0.005
 #define cpuGpuPerfGapSumRConst 0.01
 // interference control
 #define toleranceConst         0.4

 // SVD precoder parameters
 #define svdToleranceConst      1.e-7
 #define svdMaxSweeps           15

 // Normalized channel coefficients for __half range
 #define amplifyCoeConst        1

 // output file
 #define mcOutputFile           "output.txt"     
 #define mcOutputFileShort      "output_short.txt"

#define targetChanCoeRangeConst 0.1f * nPrbGrpsConst * totNumUesConst // target channel coefficients range for precision issue
#define MinNoiseRangeConst      0.001f // minimum noise figure for stability issues 

// 64TR MU-MIMO parameters
#define nMaxUeSchdPerCellTTIConst 16
