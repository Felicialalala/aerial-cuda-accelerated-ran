# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
        
cubbGpuTestbenchTvNames100MHz = [
    # F08 100 MHz avg
    "TV_cuphy_V08-DS-03_slot0_MIMO4x4_PRB22_DataSyms12_qam256.h5",
    "TV_cuphy_U08-US-03_snrdb40.00_MIMO2x4_PRB22_DataSyms13_qam256.h5",
    "TV_cuphy_F08-RA-01.h5",
    "TV_cuphy_F08-DC-40_PRB273.h5",
    "TV_cuphy_F08-UC-RC_PRB273.h5",
    "TV_cuphy_F08-SS-01.h5",
    "TV_cuphy_F08-CR-01.h5",
    # F08 100 MHz peak PDSCH, PUSCH; other channels are same
    "TV_cuphy_V08-DS-02_slot0_MIMO4x4_PRB45_DataSyms12_qam256.h5",
    "TV_cuphy_U08-US-02_snrdb40.00_MIMO2x4_PRB45_DataSyms13_qam256.h5",
    

    # F09 100 MHz avg
    "TV_cuphy_V09-DS-41_slot0_MIMO8x8_PRB34_DataSyms10_qam256.h5",
    "TV_cuphy_U09-US-41_snrdb40.00_MIMO4x4_PRB34_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F09-BW-02_slot0_MIMO8x8_PRB136.h5",
    "srs2_TV_cuphy_F09-SR-01_snrdb40.00_MIMO4x8_PRB272.h5",
    "TV_cuphy_F09-RA-01.h5",
    "TV_cuphy_F09-SS-01.h5",
    "TV_cuphy_F09-DC-02_PRB136.h5",
    "TV_cuphy_F09-UC-40_PRB273.h5",
    "TV_cuphy_F09-CR-01.h5",
    # F09 100 MHz peak PDSCH, PUSCH, BWC; other channels are 
    "TV_cuphy_V09-DS-40_slot0_MIMO8x8_PRB68_DataSyms10_qam256.h5",
    "TV_cuphy_U09-US-40_snrdb40.00_MIMO4x4_PRB68_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F09-BW-01_slot0_MIMO8x8_PRB273.h5",
    "TV_cuphy_F09-DC-01_PRB273.h5",

    # F14 100 MHz avg
    "TV_cuphy_V14-DS-41_slot0_MIMO16x16_PRB68_DataSyms11_qam256.h5",
    "TV_cuphy_U14-US-41_snrdb40.00_MIMO8x16_PRB68_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F14-BW-02_slot0_MIMO16x16_PRB136.h5",
    "srs2_TV_cuphy_F14-SR-01_snrdb40.00_MIMO8x16_PRB272.h5",
    "TV_cuphy_F14-RA-40.h5",
    "TV_cuphy_F14-SS-02.h5",
    "TV_cuphy_F14-DC-40_PRB273.h5",
    "TV_cuphy_F14-UC-40_PRB273.h5","TV_cuphy_F14-CR-01.h5",
    # F14 100 MHz peak PDSCH, PUSCH, BWC; other channels are 
    "TV_cuphy_V14-DS-40_slot0_MIMO16x16_PRB136_DataSyms11_qam256.h5",
    "TV_cuphy_U14-US-40_snrdb40.00_MIMO8x16_PRB136_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F14-BW-01_slot0_MIMO16x16_PRB273.h5"
]

cubbGpuTestbenchTvNames20MHz = [
    # F08 20MHz avg
    "TV_cuphy_V08-DS-43_slot0_MIMO4x4_PRB4_DataSyms12_qam256.h5",
    "TV_cuphy_U08-US-43_snrdb40.00_MIMO2x4_PRB4_DataSyms13_qam256.h5",
    "TV_cuphy_F08-RA-01.h5",
    "TV_cuphy_F08-DC-40_PRB273.h5",
    "TV_cuphy_F08-UC-RC_PRB273.h5",
    "TV_cuphy_F08-SS-01.h5",
    "TV_cuphy_F08-CR-02.h5",  
    # F08 20MHz peak PDSCH, PUSCH; other channels are same
    "TV_cuphy_V08-DS-42_slot0_MIMO4x4_PRB8_DataSyms12_qam256.h5",
    "TV_cuphy_U08-US-42_snrdb40.00_MIMO2x4_PRB8_DataSyms13_qam256.h5",
    # F09 20MHz avg
    "TV_cuphy_V09-DS-43_slot0_MIMO8x8_PRB6_DataSyms10_qam256.h5",
    "TV_cuphy_U09-US-43_snrdb40.00_MIMO4x4_PRB6_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F09-BW-04_slot0_MIMO8x8_PRB24.h5",
    "srs2_TV_cuphy_F09-SR-02_snrdb40.00_MIMO8x16_PRB48.h5",
    "TV_cuphy_F09-RA-01.h5",
    "TV_cuphy_F09-SS-01.h5",
    "TV_cuphy_F09-DC-02_PRB136.h5",
    "TV_cuphy_F09-UC-40_PRB273.h5",
    "TV_cuphy_F09-CR-02_PRB48.h5",
    # F09 20MHz peak PDSCH, PUSCH, BWC; other channels are 
    "TV_cuphy_V09-DS-42_slot0_MIMO8x8_PRB12_DataSyms10_qam256.h5",
    "TV_cuphy_U09-US-42_snrdb40.00_MIMO4x4_PRB12_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F09-BW-03_slot0_MIMO8x8_PRB48.h5",

    # F14 20MHz avg
    "TV_cuphy_V14-DS-43_slot0_MIMO16x16_PRB12_DataSyms10_qam256.h5",
    "TV_cuphy_U14-US-43_snrdb40.00_MIMO8x16_PRB12_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F14-BW-04_slot0_MIMO16x16_PRB25.h5",
    "srs2_TV_cuphy_F14-SR-42_snrdb40.00_MIMO8x16_PRB48.h5",
    "TV_cuphy_F14-RA-40.h5",
    "TV_cuphy_F14-SS-02.h5",
    "TV_cuphy_F14-DC-42_PRB273.h5",
    "TV_cuphy_F14-UC-40_PRB273.h5","TV_cuphy_F14-CR-42.h5",
    # F14 20MHz peak PDSCH, PUSCH, BWC; other channels are 
    "TV_cuphy_V14-DS-42_slot0_MIMO16x16_PRB25_DataSyms10_qam256.h5",
    "TV_cuphy_U14-US-42_snrdb40.00_MIMO8x16_PRB25_DataSyms12_qam256.h5",
    "bfw2_TV_cuphy_F14-BW-03_slot0_MIMO16x16_PRB51.h5"
]

def copyTvSet(tvSource, tvDestination, tvNames):
    for tv in tvNames:
        print(f'copying {tv}')
        os.system(f'cp {tvSource}/{tv} {tvDestination}/')
        
# copy TVs from source dir to target dir with given TV names
print("IMPORTANT INFO: please change the tvSource, tvDestination, tvNames as needed")
# copy 100 MHz perf TVs
copyTvSet("/mnt/cicd_tvs/develop/GPU_test_input", "../TVs/GPU_test_input", cubbGpuTestbenchTvNames100MHz)
# copy 20 MHz perf TVs
copyTvSet("/mnt/cicd_tvs/develop/GPU_test_input", "../TVs/GPU_test_input", cubbGpuTestbenchTvNames20MHz)