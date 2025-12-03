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
#include <chrono>
#include <numeric>
#include "utils.hpp"

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;


//static constexpr uint64_t bitmasku16x2 = 0x1FF000001FF;
//static constexpr __uint128_t bitmasku32x4 = 0x1FF000001FF000001FF000001FF;
//static constexpr uint16_t bitmask9b = 0x1FF;


// data contains 8 9-bit packed values

void usage(char* arg)
{
	std::cout << "Test bundle comparison" << std::endl;
    std::cout << arg << ":\n";

}

int main(int argc, char* argv[])
{
	constexpr int L_TRX = 64;
	constexpr int compressBitWidth = 9;
	constexpr int bundleSize = L_TRX * compressBitWidth * 2 / 8;
	constexpr int beta = 2048;
	TimePoint timePtBundleStart, timePtBundleStop;
	duration<float, std::micro> elpasedTimeDurationUs;
	std::vector<float> decompressCompareRes;
	std::vector<float> bundleCompareRes;
	bool verbose = false;
	int numTrials = 1;
	int c;
    while ((c = getopt(argc, argv, "hr:v")) != -1) 
    {
        switch(c) 
        {
            case 'r':
                numTrials = atoi(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
            default:
                usage(argv[0]);
                return -1;
        }
    }

	std::cout << "Running for " << numTrials << " iterations" << std::endl;
	decompressCompareRes.resize(numTrials);
	bundleCompareRes.resize(numTrials);
	uint8_t rx_buff[] = {2,
						 0xC3, 0x7D, 0x8C, 0x44, 0x93, 0x89, 0x0B, 0x3D, 0xAE, 0x09,
	                     0xDF, 0xC7, 0x18, 0xDD, 0x06, 0xB5, 0x05, 0xE3, 0xFC, 0x1E,
						 //0xDF, 0xC7, 0x18, 0xDD, 0x06, 0xB1, 0x05, 0xE3, 0xFC, 0x1E,
						 0xB1, 0xBC, 0x93, 0x87, 0x24, 0x38, 0x75, 0x2B, 0xE8, 0xF8,
						 0xC6, 0xD3, 0x56, 0xC7, 0x19, 0xCB, 0x31, 0xEE, 0x3C, 0x18,
						 0x1D, 0x2E, 0xA3, 0x0B, 0xDB, 0xDF, 0x9B, 0xF1, 0x7D, 0x13,
						 0xE7, 0x64, 0x71, 0x8C, 0x35, 0x10, 0x4B, 0x5A, 0x3D, 0x29,
						 0x60, 0x0F, 0x86, 0x15, 0x9D, 0xFF, 0x57, 0xD3, 0xC8, 0x88,
						 0xE1, 0xC6, 0x2C, 0x97, 0x85, 0x98, 0x60, 0x6E, 0x10, 0x6D,
						 0x8D, 0x1F, 0xE3, 0x10, 0x3E, 0xBD, 0xEE, 0x3B, 0x92, 0x76,
						 0x3C, 0x07, 0x0D, 0x9B, 0x9E, 0xE6, 0x18, 0xF7, 0xDA, 0xD1,
						 0xE9, 0xC0, 0x88, 0x50, 0x89, 0xFF, 0x78, 0x6B, 0xDB, 0xE6,
						 0x8E, 0x5C, 0x8F, 0xCE, 0x0B, 0x48, 0x5C, 0x00, 0x60, 0x3F,
						 0x38, 0x00, 0x76, 0x03, 0xBC, 0x7D, 0x32, 0x12, 0xAF, 0xC0,
						 0x2D, 0x89, 0x8F, 0x1B, 0xCA, 0xDD, 0xE5, 0x40, 0x08, 0x13,
						 0x29, 0x29, 0x08, 0x14};
	uint8_t tv_buff[] = {2,
						0xC3, 0x7D, 0x8C, 0x44, 0x93, 0x89, 0x0B, 0x3D, 0xAE, 0x09,
						0xDF, 0xC7, 0x18, 0xDD, 0x06, 0xB1, 0x05, 0xE3, 0xFC, 0x1E,
						//0xDF, 0xC7, 0x18, 0xDD, 0x06, 0xB5, 0x05, 0xE3, 0xFC, 0x1E,
						//0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
						0xB1, 0xBC, 0x93, 0x87, 0x24, 0x38, 0x75, 0x2B, 0xE8, 0xF8,
						0xC6, 0xD3, 0x56, 0xC7, 0x19, 0xCB, 0x31, 0xEE, 0x3C, 0x18,
						0x1D, 0x2E, 0xA3, 0x0B, 0xDB, 0xDF, 0x9B, 0xF1, 0x7D, 0x13,
						0xE7, 0x64, 0x71, 0x8C, 0x35, 0x10, 0x4B, 0x5A, 0x3D, 0x29,
						0x60, 0x0F, 0x86, 0x15, 0x9D, 0xFF, 0x57, 0xD3, 0xC8, 0x88,
						0xE1, 0xC6, 0x2C, 0x97, 0x85, 0x98, 0x60, 0x6E, 0x10, 0x6D,
						0x8D, 0x1F, 0xE3, 0x10, 0x3E, 0xBD, 0xEE, 0x3B, 0x92, 0x76,
						0x3C, 0x07, 0x0D, 0x9B, 0x9E, 0xE6, 0x18, 0xF7, 0xDA, 0xD1,
						0xEA, 0x00, 0x88, 0x50, 0x91, 0xFF, 0x78, 0x6C, 0xDB, 0xE6,
						0x4E, 0x5C, 0x8F, 0xCE, 0x0B, 0x48, 0x5C, 0x00, 0x60, 0x3F,
						0x38, 0x00, 0x76, 0x03, 0xBC, 0x7D, 0x32, 0x12, 0xAF, 0xC0,
						0x1D, 0x89, 0x8F, 0x1B, 0xCA, 0xDD, 0xE5, 0x40, 0x08, 0x13,
						0x29, 0x29, 0x08, 0x14};
		
	uint8_t* rx_ptr = &rx_buff[1];
	uint8_t* tv_ptr = &tv_buff[1];
	// Just used for logging
	int flow = 0;
	int symbol_id = 0;
	int bundle_index = 0;
	int res;
	for(auto& elapsedTime : decompressCompareRes)
	{
		timePtBundleStart = Clock::now();
		res = decompress_and_compare_approx_bfw_bundle_buffer(rx_ptr, rx_buff[0], tv_ptr, tv_buff[0], bundleSize, compressBitWidth, beta, 
															  flow, symbol_id, bundle_index, true, L_TRX);
		timePtBundleStop = Clock::now();
		if(res != 0)
		{
			std::cout << "FP Compare Error encountered.  Results do not match." << std::endl;
			//return -1;
		}
		elpasedTimeDurationUs = timePtBundleStop - timePtBundleStart;
		elapsedTime = elpasedTimeDurationUs.count();
	}

	auto minmax_pair = std::minmax_element(std::begin(decompressCompareRes),std::end(decompressCompareRes));
	float mean = std::accumulate(std::begin(decompressCompareRes),std::end(decompressCompareRes),0.0)/decompressCompareRes.size();

	std::cout << "Decompress/Compare Runtime:\nMin:  " << *minmax_pair.first << " us" << std::endl;
	std::cout << "Max:  " << *minmax_pair.second << " us" << std::endl;
	std::cout << "Mean: " << mean << " us" << std::endl;

	for(auto& elapsedTime : bundleCompareRes)
	{
		timePtBundleStart = Clock::now();
		//res = fixedpt_bundle_compare(rx_buff,tv_buff,bundleSize+1);
		res = fixedpt_bundle_compare(rx_ptr,tv_ptr,bundleSize,0,0);
		timePtBundleStop = Clock::now();
		if(res != 0)
		{
			std::cout << "Integer Compare Error encountered.  Results do not match." << std::endl;
			return -1;
		}
		elpasedTimeDurationUs = timePtBundleStop - timePtBundleStart;
		elapsedTime = elpasedTimeDurationUs.count();
	}

	minmax_pair = std::minmax_element(std::begin(bundleCompareRes),std::end(bundleCompareRes));
	mean = std::accumulate(std::begin(bundleCompareRes),std::end(bundleCompareRes),0.0)/bundleCompareRes.size();

	std::cout << "Bundle Compare Runtime:\nMin:  " << *minmax_pair.first << " us" << std::endl;
	std::cout << "Max:  " << *minmax_pair.second << " us" << std::endl;
	std::cout << "Mean: " << mean << " us" << std::endl;

	return 0;
}
