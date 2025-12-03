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

#ifndef CSIRS_LOOKUP_HPP
#define CSIRS_LOOKUP_HPP
#include <unordered_map>
#include <random>
#include <map>
#include <tuple>
#include <set>
#include "slot_command.hpp"

namespace csirs_lookup_api
{
    using std::hash;
    uint8_t static constexpr L0_RANGE[] = {0,14};
    uint8_t static constexpr L1_RANGE[] = {2,13};
    uint8_t static constexpr K_RANGE[] = {0,12};
    uint8_t static constexpr MAX_K_INDICES = 4; //K can take value K0, K1, K2
    uint8_t static constexpr MAX_ROW = 18;
    uint8_t static constexpr MAX_K_INDICES_VALUE = 12;
    uint8_t static constexpr MAX_SYMBOLS = 14;

    // Helper function to combine multiple values into a single uint64_t key
    inline uint64_t makeKey(uint8_t row, uint8_t syml0, uint8_t syml1, uint16_t freq_bits) {
        uint64_t key = 0;
        key |= static_cast<uint64_t>(row) << 56;
        key |= static_cast<uint64_t>(syml0) << 48;
        key |= static_cast<uint64_t>(syml1) << 40;
        key |= static_cast<uint64_t>(freq_bits) << 24;
        return key;
    }

    struct CsirsSymbLocRowOptimized
    {
        uint8_t numPorts;
        uint8_t maxKBits;
        uint8_t maxK;
        uint8_t maxL;
        uint8_t lenKBar;
        uint8_t lenLBar;
        uint8_t lenKPrime;
        uint8_t lenLPrime;
        uint8_t kIndices[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];
        uint8_t kOffsets[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];
        uint8_t lIndices[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];
        uint8_t lOffsets[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];
        uint8_t cdmGroupIndices[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];
        uint8_t cdmGroupSize;

    };

    #define CSIRS_SYMBOL_LOCATION_TABLE_OPTIMIZED_LENGTH (19)
    struct CsirsTableOptimized
    {
        CsirsSymbLocRowOptimized rowData[CSIRS_SYMBOL_LOCATION_TABLE_OPTIMIZED_LENGTH];
    };

    static constexpr CsirsTableOptimized csirs_table_optimized= {
        {
            {0,0,0,0,0, 0, 0, 0,    // Unused row so that we can use 1-based indexing
                {0},
                {0},
                {0},
                {0},
                {0},
                0
            },
            {1,4,1,1, 3, 1, 1, 1,    // Corresponds to 3GPP 38.211 Table 7.4.1.5.3-1 Row 1, optimized for orthogonal time and frequency
                {0, 0, 0},
                {0, 4, 8},
                {0},
                {0},
                {0, 0, 0},
                1
            },
            {1,11, 1, 1, 1, 1, 1, 1,    // Row 2
                {0},
                {0},
                {0},
                {0},
                {0},
                1
            },
            {2,6,1,1, 1, 1, 2, 1,    // Row 3
                {0},
                {0},
                {0},
                {0},
                {0},
                2
            },
            {4,3,1,1, 2, 1, 2, 1,    // Row 4
                {0, 0},
                {0, 2},
                {0},
                {0},
                {0, 1},
                2
            },
            {4,6,1,1, 1, 2, 2, 1,    // Row 5
                {0},
                {0},
                {0, 0},
                {0, 1},
                {0, 1},
                2
            },
            {8,6,4,1, 4, 1, 2, 1,    // Row 6
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0},
                {0},
                {0, 1, 2, 3},
                2
            },
            {8,6,2,1, 2, 2, 2, 1,    // Row 7
                {0, 1},
                {0, 0},
                {0, 0},
                {0, 1},
                {0, 1, 2, 3},
                2
            },
            {8,6,2,1, 2, 1, 2, 1,    // Row 8
                {0, 1},
                {0, 0},
                {0},
                {0},
                {0, 1},
                4
            },
            {12,6,6,1, 6, 1, 2, 1,   // Row 9
                {0, 1, 2, 3, 4, 5},
                {0, 0, 0, 0, 0},
                {0},
                {0},
                {0, 1, 2, 3, 4, 5},
                2
            },
            {12,6,3,1,3, 1, 2, 2,   // Row 10
                {0, 1, 2},
                {0, 0, 0},
                {0},
                {0},
                {0, 1, 2},
                4
            },
            {16,6, 3, 1, 4, 2, 2, 1,   // Row 11
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0, 0},
                {0, 1},
                {0, 1, 2, 3, 4, 5, 6, 7},
                2
            },
            {16,6, 4,1,4, 1, 2, 2,   // Row 12
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0},
                {0},
                {0, 1, 2, 3},
                4    
            },
            {24,6,3,2, 3, 4, 2, 1,  // Row 13
                {0, 1, 2},
                {0, 0, 0},
                {0, 0, 1, 1},
                {0, 1, 0, 1},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                2
            },
            {24,6, 3,2,3, 2, 2, 2,   // Row 14
                {0, 1, 2},
                {0, 0, 0},
                {0, 1},
                {0, 0},
                {0, 1, 2, 3, 4, 5},
                4
            },
            {24,6, 3, 1, 3, 1, 2, 4,   // Row 15
                {0, 1, 2},
                {0, 0, 0},
                {0},
                {0},
                {0, 1, 2},
                8
            },
            {32,6, 4, 2, 4, 4, 2, 1,  // Row 16
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0, 0, 1, 1},
                {0, 1, 0, 1},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                2
            },
            {32, 6, 4, 2, 4, 2, 2, 2,   // Row 17
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0, 1},
                {0, 0},
                {0, 1, 2, 3, 4, 5, 6, 7},
                4
            },
            {32, 6, 4, 1, 4, 1, 2, 4,   // Row 18
                {0, 1, 2, 3},
                {0, 0, 0, 0},
                {0},
                {0},
                {0, 1, 2, 3},
                8 
            }
        }
    };

    struct CsirsPortTxLocation {
        uint16_t symbol_mask;
        uint16_t re_mask;
    };
    
    // Maximum number of ports based on the csirs_table_optimized data
    static constexpr uint8_t MAX_CSIRS_PORTS = 32;
    
    struct CsirsPortData {
        uint8_t num_ports{0};
        CsirsPortTxLocation port_tx_locations[MAX_CSIRS_PORTS];
        
        // Constructor to initialize the array
        CsirsPortData() : num_ports(0) {
            // Initialize all elements to zero
            for (uint8_t i = 0; i < MAX_CSIRS_PORTS; i++) {
                port_tx_locations[i].symbol_mask = 0;
                port_tx_locations[i].re_mask = 0;
            }
        }
        
        // Helper method to add a location (replaces vector::push_back)
        bool addPortLocation(const CsirsPortTxLocation& location) {
            if (num_ports > MAX_CSIRS_PORTS) {
                printf("ERROR: Attempting to add port location %d beyond MAX_CSIRS_PORTS (%d)\n", num_ports, MAX_CSIRS_PORTS);
                return false; // Array is full
            }
            port_tx_locations[num_ports] = location;
            num_ports++;
            //printf("Added port location %d\n", num_ports);
            return true;
        }
        
        // Helper method to get size (replaces vector::size)
        uint8_t size() const {
            return num_ports;
        }
        
        // Helper method to check if empty (replaces vector::empty)
        bool empty() const {
            return num_ports == 0;
        }
        
        // Safety check method to validate data integrity
        bool isValid() const {
            if (num_ports > MAX_CSIRS_PORTS) {
                return false; // Invalid number of ports
            }
            
            // Check that all used entries have valid data
            for (uint8_t i = 0; i < num_ports; i++) {
                // Basic validation - symbol_mask and re_mask should not both be zero
                if (port_tx_locations[i].symbol_mask == 0 && port_tx_locations[i].re_mask == 0) {
                    // This might be valid for some cases, but log it
                    printf("WARNING: Port %d has zero symbol_mask and re_mask\n", i);
                }
            }
            
            return true;
        }
        
        // Method to clear all data
        void clear() {
            num_ports = 0;
            for (uint8_t i = 0; i < MAX_CSIRS_PORTS; i++) {
                port_tx_locations[i].symbol_mask = 0;
                port_tx_locations[i].re_mask = 0;
            }
        }
        
        // Safe accessor method with bounds checking
        const CsirsPortTxLocation* getPortLocation(uint8_t index) const {
            if (index >= num_ports) {
                printf("ERROR: Port index %d out of bounds (num_ports=%d)\n", index, num_ports);
                return nullptr;
            }
            return &port_tx_locations[index];
        }
    };

    class CsirsLookup {
    private:
        // Static map using a 64-bit key instead of a tuple
        inline static std::unordered_map<uint64_t, CsirsPortData> portMap;
        // Private constructor for singleton pattern
        CsirsLookup() {
            // No need to populate in constructor since the map is static and populated at program start
            if (portMap.empty()) {
                populateLookup();
            }
        }
        
        // Prevent copying and assignment
        CsirsLookup(const CsirsLookup&) = delete;
        CsirsLookup& operator=(const CsirsLookup&) = delete;

        void makeEntryForSymbolAndRESet(uint8_t row, uint8_t l0, uint8_t l1, uint8_t k0, uint8_t k1, uint8_t k2, uint8_t k3, uint8_t k4, uint8_t k5) {
            
            uint16_t freq_domain = 0;
            const auto& rowData = csirs_table_optimized.rowData[row];
            if(rowData.maxL == 1 || l1 == l0)
            {
                l1 = 0;
            }
            uint8_t li[] = {l0, l1};
            CsirsPortData portData;
            uint8_t ki[] = {k0, k1, k2, k3, k4, k5};
            uint8_t ki_expanded[] = {MAX_K_INDICES_VALUE, MAX_K_INDICES_VALUE, MAX_K_INDICES_VALUE, MAX_K_INDICES_VALUE};
            for(uint8_t i = 0; i < rowData.maxK; i++) {
                if(ki[i]>= rowData.maxKBits) {
                    return;
                }
                freq_domain |= 1 << ki[i];
                if(row == 4) {
                    ki[i] = 4*ki[i];
                }
                else if(!(row == 1 || row == 2)) {
                    ki[i] = 2*ki[i];
                }
            }

            uint64_t key = makeKey(row, l0, l1, freq_domain);
            auto it = portMap.find(key);
            if (it != portMap.end()) {
                return;
            }

            //portData.num_ports = rowData.numPorts;
            std::set<uint8_t> unique_ports;
            CsirsPortTxLocation portTxLocation;

            for(uint8_t l_bar = 0; l_bar < rowData.lenLBar; l_bar++) {
                for(uint8_t k_bar = 0; k_bar < rowData.lenKBar; k_bar++) {
                    uint8_t cdmGroupIndex = rowData.cdmGroupIndices[l_bar * rowData.lenKBar + k_bar];
                    for(uint8_t seq = 0; seq < rowData.cdmGroupSize; seq++) {
                        uint8_t port = seq + cdmGroupIndex * rowData.cdmGroupSize;
                        portTxLocation.symbol_mask = 0;
                        if(row != 1 || k_bar == 0) {
                            portTxLocation.re_mask = 0;
                        }
                        for (uint8_t l_prime = 0; l_prime < rowData.lenLPrime; l_prime++) {
                            uint8_t l = li[rowData.lIndices[l_bar]] + rowData.lOffsets[l_bar] + l_prime;
                            uint16_t reMask = 0;
                            for (uint8_t k_prime = 0; k_prime < rowData.lenKPrime; k_prime++) {
                                uint8_t k = ki[rowData.kIndices[k_bar]] + rowData.kOffsets[k_bar] + k_prime;
                                reMask |= 1 << (CUPHY_N_TONES_PER_PRB - k - 1);
                            }
                            
                            portTxLocation.symbol_mask |= 1 << l;
                            if(row == 1) {
                                portTxLocation.re_mask |= reMask;
                            }
                            else {
                                portTxLocation.re_mask = reMask;
                            }
                            unique_ports.insert(port);
                        }
                        
                        if(row == 1 && k_bar < rowData.lenKBar - 1) {
                            continue;
                        }
                        portData.addPortLocation(portTxLocation);
                    }
                }
            }
            if(portData.size() != rowData.numPorts && row != 8) {
                printf("Error: num_ports_covered %d != rowData.numPorts %d for row %d, freq_domain 0x%04x, syml0 %d, syml1 %d", portData.size(), rowData.numPorts, row, freq_domain, l0, l1);
            }
            portMap[key] = portData;
            return;
        }

        void populateLookup() {
            for(uint8_t l0 = L0_RANGE[0]; l0 < L0_RANGE[1]; l0++) {
                for(uint8_t l1 = L1_RANGE[0]; l1 < L1_RANGE[1]; l1++) {
                    for(uint8_t k0 = K_RANGE[0]; k0 < K_RANGE[1]; k0++) {
                        for(uint8_t k1 = k0 + 1; k1 <= K_RANGE[1]; k1++) {
                            for(uint8_t k2 = k1 + 1; k2 <= K_RANGE[1] + 1; k2++) {   
                                for(uint8_t k3 = k2 + 1; k3 <= K_RANGE[1] + 2; k3++) {
                                    for(uint8_t k4 = k3 + 1; k4 <= K_RANGE[1] + 3; k4++) {
                                        for(uint8_t k5 = k4 + 1; k5 <= K_RANGE[1] + 4; k5++) {
                                            for(uint8_t row = 1; row <= MAX_ROW; row++) {
                                                makeEntryForSymbolAndRESet(row, l0, l1, k0, k1, k2, k3, k4, k5);
                                            }
                                        }
                                    }
                                }
                            }
                        }   
                    }
                }
            }
            printf("Lookup table populated with %zu entries\n", portMap.size());
        }

    public:
        static CsirsLookup& getInstance() {
            static CsirsLookup instance;
            return instance;
        }
        __attribute__((hot)) inline bool getPortInfo(uint8_t row, uint16_t freq_domain, uint8_t syml0, uint8_t syml1, const CsirsPortData*& outPortInfo) const {
            const auto& rowData = csirs_table_optimized.rowData[row];
            
            // Mask frequency domain bits based on row's max bits
            freq_domain &= (1 << rowData.maxKBits) - 1;
            
            // Fast path for empty frequency domain
            if (__builtin_expect(freq_domain == 0, 0)) {
                return false;
            }
            
            // Handle l1 value when maxL is 1
            if (rowData.maxL == 1) {
                syml1 = 0;
            }
            
            // Use popcnt to quickly check if we have too many bits set
            if (__builtin_popcount(freq_domain) > rowData.maxK) {
                // Too many bits set, we need to filter
                uint16_t filtered_domain = 0;
                uint8_t count = 0;
                
                // Take only the first maxK bits
                while (freq_domain && count < rowData.maxK) {
                    int pos = __builtin_ctz(freq_domain);
                    filtered_domain |= (1U << pos);
                    freq_domain &= ~(1U << pos);
                    count++; 
                }
                freq_domain = filtered_domain;
            }
            // Create key and lookup
            uint64_t key = makeKey(row, syml0, syml1, freq_domain);
            auto it = portMap.find(key);
            
            if (__builtin_expect(it == portMap.end(), 0)) {
                return false;
            }
            
            outPortInfo = &(it->second);
            return true;
        }
        
        size_t get_size() const {
            return portMap.size();
        }

        void testCsirsLookup() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 7);
            uint8_t row = dis(gen);
            uint8_t syml0 = dis(gen);
            uint8_t syml1 = dis(gen);
            uint16_t freq_domain = dis(gen);
            const CsirsPortData* outPortInfo;
            bool result = getPortInfo(row, freq_domain, syml0, syml1, outPortInfo);
            if (result) {
                printf("Row: %d, Syml0: %d, Syml1: %d, Freq_domain: 0x%04x, Num_ports: %d\n", row, syml0, syml1, freq_domain, outPortInfo->num_ports);
            }
        }
        
        // Test method for the new fixed-array implementation
        void testFixedArrayImplementation() {
            printf("=== Testing Fixed Array Implementation ===\n");
            
            // Test 1: Create a CsirsPortData and add some locations
            CsirsPortData testData;
            printf("Test 1: Initial state - num_ports: %d, empty: %s\n", 
                   testData.num_ports, testData.empty() ? "true" : "false");
            
            // Test 2: Add some port locations
            CsirsPortTxLocation loc1 = {0x1234, 0x5678};
            CsirsPortTxLocation loc2 = {0x9ABC, 0xDEF0};
            
            bool result1 = testData.addPortLocation(loc1);
            bool result2 = testData.addPortLocation(loc2);
            
            printf("Test 2: Added locations - result1: %s, result2: %s, num_ports: %d\n", 
                   result1 ? "true" : "false", result2 ? "true" : "false", testData.num_ports);
            
            // Test 3: Access the data
            const CsirsPortTxLocation* ptr1 = testData.getPortLocation(0);
            const CsirsPortTxLocation* ptr2 = testData.getPortLocation(1);
            const CsirsPortTxLocation* ptr3 = testData.getPortLocation(2); // Should be null
            
            printf("Test 3: Access test - ptr1: %s, ptr2: %s, ptr3: %s\n",
                   ptr1 ? "valid" : "null", ptr2 ? "valid" : "null", ptr3 ? "valid" : "null");
            
            if (ptr1) {
                printf("  Port 0 - symbol_mask: 0x%04x, re_mask: 0x%04x\n", 
                       ptr1->symbol_mask, ptr1->re_mask);
            }
            if (ptr2) {
                printf("  Port 1 - symbol_mask: 0x%04x, re_mask: 0x%04x\n", 
                       ptr2->symbol_mask, ptr2->re_mask);
            }
            
            // Test 4: Validation
            printf("Test 4: Validation - isValid: %s\n", testData.isValid() ? "true" : "false");
            
            // Test 5: Clear and verify
            testData.clear();
            printf("Test 5: After clear - num_ports: %d, empty: %s\n", 
                   testData.num_ports, testData.empty() ? "true" : "false");
            
            printf("=== Fixed Array Test Complete ===\n");
        }
    };
}
#endif