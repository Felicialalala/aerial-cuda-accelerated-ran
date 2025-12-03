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

/**
 * @file mod_comp_runtime_config.hpp
 * @brief Runtime configuration structures and functions for modulation compression
 */

#ifndef MOD_COMP_RUNTIME_CONFIG_HPP
#define MOD_COMP_RUNTIME_CONFIG_HPP

#include <cstdint>
#include <memory>
#include "aerial-fh-driver/api.hpp"

namespace aerial_fh {

/**
 * @brief Runtime configuration for modulation compression
 *
 * Modulation compression is a domain-specific compression technique that exploits
 * the discrete nature of modulated symbols (QPSK, 16QAM, 64QAM, 256QAM) to achieve
 * better compression ratios than generic block floating point (BFP) compression.
 * This structure holds the runtime parameters for modulation compression.
 */
struct ModCompRuntimeConfig {
    bool enabled{false};                                       ///< Enable modulation compression (false=use BFP compression)
    uint8_t bit_width{9};                                      ///< Bit width for compressed samples (default 9 bits)
    uint16_t max_flows{32};                                    ///< Maximum number of concurrent flows (eAxC streams)
    uint16_t max_prb_info_prb_symbol{273};                     ///< Maximum PRB info entries per symbol (max 273 PRBs in 5G NR)
    uint16_t max_comp_info_per_prb{4};                         ///< Maximum compression info entries per PRB
    
    /**
     * @brief Default constructor
     *
     * Initializes with modulation compression disabled and default parameters.
     */
    ModCompRuntimeConfig() = default;
    
    /**
     * @brief Construct with explicit enable flag and bit width
     *
     * @param _enabled   - Enable modulation compression
     * @param _bit_width - Bit width for compressed samples (9 bits for default)
     */
    ModCompRuntimeConfig(bool _enabled, uint8_t _bit_width = 9) 
        : enabled(_enabled), bit_width(_bit_width) {}
    
    /**
     * @brief Construct from compression method and bit width
     *
     * Automatically enables modulation compression if method is MODULATION_COMPRESSION.
     *
     * @param method     - User data compression method
     * @param _bit_width - Bit width for compressed samples
     */
    ModCompRuntimeConfig(UserDataCompressionMethod method, uint8_t _bit_width)
        : enabled(method == UserDataCompressionMethod::MODULATION_COMPRESSION)
        , bit_width(_bit_width) {}
};

/**
 * @brief Modulation compression parameters structure
 *
 * GPU-friendly structure containing modulation compression parameters.
 * This structure is copied to device memory for use by compression kernels.
 * All parameters are sized for efficient GPU memory access.
 */
struct mod_compression_params {
    uint8_t prb_size;                                          ///< PRB size in bytes after compression
    uint8_t mod_comp_enabled;                                  ///< Modulation compression enable flag (0=disabled, 1=enabled)
    uint16_t max_flows;                                        ///< Maximum number of concurrent flows (eAxC streams)
    uint16_t max_prb_info_prb_symbol;                          ///< Maximum PRB info entries per symbol (max 273 PRBs in 5G NR)
    uint16_t max_comp_info_per_prb;                            ///< Maximum compression info entries per PRB
    
    /**
     * @brief Default constructor
     *
     * Initializes with modulation compression disabled and default resource limits.
     */
    mod_compression_params() : prb_size(0), mod_comp_enabled(0), 
                              max_flows(32), max_prb_info_prb_symbol(273), 
                              max_comp_info_per_prb(4) {}
    
    /**
     * @brief Construct from runtime configuration
     *
     * Converts ModCompRuntimeConfig to GPU-friendly parameters structure.
     * Calculates PRB size from bit width: (bit_width * 2) / 8 bytes
     * Factor of 2 accounts for I and Q components.
     *
     * @param config - Runtime configuration to convert
     */
    mod_compression_params(const ModCompRuntimeConfig& config)
        : prb_size(config.enabled ? (config.bit_width * 2) / 8 : 0)
        , mod_comp_enabled(config.enabled ? 1 : 0)
        , max_flows(config.max_flows)
        , max_prb_info_prb_symbol(config.max_prb_info_prb_symbol)
        , max_comp_info_per_prb(config.max_comp_info_per_prb) {}
};

/**
 * @brief Check if modulation compression is enabled for a given compression method
 *
 * @param method - User data compression method to check
 * @return true if modulation compression is enabled, false if BFP or no compression
 */
inline bool is_mod_comp_enabled(UserDataCompressionMethod method) {
    return method == UserDataCompressionMethod::MODULATION_COMPRESSION;
}

/**
 * @brief Check if modulation compression is enabled (overload with bit width parameter)
 *
 * Note: Currently the bit_width parameter is not used in the check, but provided
 * for API consistency and potential future use.
 *
 * @param method    - User data compression method to check
 * @param bit_width - Bit width (unused, reserved for future use)
 * @return true if modulation compression is enabled, false if BFP or no compression
 */
inline bool is_mod_comp_enabled(UserDataCompressionMethod method, uint8_t bit_width) {
    return method == UserDataCompressionMethod::MODULATION_COMPRESSION;
}

/**
 * @brief Get the effective PRB size based on compression method
 *
 * Calculates the PRB size in bytes based on whether modulation compression
 * is enabled. For mod comp: (bit_width * 2) / 8 bytes. Factor of 2 accounts
 * for I and Q components.
 *
 * @param method           - User data compression method
 * @param bit_width        - Bit width for compressed samples
 * @param default_prb_size - Default PRB size when modulation compression is not enabled (BFP or uncompressed)
 * @return Effective PRB size in bytes
 */
inline size_t get_effective_prb_size(UserDataCompressionMethod method, uint8_t bit_width, size_t default_prb_size) {
    if (is_mod_comp_enabled(method)) {
        // For modulation compression, PRB size is determined by bit width
        return (bit_width * 2) / 8; // 2 for I and Q components
    }
    return default_prb_size;
}

/**
 * @brief Create modulation compression configuration from compression method and bit width
 *
 * Convenience factory function that constructs a ModCompRuntimeConfig.
 *
 * @param method    - User data compression method
 * @param bit_width - Bit width for compressed samples
 * @return Initialized ModCompRuntimeConfig object
 */
inline ModCompRuntimeConfig create_mod_comp_config(UserDataCompressionMethod method, uint8_t bit_width) {
    return ModCompRuntimeConfig(method, bit_width);
}

/**
 * @brief Create GPU-friendly modulation compression parameters from runtime config
 *
 * Convenience factory function that converts runtime configuration to
 * GPU-friendly parameters structure for device memory upload.
 *
 * @param config - Runtime configuration to convert
 * @return Initialized mod_compression_params object ready for GPU upload
 */
inline mod_compression_params create_mod_comp_params(const ModCompRuntimeConfig& config) {
    return mod_compression_params(config);
}

/**
 * @brief Check if all cells have the same compression bit width for optimization
 *
 * Determines if BFP compression parameters are constant across all cells,
 * enabling optimization paths that can use a single compression configuration.
 * Returns false if modulation compression is used on any cell, as mod comp
 * requires per-cell handling.
 *
 * @param comp_meth - Array of compression methods per cell (size: num_cells)
 * @param bit_width - Array of bit widths per cell (size: num_cells)
 * @param num_cells - Number of cells to check
 * @return true if all cells use same BFP bit width, false if mod comp or varying bit widths
 */
inline bool check_const_bfp(const uint8_t* comp_meth, const uint8_t* bit_width, int num_cells) {
    if (num_cells == 0) return true;
    
    // If first cell uses mod comp, we can't optimize
    if (comp_meth[0] == static_cast<uint8_t>(UserDataCompressionMethod::MODULATION_COMPRESSION)) {
        return false;
    }
    
    // Check if all cells have the same bit width
    int first_bit_width = bit_width[0];
    for (int i = 1; i < num_cells; i++) {
        if (bit_width[i] != first_bit_width) {
            return false;
        }
    }
    return true;
}

} // namespace aerial_fh

#endif // MOD_COMP_RUNTIME_CONFIG_HPP
