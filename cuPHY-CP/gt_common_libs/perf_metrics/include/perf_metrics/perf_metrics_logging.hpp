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

#ifndef PERF_METRICS_LOGGING_HPP
#define PERF_METRICS_LOGGING_HPP

// NVLOG includes for logging functionality
#include "aerial_event_code.h"
#include "nvlog_fmt.hpp"

/**
 * Performance metrics logging utility
 * 
 * Encapsulates the NVLOG dispatch logic for different log levels.
 * This utility handles the compile-time selection of the appropriate
 * NVLOG_FMT macro based on the log level.
 */

namespace perf_metrics {

/**
 * Log level enumeration for performance metrics
 */
enum class LogLevel {
    VERBOSE,  //!< Verbose logging (NVLOGV_FMT)
    DEBUG,    //!< Debug logging (NVLOGD_FMT)
    INFO,     //!< Info logging (NVLOGI_FMT) - default
    WARN,     //!< Warning logging (NVLOGW_FMT)
    ERROR     //!< Error logging (NVLOGE_FMT) - requires event code
};

/**
 * Direct template function specializations for each log level
 * Zero overhead - compiles directly to NVLOG macro calls
 */

/**
 * Macro-based logging dispatch for different levels
 * These macros properly handle variadic arguments with NVLOG
 */

// VERBOSE level logging macro
#define PERF_METRICS_LOG_VERBOSE(NvlogTag, format, ...) \
    NVLOGV_FMT(NvlogTag, format, ##__VA_ARGS__)

// DEBUG level logging macro  
#define PERF_METRICS_LOG_DEBUG(NvlogTag, format, ...) \
    NVLOGD_FMT(NvlogTag, format, ##__VA_ARGS__)

// INFO level logging macro
#define PERF_METRICS_LOG_INFO(NvlogTag, format, ...) \
    NVLOGI_FMT(NvlogTag, format, ##__VA_ARGS__)

// WARN level logging macro
#define PERF_METRICS_LOG_WARN(NvlogTag, format, ...) \
    NVLOGW_FMT(NvlogTag, format, ##__VA_ARGS__)

// ERROR level logging macro
#define PERF_METRICS_LOG_ERROR(NvlogTag, format, ...) \
    NVLOGE_FMT(NvlogTag, AERIAL_CUPHY_EVENT, format, ##__VA_ARGS__)

/**
 * Main logging macro with compile-time level dispatch
 * Usage: PERF_METRICS_LOG(TAG, LogLevel::INFO, "format", args...)
 * 
 * @param NvlogTag Compile-time NVLOG tag
 * @param Level Log level to use (perf_metrics::LogLevel enum)
 * @param format Format string (fmt library syntax)
 * @param ... Format arguments
 */
#define PERF_METRICS_LOG(NvlogTag, Level, format, ...) \
    do { \
        if constexpr (Level == perf_metrics::LogLevel::VERBOSE) { \
            PERF_METRICS_LOG_VERBOSE(NvlogTag, format, ##__VA_ARGS__); \
        } else if constexpr (Level == perf_metrics::LogLevel::DEBUG) { \
            PERF_METRICS_LOG_DEBUG(NvlogTag, format, ##__VA_ARGS__); \
        } else if constexpr (Level == perf_metrics::LogLevel::INFO) { \
            PERF_METRICS_LOG_INFO(NvlogTag, format, ##__VA_ARGS__); \
        } else if constexpr (Level == perf_metrics::LogLevel::WARN) { \
            PERF_METRICS_LOG_WARN(NvlogTag, format, ##__VA_ARGS__); \
        } else if constexpr (Level == perf_metrics::LogLevel::ERROR) { \
            PERF_METRICS_LOG_ERROR(NvlogTag, format, ##__VA_ARGS__); \
        } \
    } while(0)

} // namespace perf_metrics

#endif // PERF_METRICS_LOGGING_HPP
