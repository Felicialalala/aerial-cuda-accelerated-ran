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

#ifndef LOGGER_H
#define LOGGER_H

#include <sstream>
#include <iostream>
#include "time.hpp"
#include "nvlog.hpp"

/**
 * @brief String builder utility using stream operators
 *
 * Convenience class for building strings using << operator chaining.
 * Automatically converts to std::string when needed. Provides a fluent
 * interface for string concatenation.
 */
class string_builder {
    std::stringstream ss_;                                     ///< Internal string stream for accumulating output

public:
    /**
     * @brief Append value to string using stream operator
     *
     * @tparam T - Type of value to append (must support operator<<)
     * @param x  - Value to append to string
     * @return Reference to this string_builder for chaining
     */
    template <class T>
    string_builder& operator<<(T const& x)
    {
        ss_ << x;
        return *this;
    }
    
    /**
     * @brief Convert to std::string
     *
     * @return Accumulated string content
     */
    operator std::string()
    {
        return ss_.str();
    }
};

/**
 * @brief Logging helper with automatic context and flushing
 *
 * RAII-style logging helper that accumulates log message via << operators
 * and automatically flushes to the log handler in the destructor. Prepends
 * file location context (file:line) to every log message.
 */
class logger_helper {
    std::stringstream ss_;                                     ///< String stream for accumulating log message
    log_handler_fn_t  log_fn_;                                 ///< Log handler function pointer (from nvlog)

public:
    /**
     * @brief Construct logger with context information
     *
     * Prepends "[PHYDRV file:line]: " prefix to the log message.
     *
     * @param log_fn - Log handler function to call on destruction
     * @param file   - Source file name (typically __FILE__ or FILE_BNAME)
     * @param func   - Function name (typically __FUNCTION__)
     * @param lineno - Line number (typically __LINE__)
     */
    logger_helper(log_handler_fn_t log_fn, const char* file, const char* func, int lineno) :
        log_fn_(log_fn)
    {
        ss_ << "[PHYDRV " << file << ":" << lineno << "]: ";
    }
    
    /**
     * @brief Append value to log message using stream operator
     *
     * @tparam T - Type of value to log (must support operator<<)
     * @param x  - Value to append to log message
     * @return Reference to this logger_helper for chaining
     */
    template <class T>
    logger_helper& operator<<(T const& x)
    {
        ss_ << x;
        return *this;
    }

    /**
     * @brief Flush log message to handler
     *
     * Automatically called when logger_helper goes out of scope.
     * Appends newline and sends complete message to log handler.
     */
    ~logger_helper()
    {
        ss_ << std::endl;
        std::string s = ss_.str();
        log_fn_(s.c_str());
    }
};

/**
 * @brief Extract basename from file path
 *
 * Macro that extracts the filename portion from __FILE__ path.
 * Uses compiler builtin to find last '/' separator at compile time.
 * Returns full path if no '/' found (for platform compatibility).
 */
#define FILE_BNAME (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)

/**
 * @brief Error-level logging macro
 *
 * Logs error messages if error logging is enabled in the context.
 * Usage: LOGE(pdctx) << "Error message: " << value;
 *
 * The message is only constructed if error logging is enabled (lazy evaluation).
 * Automatically includes file:line context and flushes on destruction.
 *
 * @param pdctx - Physical driver context pointer (must have error_log_enabled() and get_error_logger())
 */
#define LOGE(pdctx)                \
    if(pdctx->error_log_enabled()) \
    logger_helper(pdctx->get_error_logger(), FILE_BNAME, __FUNCTION__, __LINE__)

/**
 * @brief Info-level logging macro
 *
 * Logs informational messages if info logging is enabled in the context.
 * Usage: LOGI(pdctx) << "Info message: " << value;
 *
 * The message is only constructed if info logging is enabled (lazy evaluation).
 * Automatically includes file:line context and flushes on destruction.
 *
 * @param pdctx - Physical driver context pointer (must have info_log_enabled() and get_info_logger())
 */
#define LOGI(pdctx)               \
    if(pdctx->info_log_enabled()) \
    logger_helper(pdctx->get_info_logger(), FILE_BNAME, __FUNCTION__, __LINE__)

/**
 * @brief Debug-level logging macro
 *
 * Logs debug messages if debug logging is enabled in the context.
 * Usage: LOGD(pdctx) << "Debug message: " << value;
 *
 * The message is only constructed if debug logging is enabled (lazy evaluation).
 * Automatically includes file:line context and flushes on destruction.
 *
 * @param pdctx - Physical driver context pointer (must have debug_log_enabled() and get_debug_logger())
 */
#define LOGD(pdctx)                \
    if(pdctx->debug_log_enabled()) \
    logger_helper(pdctx->get_debug_logger(), FILE_BNAME, __FUNCTION__, __LINE__)

#endif
