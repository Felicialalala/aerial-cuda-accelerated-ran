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
 * @file perf_metrics_utils.hpp
 * @brief Utility functions for performance metrics library
 * 
 * Self-contained timing utilities to avoid heavy dependencies.
 */

#ifndef PERF_METRICS_UTILS_HPP
#define PERF_METRICS_UTILS_HPP

#include <chrono>

namespace perf_metrics {

using t_ns = std::chrono::nanoseconds;
using t_raw = std::uint64_t;  //!< Raw counter ticks (platform-specific)

/**
 * Get current time in nanoseconds since epoch
 * 
 * Equivalent to Time::nowNs() from cuphydriver but without the dependency.
 * Uses std::chrono::system_clock for high-precision timing.
 * 
 * @return Current time as nanoseconds since epoch
 */
t_ns nowNs();

/**
 * Get monotonic time in nanoseconds for performance measurements
 * 
 * Optimized for minimal overhead when only relative time differences are needed.
 * Uses platform-specific high-performance counters when available:
 * - ARM (aarch64): Virtual counter register (cntvct_el0)
 * - x86/x86_64: TSC via RDTSC instruction
 * - Other: std::chrono::steady_clock
 * 
 * @return Monotonic time in nanoseconds
 * 
 * @note This function is optimized for speed, not absolute time accuracy.
 *       Only use for measuring time intervals, not wall-clock time.
 */
t_ns monotonicNowNs();

/**
 * Get raw monotonic counter value (fastest - no conversion)
 * 
 * Returns platform-specific raw counter value without any conversion.
 * Use for accumulating timing data; convert to nanoseconds only when needed.
 * 
 * @return Raw counter ticks (platform-specific units)
 * 
 * @note Must use rawToNs() to convert accumulated ticks to nanoseconds
 */
t_raw monotonicNowRaw();

/**
 * Convert raw counter ticks to nanoseconds
 * 
 * @param[in] raw_ticks Raw counter value from monotonicNowRaw()
 * @return Time in nanoseconds
 */
std::uint64_t rawToNs(t_raw raw_ticks);

/**
 * Convert nanoseconds to raw counter ticks
 * 
 * @param[in] nanoseconds Duration in nanoseconds
 * @return Equivalent duration in raw counter ticks
 */
t_raw nsToRaw(std::uint64_t nanoseconds);

} // namespace perf_metrics

#endif // PERF_METRICS_UTILS_HPP
