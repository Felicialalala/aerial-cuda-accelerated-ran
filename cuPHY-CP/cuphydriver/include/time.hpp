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

#ifndef TIME_H
#define TIME_H

#include <cstdint>
#include <iostream>
#include <chrono>
#include <time.h>

/**
 * @brief Gets the current time in nanoseconds.
 * 
 * @return Current time in nanoseconds since epoch
 */
uint64_t get_ns(void);

/**
 * @brief Waits for a specified duration in nanoseconds.
 * 
 * @param ns Nanoseconds to wait
 */
void     wait_ns(uint64_t);

/**
 * @brief Nanoseconds duration type alias.
 */
using t_ns = std::chrono::nanoseconds;

/**
 * @brief Microseconds duration type alias.
 */
using t_us = std::chrono::microseconds;

/**
 * @brief Milliseconds duration type alias.
 */
using t_ms = std::chrono::milliseconds;

/**
 * @brief System clock timepoint type alias.
 */
using t_tp = std::chrono::time_point<std::chrono::system_clock>;

/**
 * @brief Time utility class for timing operations and conversions.
 * 
 * Provides static methods for getting timestamps, calculating time differences,
 * and converting between time units.
 */
class Time {
public:
    /**
     * @brief Constructor.
     */
    Time();
    
    /**
     * @brief Destructor.
     */
    ~Time();

    /**
     * @brief Gets the current system clock timepoint.
     * 
     * @return Current system clock timepoint
     */
    static t_tp nowTimepoint();
    
    /**
     * @brief Gets the current time in nanoseconds since epoch.
     * 
     * @return Current time in nanoseconds
     */
    static t_ns nowNs();
    
    /**
     * @brief Returns a zero-duration nanosecond value.
     * 
     * @return Zero nanoseconds
     */
    static t_ns zeroNs();
    
    /**
     * @brief Calculates the difference between two timestamps.
     * 
     * @param first First timestamp
     * @param second Second timestamp
     * @return Time difference (first - second)
     */
    static t_ns getDifference(t_ns& first, t_ns& second);
    
    /**
     * @brief Calculates the time elapsed from a timestamp to now.
     * 
     * @param first Reference timestamp
     * @return Time difference (now - first)
     */
    static t_ns getDifferenceNowToNs(t_ns& first);
    
    /**
     * @brief Calculates the time remaining from now to a future timestamp.
     * 
     * @param first Future timestamp
     * @return Time difference (first - now)
     */
    static t_ns getDifferenceNsToNow(t_ns& first);
    
    /**
     * @brief Checks if first timestamp is greater than second.
     * 
     * @param first First timestamp
     * @param second Second timestamp
     * @return true if first > second, false otherwise
     */
    static bool greater(t_ns& first, t_ns& second);
    
    /**
     * @brief Checks if first timestamp is greater than or equal to second.
     * 
     * @param first First timestamp
     * @param second Second timestamp
     * @return true if first >= second, false otherwise
     */
    static bool greatereq(t_ns& first, t_ns& second);
    
    /**
     * @brief Checks if the difference between two timestamps is below a threshold.
     * 
     * @param first First timestamp
     * @param second Second timestamp
     * @param threshold Threshold duration
     * @return true if (first - second) < threshold, false otherwise
     */
    static bool belowThreshold(t_ns& first, t_ns& second, t_ns& threshold);
    
    /**
     * @brief Checks if the difference between two timestamps is above a threshold.
     * 
     * @param first First timestamp
     * @param second Second timestamp
     * @param threshold Threshold duration
     * @return true if (first - second) > threshold, false otherwise
     */
    static bool aboveThreshold(t_ns& first, t_ns& second, t_ns& threshold);
    
    /**
     * @brief Waits for a specified duration.
     * 
     * @param waitns Duration to wait in nanoseconds
     * @return true after wait completes
     */
    static bool waitDurationNs(t_ns& waitns);
    
    /**
     * @brief Waits until the current time reaches the specified timestamp.
     * 
     * @param waitns Target absolute timestamp (spins until nowNs() >= waitns)
     * @return true after reaching the target time
     */
    static bool waitTimeNs(t_ns& waitns);
    
    /**
     * @brief Converts nanoseconds to microseconds.
     * 
     * @param time Time in nanoseconds
     * @return Time in microseconds
     */
    static t_us NsToUs(t_ns time);
    
    /**
     * @brief Converts microseconds to nanoseconds.
     * 
     * @param time Time in microseconds
     * @return Time in nanoseconds
     */
    static t_ns UsToNs(t_us time);
};

#endif