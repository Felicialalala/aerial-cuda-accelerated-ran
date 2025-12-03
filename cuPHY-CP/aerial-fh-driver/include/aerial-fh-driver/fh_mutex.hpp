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
 * @file fh_mutex.hpp
 * @brief High-performance mutex implementations for Aerial FH
 *
 * Provides multiple lock implementations optimized for different scenarios:
 * - PThreadSpinLock: POSIX spinlock for general use
 * - MMPSpinLock: Spinlock with architecture-specific pause instructions
 * - ASMSpinLock: Hand-optimized assembly spinlock for maximum performance
 * - SyncLockSpinLock: Portable spinlock using GCC builtins
 * - FHMutex: Main mutex interface using PThreadSpinLock
 */

#ifndef AERIAL_FH_MUTEX_HPP
#define AERIAL_FH_MUTEX_HPP

#include <pthread.h>
#include <atomic>
#include <cstdint>
#include <mutex>

namespace aerial_fh {

/**
 * @class PThreadSpinLock
 * @brief POSIX spinlock implementation
 *
 * Simple wrapper around pthread_spinlock_t for reliable performance.
 * Currently used by FHMutex for optimal balance of performance and compatibility.
 */
class PThreadSpinLock {
public:
    PThreadSpinLock();
    ~PThreadSpinLock();

    // Non-copyable and non-movable
    PThreadSpinLock(const PThreadSpinLock&) = delete;
    PThreadSpinLock& operator=(const PThreadSpinLock&) = delete;
    PThreadSpinLock(PThreadSpinLock&&) = delete;
    PThreadSpinLock& operator=(PThreadSpinLock&&) = delete;

    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;

private:
    alignas(64) pthread_spinlock_t m_spinlock{};  ///< Cache-aligned spinlock
};

/**
 * @class MMPSpinLock
 * @brief Memory Management Pause SpinLock with architecture-specific pause instructions
 *
 * Uses _mm_pause() on x86 and yield on ARM for efficient spinning.
 * Cache-aligned and optimized for short critical sections with low contention.
 */
class MMPSpinLock {
public:
    MMPSpinLock();
    ~MMPSpinLock();

    // Non-copyable and non-movable
    MMPSpinLock(const MMPSpinLock&) = delete;
    MMPSpinLock& operator=(const MMPSpinLock&) = delete;
    MMPSpinLock(MMPSpinLock&&) = delete;
    MMPSpinLock& operator=(MMPSpinLock&&) = delete;

    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;

private:
    alignas(64) pthread_spinlock_t m_spinlock{};  ///< Cache-aligned spinlock
};



/**
 * @class ASMSpinLock
 * @brief Ultra-optimized assembly spinlock for maximum performance (in some scenarios)
 *
 * Hand-optimized assembly implementation using direct XCHG (x86) or CASAL (ARM).
 * Eliminates TEST instruction overhead and uses architecture-specific optimizations.
 * Cache-aligned for optimal performance in multi-core environments.
 */
class ASMSpinLock {
public:
    ASMSpinLock() = default;
    ~ASMSpinLock() = default;

    // Non-copyable and non-movable
    ASMSpinLock(const ASMSpinLock&) = delete;
    ASMSpinLock& operator=(const ASMSpinLock&) = delete;
    ASMSpinLock(ASMSpinLock&&) = delete;
    ASMSpinLock& operator=(ASMSpinLock&&) = delete;

    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;

private:
    alignas(64) volatile int lock_word = 0;  ///< Cache line aligned lock word
};

/**
 * @class SyncLockSpinLock
 * @brief Portable spinlock using GCC builtins - works on all architectures
 *
 * Uses __sync_lock_test_and_set and __sync_lock_release GCC builtins.
 * Portable across all architectures supported by GCC.
 * Cache-aligned for optimal performance.
 */
class SyncLockSpinLock {
public:
    SyncLockSpinLock() = default;
    ~SyncLockSpinLock() = default;

    // Non-copyable and non-movable
    SyncLockSpinLock(const SyncLockSpinLock&) = delete;
    SyncLockSpinLock& operator=(const SyncLockSpinLock&) = delete;
    SyncLockSpinLock(SyncLockSpinLock&&) = delete;
    SyncLockSpinLock& operator=(SyncLockSpinLock&&) = delete;

    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;

private:
    alignas(64) volatile int lock_word = 0;  ///< Cache line aligned lock word
};

/**
 * @class FHMutex
 * @brief High-performance mutex for Aerial FH packet transmission
 *
 * Uses pthread spinlock for optimal performance in low-contention scenarios.
 * Designed for short critical sections with dedicated CPU cores.
 *
 * Compatible with std::lock_guard and other standard lock utilities.
 */
class FHMutex {
public:
    FHMutex() = default;
    ~FHMutex() = default;

    // Non-copyable and non-movable
    FHMutex(const FHMutex&) = delete;
    FHMutex& operator=(const FHMutex&) = delete;
    FHMutex(FHMutex&&) = delete;
    FHMutex& operator=(FHMutex&&) = delete;

    void lock() noexcept;
    void unlock() noexcept;
    bool try_lock() noexcept;

private:
    //Note: This lock seems to emperically work best for RU enqueue, which is why we chose it for the FHMutex
    PThreadSpinLock m_lock;
};

} // namespace aerial_fh

#endif // AERIAL_FH_MUTEX_HPP
