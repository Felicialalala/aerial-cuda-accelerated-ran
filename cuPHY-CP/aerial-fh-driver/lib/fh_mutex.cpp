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
 * @file fh_mutex.cpp
 * @brief Implementation of high-performance mutex classes for Aerial FH
 */

#include "aerial-fh-driver/fh_mutex.hpp"
#include <stdexcept>
#include <string>
#include <cerrno>

// Architecture-specific pause instruction
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>  // For _mm_pause() on x86
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 yield instruction equivalent to _mm_pause()
    static inline void _mm_pause() { __asm__ __volatile__("yield" ::: "memory"); }
#else
    // No-op for other architectures
    static inline void _mm_pause() { /* No-op */ }
#endif

namespace aerial_fh {

//=============================================================================
// PThreadSpinLock Implementation
//=============================================================================

PThreadSpinLock::PThreadSpinLock() {
    if (const int result = pthread_spin_init(&m_spinlock, PTHREAD_PROCESS_PRIVATE); result != 0) {
        throw std::runtime_error("Failed to initialize pthread spinlock: " + std::to_string(result));
    }
}

PThreadSpinLock::~PThreadSpinLock() {
    pthread_spin_destroy(&m_spinlock);
}

void PThreadSpinLock::lock() noexcept {
    pthread_spin_lock(&m_spinlock);
}

void PThreadSpinLock::unlock() noexcept {
    pthread_spin_unlock(&m_spinlock);
}

bool PThreadSpinLock::try_lock() noexcept {
    return pthread_spin_trylock(&m_spinlock) != EBUSY;
}

//=============================================================================
// MMPSpinLock Implementation
//=============================================================================

MMPSpinLock::MMPSpinLock() {
    if (const int result = pthread_spin_init(&m_spinlock, PTHREAD_PROCESS_PRIVATE); result != 0) {
        throw std::runtime_error("Failed to initialize MMP spinlock: " + std::to_string(result));
    }
}

MMPSpinLock::~MMPSpinLock() {
    pthread_spin_destroy(&m_spinlock);
}

void MMPSpinLock::lock() noexcept {
    // Try once without pause for fast path
    if (pthread_spin_trylock(&m_spinlock) == 0) {
        return;
    }

    // Spin with pause instruction for better performance
    while (pthread_spin_trylock(&m_spinlock) == EBUSY) {
        _mm_pause(); // Architecture-specific pause instruction
    }
}

void MMPSpinLock::unlock() noexcept {
    pthread_spin_unlock(&m_spinlock);
}

bool MMPSpinLock::try_lock() noexcept {
    return pthread_spin_trylock(&m_spinlock) != EBUSY;
}

//=============================================================================
// ASMSpinLock Implementation
//=============================================================================

void ASMSpinLock::lock() noexcept {
#ifdef __x86_64__
    // Ultra-optimized x86_64: eliminate TEST instruction entirely
    asm volatile(
        "movl $1, %%eax\n\t"           // Load 1 into eax (value to exchange)
        "1: xchgl %%eax, %0\n\t"       // Exchange with memory (implicit LOCK)
        "testl %%eax, %%eax\n\t"       // Test if we got 0 (was unlocked)
        "jnz 1b"                       // Retry if was locked
        : "+m" (lock_word)
        :
        : "eax", "memory", "cc"
    );
#elif defined(__aarch64__) || defined(__arm__)
    // ARM64/Grace CPU: Use CAS (Compare-And-Swap) for proper test-and-set
    int expected = 0, desired = 1;
    asm volatile(
        "1: mov %w0, #0\n\t"            // Set expected = 0 (unlocked)
        "casal %w0, %w1, [%2]\n\t"     // Compare-and-swap with acquire-release
        "cbnz %w0, 1b"                 // If CAS failed (returned non-zero), retry
        : "=&r" (expected)
        : "r" (desired), "r" (&lock_word)
        : "memory"
    );
#else
    while (__sync_lock_test_and_set(&lock_word, 1));
#endif
}

void ASMSpinLock::unlock() noexcept {
#ifdef __x86_64__
    asm volatile("movl $0, %0" : "=m" (lock_word) :: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    // ARM64/Grace CPU: Store-release for proper unlock semantics
    asm volatile(
        "stlr wzr, [%0]"               // Store-release zero (proper release semantics)
        :
        : "r" (&lock_word)
        : "memory"
    );
#else
    __sync_lock_release(&lock_word);
#endif
}

bool ASMSpinLock::try_lock() noexcept {
#ifdef __x86_64__
    int result;
    asm volatile(
        "movl $1, %%eax\n\t"           // Load 1 into eax
        "xchgl %%eax, %1\n\t"          // Exchange with memory
        "movl %%eax, %0"               // Store result
        : "=r" (result), "+m" (lock_word)
        :
        : "eax", "memory", "cc"
    );
    return result == 0;  // Return true if we got the lock (old value was 0)
#elif defined(__aarch64__) || defined(__arm__)
    // ARM64: Single attempt with compare-and-swap
    int expected = 0, desired = 1;
    asm volatile(
        "mov %w0, #0\n\t"               // Set expected = 0 (unlocked)
        "casal %w0, %w1, [%2]"          // Compare-and-swap with acquire-release
        : "=&r" (expected)
        : "r" (desired), "r" (&lock_word)
        : "memory"
    );
    return expected == 0;  // Return true if we got the lock (old value was 0)
#else
    return !__sync_lock_test_and_set(&lock_word, 1);
#endif
}

//=============================================================================
// SyncLockSpinLock Implementation
//=============================================================================

void SyncLockSpinLock::lock() noexcept {
    while (__sync_lock_test_and_set(&lock_word, 1));
}

void SyncLockSpinLock::unlock() noexcept {
    __sync_lock_release(&lock_word);
}

bool SyncLockSpinLock::try_lock() noexcept {
    return !__sync_lock_test_and_set(&lock_word, 1);
}

//=============================================================================
// FHMutex Implementation
//=============================================================================

void FHMutex::lock() noexcept {
    m_lock.lock();
}

void FHMutex::unlock() noexcept {
    m_lock.unlock();
}

bool FHMutex::try_lock() noexcept {
    return m_lock.try_lock();
}

} // namespace aerial_fh