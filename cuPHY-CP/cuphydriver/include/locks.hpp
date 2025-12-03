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

#ifndef LOCKS_CLASS_H
#define LOCKS_CLASS_H

#include "exceptions.hpp"
#include <pthread.h>
#include <mutex>

/**
 * @brief Thread synchronization mutex wrapper
 *
 * RAII-style mutex wrapper around std::mutex for thread-safe critical sections.
 * Provides lock/unlock interface compatible with legacy pthread mutex code.
 * Used throughout cuPHYDriver for protecting shared data structures.
 */
class Mutex {
public:
    /**
     * @brief Construct mutex
     *
     * Initializes the underlying std::mutex (no-op for std::mutex).
     */
    Mutex(){
        // if(pthread_mutex_init(&mlock, NULL))
        //     PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Mutex init error");
    };
    
    /**
     * @brief Destroy mutex
     *
     * Destroys the underlying std::mutex (no-op for std::mutex).
     */
    ~Mutex(){
        // pthread_mutex_destroy(&mlock);
        // PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Mutex destroy error");
    };

    /**
     * @brief Acquire mutex lock (blocking)
     *
     * Blocks until the mutex is acquired. Should always be paired with unlock().
     *
     * @return 0 on success
     */
    int lock()
    {
        mlock.lock();
        return 0;
        // return pthread_mutex_lock(&mlock);
    }

    /**
     * @brief Try to acquire mutex lock (non-blocking)
     *
     * Attempts to acquire the mutex without blocking.
     *
     * @return true if lock was acquired, false if already locked
     */
    bool try_lock()
    {
        return mlock.try_lock();
    }

    /**
     * @brief Release mutex lock
     *
     * Releases the previously acquired mutex. Must be called by the same
     * thread that called lock().
     *
     * @return 0 on success
     */
    int unlock()
    {
        mlock.unlock();
        return 0;
        // return pthread_mutex_unlock(&mlock);
    }

private:
    std::mutex mlock;                                          ///< Underlying C++ standard mutex (replaced pthread_mutex_t)
};

/**
 * @brief Counting semaphore (currently disabled)
 *
 * POSIX semaphore wrapper for counting semaphore synchronization.
 * Currently not used in the codebase (disabled with #if 0).
 * Can be enabled if counting semaphore functionality is needed in the future.
 */
#if 0
class Semaphore
{
    public:
        /**
         * @brief Construct semaphore with initial count
         *
         * @param size - Initial semaphore count (number of available resources)
         */
        Semaphore(size_t size) {
            if(sem_init(&semaphore, 0, size))
                PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Semaphore init error");
        };
        
        /**
         * @brief Destroy semaphore
         */
        ~Semaphore() {
            if(sem_destroy(&semaphore))
                PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Semaphore destroy error");
        };
        
        /**
         * @brief Wait for semaphore (decrement count, blocking if zero)
         *
         * @return 0 on success, negative error code on failure
         */
        int wait() {
            return sem_wait(&semaphore);
        }

        /**
         * @brief Signal semaphore (increment count, waking one waiter)
         *
         * @return 0 on success, negative error code on failure
         */
        int signal() {
            return sem_post(&semaphore);
        }

    private:
        sem_t semaphore;                                       ///< POSIX semaphore object
};
#endif

#endif
