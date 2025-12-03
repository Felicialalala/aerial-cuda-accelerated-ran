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

#include <memory>
#include <mutex>

#pragma once

namespace scf_5g_fapi {
    template<typename T>
    class Singleton {
    public:
        // Constructor is private to disallow direct instantiation
        Singleton(Singleton const&) = delete;
        void operator=(Singleton const&) = delete;

        // Static method to create the instance
        template<typename... Args>
        static void makeInstance(Args&&... args) {
        std::call_once(s_flag, [&]() { 
            s_instance = std::make_unique<T>(std::forward<Args>(args)...); 
        });

        }

        // Static method to get a reference to the instance
        static T& getInstance() {
            return *s_instance.get();
        }

        // Static method to get a pointer to the instance
        static T* getInstancePtr() {
            return s_instance.get();
        }

        static T& getInstanceRef() {
            return *s_instance.get();
        }

    private:
        // Constructor and destructor
        Singleton() {}
        ~Singleton() {}

        // Method to create the instance
        template<typename... Args>
        static void create_instance(Args&&... args) {
            try {
                s_instance = std::make_unique<T>(std::forward<Args>(args)...);
            } catch (...) {
                throw;
            }
        }

        // Flags for thread-safety
        static std::once_flag s_flag;

        // Unique pointer to the instance
        static std::unique_ptr<T> s_instance;
    };

    // Static initialization
    template<typename T>
    std::once_flag Singleton<T>::s_flag;

    template<typename T>
    std::unique_ptr<T> Singleton<T>::s_instance = nullptr;

// Primary template for detecting the presence of a static member function.
    template <typename, typename = void>
    struct has_static_member_function : std::false_type {};

    // Specialization for detecting static member function.
    template <typename C>
    struct has_static_member_function<C, std::void_t<decltype(C::execute_impl)>> : std::true_type {};

    template <typename T, typename ...Args>
    inline auto execute(Args&&... args) -> std::enable_if_t<has_static_member_function<T>::value, void>  {
        return T::execute_impl(std::forward<Args>(args)...);
    }

    // Fallback
    template <typename T, typename ...Args>
    inline auto execute(Args&&... args) -> std::enable_if_t<!has_static_member_function<T>::value, void> = delete;
}
