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

#pragma once
#ifndef MEMTRACE_H
#define MEMTRACE_H
#include <cstdlib>

#define MI_MEMTRACE_CONFIG_ENABLE (1)
#define MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE (2)

#if defined(__cplusplus)
#define aerial_decl_externc extern "C"
#else
#define aerial_decl_externc
#endif

aerial_decl_externc int mi_memtrace_get_config(void) __attribute__((weak));;
aerial_decl_externc void mi_memtrace_set_config(int config) __attribute__((weak));;

inline int memtrace_get_config(void)
{
   if (mi_memtrace_get_config)
   {
      return mi_memtrace_get_config();
   }
   return 0;
}

inline void memtrace_set_config(int value)
{
   if (mi_memtrace_set_config)
   {
      static const char* env = std::getenv("AERIAL_MEMTRACE");
      if (env != nullptr)
      {
         mi_memtrace_set_config(value);
      }
   }
}

class MemtraceDisableScope
{
public:
   MemtraceDisableScope()
   {
      prev_config = memtrace_get_config();
      memtrace_set_config(0);
   };

   ~MemtraceDisableScope()
   {
      memtrace_set_config(prev_config);
   };

private:
   int prev_config;
};

#endif // MEMTRACE_H
