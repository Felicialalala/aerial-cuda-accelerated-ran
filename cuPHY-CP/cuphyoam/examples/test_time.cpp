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

#include <stdio.h>
#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_errno.h>
#include <chrono>

using t_ns = std::chrono::nanoseconds;
using namespace std::literals;

static int
lcore_test_time(__rte_unused void *arg)
{
   unsigned lcore_id;
   lcore_id = rte_lcore_id();
   int count = 0;
   const int NUM_ITERATIONS = 1000000000;
   printf("hello from core %u\n", lcore_id);

   if (lcore_id == 7)
   {
      t_ns t_start;
      t_ns t_next;
      t_ns t_end;
      t_ns t_max = 0s;

      t_start = std::chrono::system_clock::now().time_since_epoch();
      t_next = std::chrono::system_clock::now().time_since_epoch();
      for (int k=0; k<NUM_ITERATIONS; k++)
      {
         t_ns t_delta;
         t_end = std::chrono::system_clock::now().time_since_epoch();
         t_delta = t_end - t_next;
         if (t_delta > t_max)
         {
            t_max = t_delta;
         }
         t_next = t_end;
      }
      printf("max loop delta time using std::chrono::system_clock::now().time_since_epoch() %ld ns\n",t_max.count());
      t_ns t_delta = t_end - t_start;
      printf("avg loop delta time using std::chrono::system_clock::now().time_since_epoch() %lf ns\n",static_cast<double>(t_delta.count())/NUM_ITERATIONS);
   }
   else if (lcore_id == 8)
   {
      uint64_t t_start;
      uint64_t t_next;
      uint64_t t_end;
      uint64_t t_max = 0;

      t_start = rte_get_tsc_cycles();
      t_next = rte_get_tsc_cycles();
      for (int k=0; k<NUM_ITERATIONS; k++)
      {
         uint64_t t_delta;
         t_end = rte_get_tsc_cycles();
         t_delta = t_end - t_next;
         if (t_delta > t_max)
         {
            t_max = t_delta;
         }
         t_next = t_end;
      }
      printf("max loop delta time using rte_get_tsc_cycles() %lf ns\n",static_cast<double>(t_max)/rte_get_timer_hz() * 1000000000);
      uint64_t t_delta = t_end - t_start;
      printf("avg loop delta time using rte_get_tsc_cycles() %lf ns\n",static_cast<double>(t_delta)/rte_get_timer_hz()/NUM_ITERATIONS * 1000000000);
   }
   return 0;
}

int main(int argc, char** argv) {
   int ret;
   unsigned lcore_id;

   ret = rte_eal_init(argc, argv);
   if (ret < 0) rte_panic("Cannot init EAL\n");

   RTE_LCORE_FOREACH_WORKER(lcore_id)
   {
      rte_eal_remote_launch(lcore_test_time, NULL, lcore_id);
   }

   rte_eal_mp_wait_lcore();

   return 0;
}
