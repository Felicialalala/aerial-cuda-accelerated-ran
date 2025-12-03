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

#include "cuphyoam.hpp"

#ifdef CUPHYOAM_WIP
static int
lcore_hello(__rte_unused void *arg)
{
   unsigned lcore_id;
   lcore_id = rte_lcore_id();
   int count = 0;
   printf("hello from core %u\n", lcore_id);

   CuphyOAM* oam = CuphyOAM::getInstance();

   int primary_lcore_id;
   for (primary_lcore_id = 0; primary_lcore_id < RTE_MAX_LCORE; primary_lcore_id++)
   {
      if (rte_lcore_is_enabled(primary_lcore_id)) break;
   }

   if (lcore_id == primary_lcore_id)
   {
      int ret;
      char *msg;
      cuphyoam_pool_t *p = oam->getCtrlPool();

      for (int k=0; k<RTE_MAX_LCORE; k++)
      {
         if (k == primary_lcore_id) continue;
         if (!rte_lcore_is_enabled(k)) continue;

         ret = cuphyoam_msg_alloc(p, reinterpret_cast<void**>(&msg));
         if (ret != 0)
         {
            printf("cuphyoam_msg_alloc: %d\n",ret);
            return -1;
         }

         cuphyoam_queue_t *q = oam->getCtrlQueue(k, true);
         sprintf(msg,"Hello from core %u sent to core %d",lcore_id,k);
         ret = cuphyoam_enqueue(q, msg);
         if (ret != 0)
         {
            printf("cuphyoam_enqueue: %d\n",ret);
         }
         count++;
      }

      while (count > 0)
      {
         for (int k=0; k<MAX_CORES; k++)
         {
            if (k == primary_lcore_id) continue;
            if (!rte_lcore_is_enabled(k)) continue;

            cuphyoam_queue_t *q_tx = oam->getCtrlQueue(k, false);
            if (!cuphyoam_queue_isEmpty(q_tx))
            {
               ret = cuphyoam_dequeue(q_tx, reinterpret_cast<void**>(&msg));
               if (ret != 0)
               {
                  printf("cuphyoam_dequeue: %d\n",ret);
               }
               printf("[%d] Core %u Received msg: %s\n",count,lcore_id,msg);
               cuphyoam_msg_free(p, reinterpret_cast<void*>(msg));
               count--;
            }
         }
      }
   }
   else
   {
      int ret;
      char *msg;
      cuphyoam_queue_t *q_rx = oam->getCtrlQueue(lcore_id, true);
      cuphyoam_queue_t *q_tx = oam->getCtrlQueue(lcore_id, false);

      while (cuphyoam_queue_isEmpty(q_rx));

      ret = cuphyoam_dequeue(q_rx, reinterpret_cast<void**>(&msg));
      if (ret != 0)
      {
         printf("cuphyoam_dequeue: %d\n",ret);
      }
      printf("Core %u Received msg: %s\n",lcore_id,msg);

      sprintf(msg,"Response from core %d",lcore_id);
      ret = cuphyoam_enqueue(q_tx, msg);
      if (ret != 0)
      {
         printf("cuphyoam_enqueue: %d\n",ret);
      }
   }
   return 0;
}
#endif

int main(int argc, char** argv) {
   int ret;
   unsigned lcore_id;
   CuphyOAM* oam = CuphyOAM::getInstance();

#ifdef CUPHYOAM_WIP
   ret = rte_eal_init(argc, argv);
   if (ret < 0) rte_panic("Cannot init EAL\n");

   oam->init_everything();

   RTE_LCORE_FOREACH_WORKER(lcore_id)
   {
      rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
   }

   lcore_hello(NULL);

   rte_eal_mp_wait_lcore();

   oam->shutdown();
   oam->wait_shutdown();
#else
   printf("No functionality, enable CUPHYOAM_WIP\n");
#endif

   return 0;
}
