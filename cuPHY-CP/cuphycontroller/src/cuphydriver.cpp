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

#include "cuphydriver.hpp"
#include <sstream>
#include "nvlog.hpp"
#include "cuphyoam.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 3) // "CTL.DRV"

//!
// \brief Create PhyDriver context
//
// \param filename for input yaml file
// @return YP_OK on success, YP_ERR code otherwise
// tries to take the yaml input yaml file and parse the params.
// Stores the params in member variables.
//
int pc_init_phydriver(
                phydriver_handle * pdh,
                const context_config& ctx_cfg,
                std::vector<phydriverwrk_handle>& workers_descr
            )
{
    if(ctx_cfg.ul_cores.size() == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No UL worker provided");
        return EINVAL;
    }

    if(ctx_cfg.dl_cores.size() == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No DL worker provided");
        return EINVAL;
    }

    if(workers_descr.size() == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No worker descriptor provided");
        return EINVAL;
    }

    if(l1_init(pdh, ctx_cfg))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_init");
        return -1;
    }

    CuphyOAM *oam = CuphyOAM::getInstance();
    if (oam->init_everything())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error oam init");
        return -1;
    }

    return 0;
}

int pc_finalize_phydriver(phydriver_handle pdh) {

    if(l1_finalize(pdh))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_finalize");
        return -1;
    }

    return 0;
}

int pc_standalone_create_cells(phydriver_handle pdh, std::vector<cell_phy_info>& cell_configs) {

    struct cell_phy_info cell;
    char * pch;
    int i=0, tmp=0;

    for(auto c : cell_configs)
    {
        NVLOGC_FMT(TAG, "Creating new cell {}", c.name.c_str());
        if(l1_cell_create(pdh, c))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_cell_create {}", c.phy_stat.phyCellId);
            return -1;
        }

        if(l1_cell_start(pdh, c.phy_stat.phyCellId))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_cell_start {}", c.phy_stat.phyCellId);
            return -1;
        }
    }

    return 0;
}

struct l2_adapter_args {
    std::vector<struct slot_command_api::slot_command *> scl;
    int usec;
    int num_slots;
};

static uint64_t get_ns(void) {
    struct timespec t;
    int ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if (ret != 0) {
        NVLOGE_FMT(TAG, AERIAL_CLOCK_API_EVENT, "clock_gettime failed");
    }
    return (uint64_t) t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

static void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_ns() + ns, start_t = 0;
    while ((start_t = get_ns()) < end_t) {
        for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
            __asm__ __volatile__ ("");
        }
    }
}

int pc_l2_adapter(phydriverwrk_handle wh, void *arg) {
    int ret = 0, slot_index = 0, s = 0;
    bool increment = true;
    uint8_t sfn = 0, slot = 0;
    struct l2_adapter_args * l2args = (struct l2_adapter_args *)arg;
    phydriver_handle pd_h = l1_worker_get_phydriver_handler(wh);

    /*
     * Infinite slot
     */
    if(l2args->num_slots == 0)
    {
        l2args->num_slots = 1;
        increment = false;
    }

    while(l1_worker_check_exit(wh) == false)
    {
        while(s < l2args->num_slots)
        {
            for(auto& c : (l2args->scl[slot_index]->cells))
            {
                c.slot.slot_3gpp.sfn_ = sfn;
                c.slot.slot_3gpp.slot_ = slot;
                NVLOGC_FMT(TAG, "Starting iteration {} slot {} cell {} slot {} sfn {}", s, slot_index, c.cell, c.slot.slot_3gpp.slot_, c.slot.slot_3gpp.sfn_);
            }

            l2args->scl[slot_index]->tick_original = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
            ret = l1_enqueue_phy_work(pd_h, l2args->scl[slot_index]);
            if(ret)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_enqueue_phy_work returned error {}", ret);
                goto quit;
            }

            /*
            * TBD: better time management here is required!
            */
            wait_ns(l2args->usec-2 * 1000);
            slot_index = (slot_index+1)%((int)l2args->scl.size());
            sfn = (sfn+1)%256;
            slot = (slot+1)%20;

            if(increment == true) s++;
        }
    }
quit:
    return 0;
}

int pc_start_l1(phydriver_handle pdh) {
    // if(l1_start(pdh))
    // {
    //     NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_start");
    //     return -1;
    // }

    return 0;
}

int pc_standalone_simulate_l2(phydriver_handle pdh, int usec, int num_slots, std::vector<struct slot_command_api::slot_command *> scl, int core, uint32_t workers_sched_priority)
{
    phydriverwrk_handle wh1;
    struct l2_adapter_args l2args;

    NVLOGC_FMT(TAG, "num slots = {}", num_slots);
    l2args.num_slots = num_slots;
    l2args.usec = usec;
    l2args.scl = scl;

    if(l1_worker_start_generic(pdh, &wh1, "L2Adapter", core, workers_sched_priority, pc_l2_adapter, &l2args)) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_worker_start_generic");
        exit(EXIT_FAILURE);
    }

    NVLOGC_FMT(TAG, "Waiting for 15 secs of execution....");
    wait_ns(15000000000); //15 sec

    if(l1_worker_stop(wh1)) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error l1_worker_stop");
        exit(EXIT_FAILURE);
    }
    return 0;
}
