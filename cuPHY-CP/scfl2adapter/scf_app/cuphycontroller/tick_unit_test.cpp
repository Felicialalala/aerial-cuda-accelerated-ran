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

#include "nv_phy_group.hpp"
#include "yaml.hpp"

// #include "l2_adapter_api.hpp"
#include "scf_5g_fapi.hpp"
#include "nv_tick_generator.hpp"
#include "nv_phy_driver_proxy.hpp"

#define TAG "SCF.TICK_TEST"

#define YAML_CONFIG_FILE "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/tick_sleep_mode.yaml"

using namespace std;
using namespace nv;

void usage()
{
    NVLOGI_FMT(TAG, "Usage: l2_adapter_cuphycontroller_scf <file>.yaml");
}

// std::unique_ptr<context> l2_api;

int main(int argc, const char* argv[])
{
    int return_value = 0;
    try
    {
        // Relative path of this process is build/cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/
        char yaml_file[1024];
        std::string relative_path = std::string("../../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        nv_get_absolute_path(yaml_file, relative_path.c_str());

        pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file, "tick_unit_test.log",NULL);
        nvlog_fmtlog_thread_init();
        NVLOGC_FMT(TAG, "Thread {} initialized fmtlog", __FUNCTION__);

        //--------------------------------------------------------------
        // Initialize an L1 group from the given configuration file
       scf_5g_fapi::init();

       // Parse configuration yaml file path from command line arguments if exist
       if(argc >= 2)
       {
           nvlog_safe_strncpy(yaml_file, argv[1], 1024);
       }
       else
       {
           relative_path = std::string("../../../../../").append(YAML_CONFIG_FILE);
           nv_get_absolute_path(yaml_file, relative_path.c_str());
       }

       nv::PHY_group grp(yaml_file);

        // l2_api = init(argv[1]);
        nv::PHYDriverProxy::make();
        // start(*l2_api);
        grp.start();

        //nv::thread_config config = {"l2adapter_main", 19, 95}; // Hard code for now
        //nv::tti_gen tti_handler(config, l2_api.get());
        //tti_handler.start_slot_indication_thread();
        //config_thread_property(config);
       //--------------------------------------------------------------
       // Run until all group members complete
       // join(*l2_api);
        grp.join();
    }
    catch(std::exception& e)
    {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return_value = 1;
    }
    catch(...)
    {
        fprintf(stderr, "UNKNOWN EXCEPTION\n");
        return_value = 2;
    }
    return return_value;
}
