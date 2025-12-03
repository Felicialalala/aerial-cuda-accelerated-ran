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

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "order_kernel_test_bench.hpp"

using namespace order_kernel_tb;

void signal_handler(int signum)
{
    if(signum == SIGINT || signum == SIGTERM)
    {
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"Signal {} received, preparing to exit..." ,signum);
    }

    exit(0);
}

void usage(const char* programName)
{
  std::cerr << "\nUsage:\n"
            << "  " << programName << " --config_file <file> --num_cells <val> --lp_file <file> --bin_files <file1> [file2...] [OPTIONS]\n\n"
            << "Required Arguments:\n"
            << "  --config_file <file>    Path to the configuration file.\n"
            << "  --num_cells   <val>     Number of cells (integer between 1 and $UL_MAX_CELLS_PER_SLOT).\n"
            << "  --lp_file     <file>    Path to the corresponding launch pattern yaml file used for evaluating test bench output.\n"
            << "  --bin_files   <file(s)> Path(s) to binary file(s). Must provide num_cells binary files.\n\n"
            << "Optional Arguments:\n"
            << "  -h, --help              Print this help message and exit.\n"
            << "  --start_slot  <val>     Start slot (default: 0).\n"
            << "  --num_slots   <val>     Number of slots (default: 1).\n"
            << "  --same_slot   <0/1>     Same slot flag (0: disabled, 1: enabled, default: 0).\n"
            << "  --num_mps_sms <val>     Enable Multi-Process Service (MPS) and set the number of SMs to use (default: 0 | disabled).\n"
            << "  --num_gc_sms  <val>     Enable Green Context (GC) mode and set the number of SMs to use (default: 0 | disabled).\n"
            << "                          Note: --num_mps_sms and --num_gc_sms cannot be used simultaneously. If both are specified, MPS will take precedence.\n"
            << "  --out_file    <file>    Path to the output file.\n\n"
            << "Example:\n"
            << "  " << programName << " --config_file ok_tb_config.txt --num_cells 4 --lp_file launch_pattern_F08_4C_59c.yam --bin_files bin1.bin bin2.bin bin3.bin bin4.bin --start_slot 0 --out_file /tmp/ok_tb_stats.csv\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if(argc < 2)
    {
        usage(argv[0]);
    }

    std::array<std::string, UL_MAX_CELLS_PER_SLOT> binaryFilePath;
    std::string configFilePath;
    std::string outputFilePath;
    std::string launchPatternFilePath;
    std::string clParams;
    std::string str_start_slot = "--start_slot";
    std::string str_num_slots = "--num_slots";
    std::string str_num_cells = "--num_cells";
    std::string str_same_slot = "--same_slot";
    std::string str_mps_sms = "--num_mps_sms";
    std::string str_gc_sms = "--num_gc_sms";
    std::string str_out_file = "--out_file";
    std::string str_config_file = "--config_file";
    std::string str_bin_files = "--bin_files";
    std::string str_lp_file = "--lp_file";
    std::string str_help_short = "-h";
    std::string str_help_long = "--help";
    std::string str_mimo = "--mimo";
    std::string str_srs = "--srs";

    uint32_t start_test_slot = 0;
    uint32_t num_test_slots = 1;
    uint32_t num_test_cells = 1;
    uint8_t  same_slot = 0;
    uint32_t mps_sms = 0;
    uint32_t gc_sms = 0;
    uint8_t  mimo = 0;
    uint8_t  srs_enabled = 0;

    int argc_proc_count = 0;
    bool config_file_provided = false;
    bool num_cells_provided = false;
    bool bin_files_provided = false;
    bool lp_file_provided = false;
    int bin_files_count = 0;

    while(1)
    {
      argc_proc_count++;
      if(argc_proc_count>(argc-1))
      {
        break;
      }
      clParams = argv[argc_proc_count];
      if(0==clParams.compare(str_help_short) || 0==clParams.compare(str_help_long))
      {
        usage(argv[0]);
      }
      else if(0==clParams.compare(str_start_slot))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        start_test_slot=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_num_slots))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        num_test_slots=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_num_cells))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        num_test_cells=std::stoi(argv[argc_proc_count]);
        num_cells_provided = true;
      }
      else if(0==clParams.compare(str_same_slot))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        same_slot=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_mps_sms))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        mps_sms=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_gc_sms))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        gc_sms=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_out_file))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        outputFilePath=argv[argc_proc_count];
      }
      else if(0==clParams.compare(str_config_file))
      {
        argc_proc_count++;
        if(argc_proc_count>(argc-1))
        {
          break;
        }
        configFilePath=argv[argc_proc_count];
        config_file_provided = true;
      }
      else if(0==clParams.compare(str_lp_file))
      {
        argc_proc_count++;
        lp_file_provided = true;
        if(argc_proc_count>(argc-1))
        {
          break;

        }
        launchPatternFilePath=argv[argc_proc_count];
      }
      else if(0==clParams.compare(str_mimo))
      {
        argc_proc_count++;
        mimo=std::stoi(argv[argc_proc_count]);
      }
      else if(0==clParams.compare(str_srs))
      {
        argc_proc_count++;
        srs_enabled=std::stoi(argv[argc_proc_count]);
      }      
      else if(0==clParams.compare(str_bin_files))
      {
        bin_files_provided = true;
        for(int cell_idx=0; cell_idx<UL_MAX_CELLS_PER_SLOT; cell_idx++)
        {
          argc_proc_count++;
          if(argc_proc_count>(argc-1))
          {
            break;
          }
          // there is no guarantee that --bin_files is the last option, check if a new option is being read
          if (argv[argc_proc_count][0] == '-')
          {
            argc_proc_count--;
            break;
          } else {
            binaryFilePath[cell_idx]=argv[argc_proc_count];
            bin_files_count++;
          }
        }
      }
      else
      {
        NVLOGE_FMT(TAG_ORDER_TB_BASE, AERIAL_TESTBENCH_EVENT, "Error: Unknown option {}", clParams);
        usage(argv[0]);
      }
    }

    // Check if mandatory fields are provided
    if (!config_file_provided || !num_cells_provided || !lp_file_provided || !bin_files_provided)
    {
      NVLOGE_FMT(TAG_ORDER_TB_BASE, AERIAL_TESTBENCH_EVENT, "Error: Missing required arguments.");
      usage(argv[0]);
    }

    // Check if num_cells matches with number of binary files
    if (num_test_cells != bin_files_count)
    {
      NVLOGE_FMT(TAG_ORDER_TB_BASE, AERIAL_TESTBENCH_EVENT, "Error: Number of binary files {} does not match num_cells {}", bin_files_count, num_test_cells);
      usage(argv[0]);
    }

    NVLOGI_FMT(TAG_ORDER_TB_BASE, "Test configs: start_test_slot {}, num_test_slots {}, num_test_cells {}, same_slot {}, \n"
                                  "num_mps_sms {}, num_gc_sms {}, mimo {}, srs_enabled {}",start_test_slot,num_test_slots,num_test_cells,same_slot,mps_sms, gc_sms, mimo, srs_enabled);
    char        yaml_file[1024];
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(yaml_file, relative_path.c_str());
    pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_file, "order_kernel_tb.log", NULL);

    OrderKernelTestBench testbench(configFilePath,binaryFilePath,launchPatternFilePath,outputFilePath,start_test_slot,num_test_slots,num_test_cells,same_slot,mps_sms, gc_sms, mimo, srs_enabled);
    int ret = testbench.run_test();

    return ret;
}
