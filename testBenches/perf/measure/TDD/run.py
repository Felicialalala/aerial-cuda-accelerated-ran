# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import os
import uuid
import sys

from .traffic import traffic_avg, traffic_het
from .execute import run
from .check_cell_capacity import check_cell_capacity

def run_TDD(args, sms, mig=None):

    ifile = open(args.config)
    config = json.load(ifile)
    ifile.close()

    ifile = open(args.uc)
    uc = json.load(ifile)
    ifile.close()

    if args.is_graph:
        output = "sweep_graphs_" + args.uc.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 1
    else:
        output = "sweep_streams_" + args.uc.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 0

    data_targets = None
    target = None

    if not args.is_no_mps:
        if args.target is not None:

            if len(args.target) == 1:

                buffer_target = args.target[0]

                if buffer_target.isnumeric():
                    target = []
                    target.append(np.min([int(buffer_target), sms]))
                    if(int(buffer_target) > sms):
                        print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    
                    output = buffer_target.zfill(3) + "_" + output
                else:
                    ifile = open(buffer_target, "r")
                    data_targets = json.load(ifile)
                    ifile.close()

            else:

                target = []

                for buffer_target in args.target:

                    if buffer_target.isnumeric():
                        target.append(np.min([int(buffer_target), sms]))
                        if(int(buffer_target) > sms):
                            print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    else:
                        raise ValueError

                buffer = [x.zfill(3) for x in args.target]
                output = "_".join(buffer) + "_" + output
        else:
            raise NotImplementedError

    if mig is not None:
        mig_gpu = mig.replace("/", "-")
        output = output + "_" + mig_gpu
    else:
        mig_gpu = None

    if args.seed is not None:
        output = output + "_s" + str(args.seed) + "_" + str(uuid.uuid4())

    if not args.is_no_mps:
        if mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=."
        else:
            os.mkdir(mig_gpu)
            if args.is_test:
                print(f"Created: {mig_gpu}")

            system = f"CUDA_VISIBLE_DEVICES={mig} CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu}"

        # only enable MPS if not running in green contexts mode or if it was explicitly enabled; terminate otherwise.
        if not args.is_use_green_contexts:
            system = " ".join([system, "nvidia-cuda-mps-control -d"])
        elif args.is_enable_mps_for_green_contexts:
            system = " ".join([system, "nvidia-cuda-mps-control -d"])
        else:
            # If there is no MPS running to terminate, the following command will show an informative "Cannot find MPS control daemon process" message.
            system = f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control"
            system = " ".join([system])

        if args.is_test:
            if args.debug_mode not in ["ncu"]:
                print(system)

        if args.debug_mode not in ["ncu"]:
            os.system(system)

    k = args.start

    sweeps = {}
    powers = {}

    # save test configs and GPU product name
    if args.is_power:
        powers['testConfig'] = vars(args)
        powers['testConfig']['smAllocation'] = str(target) 
    else:
        sweeps['testConfig'] = vars(args)
        sweeps['testConfig']['smAllocation'] = str(target)
    
    if args.seed is not None:
        np.random.seed(args.seed)

    command = os.path.join(args.cfld, "cubb_gpu_test_bench/cubb_gpu_test_bench")

    while k <= args.cap:

        if mig is None:
            vectors = os.path.join(os.getcwd(), "vectors-" + str(k).zfill(2) + ".yaml")
        else:
            vectors = os.path.join(
                os.getcwd(), "vectors-" + mig_gpu + "-" + str(k).zfill(2) + ".yaml"
            )

        if "_het_" in args.uc:

            if not args.is_no_mps and data_targets is not None:

                key = str(k).zfill(2)
                buffer_target = data_targets.get(key, None)

                if buffer_target is None:
                    k += 1
                    continue
                else:
                    if type(buffer_target) == int:
                        target = []
                        target.append(np.min([int(buffer_target), sms]))
                        if(int(buffer_target) > sms):
                            print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    elif type(buffer_target) == list:

                        target = []

                        for itm in buffer_target:
                            target.append(np.min([int(buffer_target), sms]))
                            if(int(buffer_target) > sms):
                                print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                                
            uc_keys = list(uc.keys())

            uc_dl = [x for x in uc_keys if "PDSCH" in x]

            if len(uc_dl) != 1:
                sys.exit("error: use case file exhibits an unexpected structure")
            else:
                uc_dl = uc_dl[0]

            uc_ul = [x for x in uc_keys if "PUSCH" in x]

            if len(uc_ul) != 1:
                sys.exit("error: use case file exhibits an unexpected structure")
            else:
                uc_ul = uc_ul[0]

            testcases_dl = uc[uc_dl]
            testcases_ul = uc[uc_ul]

            filenames_dl = config[uc_dl]
            filenames_ul = config[uc_ul]

            message = "Number of active cells: " + str(k)

            if target is not None:
                message += "(" + ",".join(list(map(str, target))) + ")"

            print(message)

            traffic_het(
                args,
                vectors,
                k,
                (testcases_dl, testcases_ul),
                (filenames_dl, filenames_ul),
            )

            if args.is_power:
                powers[str(k).zfill(2)] = run(
                    args, mig, mig_gpu, command, vectors, mode, target, k, k
                )
            else:
                sweeps[str(k).zfill(2)] = run(
                    args, mig, mig_gpu, command, vectors, mode, target, k, k
                )

        elif "_avg_":

            interval = uc["Peak: " + str(k)]

            for subcase in interval.keys():

                uc_keys = list(interval[subcase].keys())

                uc_dl = [x for x in uc_keys if "PDSCH" in x]

                if len(uc_dl) != 1:
                    sys.exit(
                        "error: use case file exhibits an unexpected struture (PDSCH)"
                    )
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit(
                        "error: use case file exhibits an unexpected struture (PUSCH)"
                    )
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = interval[subcase][uc_dl]
                testcases_ul = interval[subcase][uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                testcases_dlbf = None
                testcases_ulbf = None
                filenames_dlbf = None
                filenames_ulbf = None
                testcases_sr = None
                filenames_sr = None
                testcases_ra = None
                filenames_ra = None
                testcases_cdl = None
                filenames_cdl = None
                testcases_cul = None
                filenames_cul = None
                testcases_ssb = None
                filenames_ssb = None
                testcases_cr = None
                filenames_cr = None
                testcases_mac = None
                filenames_mac = None
                testcases_mac2 = None
                filenames_mac2 = None

                if args.is_rec_bf:
                    uc_dlbf = [x for x in uc_keys if "DLBFW" in x]
                    if len(uc_dlbf) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (DL BFW)"
                        )
                    else:
                        uc_dlbf = uc_dlbf[0]

                    testcases_dlbf = interval[subcase][uc_dlbf]
                    filenames_dlbf = config[uc_dlbf]

                    uc_ulbf = [x for x in uc_keys if "ULBFW" in x]
                    if len(uc_ulbf) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (UL BFW)"
                        )
                    else:
                        uc_ulbf = uc_ulbf[0]

                    testcases_ulbf = interval[subcase][uc_ulbf]
                    filenames_ulbf = config[uc_ulbf]
                    
                    uc_sr = [x for x in uc_keys if "SRS" in x]
                    if len(uc_sr) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (SRS)"
                        )
                    else:
                        uc_sr = uc_sr[0]

                    testcases_sr = interval[subcase][uc_sr]
                    filenames_sr = config[uc_sr]

                if args.is_prach:
                    uc_ra = [x for x in uc_keys if "PRACH" in x]
                    if len(uc_ra) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PRACH)"
                        )
                    else:
                        uc_ra = uc_ra[0]

                    testcases_ra = interval[subcase][uc_ra]
                    filenames_ra = config[uc_ra]

                if args.is_pdcch:
                    uc_cdl = [x for x in uc_keys if "PDCCH" in x]
                    if len(uc_cdl) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PDCCH)"
                        )
                    else:
                        uc_cdl = uc_cdl[0]

                    testcases_cdl = interval[subcase][uc_cdl]
                    filenames_cdl = config[uc_cdl]

                if args.is_ssb:
                    uc_ssb = [x for x in uc_keys if "SSB" in x]
                    if len(uc_ssb) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (SSB)"
                        )
                    else:
                        uc_ssb = uc_ssb[0]

                    testcases_ssb = interval[subcase][uc_ssb]
                    filenames_ssb = config[uc_ssb]

                if args.is_csirs:
                    uc_cr = [x for x in uc_keys if "CSIRS" in x]
                    if len(uc_cr) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (CSIRS)"
                        )
                    else:
                        uc_cr = uc_cr[0]

                    testcases_cr = interval[subcase][uc_cr]
                    filenames_cr = config[uc_cr]

                if args.is_pucch:
                    uc_cul = [x for x in uc_keys if "PUCCH" in x]
                    if len(uc_cul) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PUCCH)"
                        )
                    else:
                        uc_cul = uc_cul[0]
                    testcases_cul = interval[subcase][uc_cul]
                    filenames_cul = config[uc_cul]

                if args.is_mac:
                    uc_mac = [x for x in uc_keys if "MAC" == x[-3:]] # 'FXX - MAC'
                    if len(uc_mac) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (MAC)"
                        )
                    else:
                        uc_mac = uc_mac[0]

                    testcases_mac = interval[subcase][uc_mac]
                    filenames_mac = config[uc_mac]

                if args.mac2 > 0:
                    uc_mac2 = [x for x in uc_keys if "MAC2" in x]
                    if len(uc_mac2) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (MAC2)"
                        )
                    else:
                        uc_mac2 = uc_mac2[0]

                    testcases_mac2 = interval[subcase][uc_mac2]
                    filenames_mac2 = config[uc_mac2]

                label = int(subcase.replace("Average: ", ""))

                if not args.is_no_mps and data_targets is not None:

                    key = "+".join([str(k).zfill(2), str(label).zfill(2)])

                    buffer_target = data_targets.get(key, None)

                    if buffer_target is None:
                        k += 1
                        continue
                    else:
                        if type(buffer_target) == int:
                            target = []
                            target.append(np.min([int(buffer_target), sms]))
                            if(int(buffer_target) > sms):
                                print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                        elif type(buffer_target) == list:

                            target = []

                            for itm in buffer_target:
                                target.append(np.min([int(buffer_target), sms]))
                                if(int(buffer_target) > sms):
                                    print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                                    
                message = "Number of active cells: " + str(k) + "+" + str(label)

                if target is not None:
                    message += "(" + ",".join(list(map(str, target))) + ")"

                print(message)

                if args.pattern == "dddsu" and args.is_mac == True:
                    sys.exit(
                        "error: cuMAC run with dddsu pattern not supported"
                    )
                elif args.pattern == "dddsu" and args.is_mac == True:
                    sys.exit(
                        "error: cuMAC2 run with dddsu pattern not supported"
                    )
                else: # no need to include mac and mac2
                    traffic_avg(
                        args,
                        vectors,
                        (
                            testcases_dl,
                            testcases_ul,
                            testcases_dlbf,
                            testcases_ulbf,
                            testcases_sr,
                            testcases_ra,
                            testcases_cdl,
                            testcases_cul,
                            testcases_ssb,
                            testcases_cr,
                            testcases_mac,
                            testcases_mac2,
                        ),
                        (
                            filenames_dl,
                            filenames_ul,
                            filenames_dlbf,
                            filenames_ulbf,
                            filenames_sr,
                            filenames_ra,
                            filenames_cdl,
                            filenames_cul,
                            filenames_ssb,
                            filenames_cr,
                            filenames_mac,
                            filenames_mac2,
                        ),
                    )

                if args.is_power:
                    powers["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                        args,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                        len(testcases_dl),
                    )
                else:
                    sweeps["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                        args,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                        len(testcases_dl),
                    )
        else:
            raise NotImplementedError

        k += args.step_size

    if args.is_power:
        if len(list(powers.keys())) > 0:
            ofile = open(output.replace("sweep", "power") + ".json", "w")
            json.dump(powers, ofile, indent=2)
            ofile.close()
    else:
        if not args.is_debug:
            if args.is_check_traffic:
                output_file = output.replace("sweep", "error") + ".json"
            else:
                output_file = output + ".json"
            
            # auto dectect cell capacity for F08, F09, F14 of TDD long pattern (dddsuudddd)
            # the max cell count that all channels pass (within latency threshold) will be the cell capacity
            # a warning of unknown cell capacity will be given if all tested cell counts pass or none passes
            if not args.is_test:
                check_cell_capacity(sweeps)
            
                ofile = open(output_file, "w")
                json.dump(sweeps, ofile, indent=2)
                ofile.close()
