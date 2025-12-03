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

import os
import numpy as np
import subprocess
from .parse_power import parse_power


def run(args, mig, mig_gpu, command, vectors, mode, target, k, actual):

    results = None

    if args.force is not None:
        if args.force == 0:
            connections = actual
        else:
            connections = args.force
    else:
        connections = np.min(
            [32, int(np.power(2, np.floor(np.log2(96 / (len(target) + 1)))))]
        )

    if args.is_power:

        if args.is_debug:

            from .configure_power_debug import configure

            system = configure(
                args, mig, mig_gpu, connections, command, vectors, mode, k, target
            )

            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

        else:

            from .configure_power import configure

            system = configure(
                args, mig, mig_gpu, connections, command, vectors, mode, k, target
            )
            ofile = None

            if not os.path.exists("power.txt"):

                if args.is_test:
                    print(
                        " ".join(
                            [
                                "nvidia-smi",
                                "-i",
                                f"{args.gpu}",
                                "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                                "-lms",
                                "10",
                                "--format=csv",
                            ]
                        )
                    )
                else:
                    ofile = open("power.txt", "w")
                    proc = subprocess.Popen(
                        [
                            "nvidia-smi",
                            "-i",
                            f"{args.gpu}",
                            "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                            "-lms",
                            "10",
                            "--format=csv",
                        ],
                        stdout=ofile,
                    )

            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                if args.is_unsafe:
                    try:
                        os.system(system)
                    finally:
                        if not args.is_save_buffers:
                            os.remove(vectors)
                else:
                    buffer = system.split(args.cfld)[0].strip().split()
                    env = {}

                    for itm in buffer:
                        mapping = itm.split("=")
                        env[mapping[0]] = mapping[1]

                    cmd = args.cfld + system.split(args.cfld)[-1].strip()
                    cmd, stdout = cmd.split(">")

                    ofile = open(stdout, "w")

                    try:
                        mproc = subprocess.Popen(cmd.split(), env=env, stdout=ofile)
                        mproc.wait(600)  # time out for 10 minutes. Note: this incldues setup and run time. It should be increased with higher cell count
                    except subprocess.TimeoutExpired:
                        print("Timeout when running power measurement. Killing subprocess")
                        mproc.kill()
                    finally:
                        ofile.close()
                        if not args.is_save_buffers:
                            os.remove(vectors)

            if ofile is not None:
                proc.kill()
                ofile.close()

                ifile = open("power.txt", "r")
                lines = ifile.readlines()
                ifile.close()
                os.remove("power.txt")

                if mig is None:
                    os.remove(f"buffer-{str(k).zfill(2)}.txt")
                else:
                    os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

                results = parse_power(lines)

    else:
        if args.is_debug:
            from .configure_debug import configure

            system = configure(
                args, mig, mig_gpu, connections, command, vectors, mode, k, target
            )
            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

        else:
            from .configure import configure

            system = configure(
                args, mig, mig_gpu, connections, command, vectors, mode, k, target
            )

            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

                if mig is None:
                    ifile = open(f"buffer-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{str(k).zfill(2)}.txt")
                else:
                    ifile = open(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

                if args.is_check_traffic:
                    from ..error import parse
                else:
                    from .sweep import parse

                results = parse(args, lines)

    return results
