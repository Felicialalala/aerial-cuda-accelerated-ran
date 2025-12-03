# cuBB system checks

Script for dumping system configuration relevant for running cuBB.

The goal of `cuBB_system_checks.py` is to verify that a platform on which the script is executed is correctly configured to run the cuBB software stack.

`cuBB_system_checks.py` provides a standardized view of the system which makes finding any misconfigurations easier.


## Example commands

Show help message:
```
./cuBB_system_checks.py -h
```

Dump cuBB-related system configuration:
```
sudo -E ./cuBB_system_checks.py -bcegilmnps
```

Dump NIC configuration only:
```
sudo -E ./cuBB_system_checks.py --nic
```

Dump host configurations from a container via SSH:
```
./cuBB_system_checks --host XXX.XXX.XXX.XXX --username YYYY
```

Dump configurations and version on a RHOCP cluster (assuming that `oc` command is executable):
```
./cuBB_system_checks --cli oc
```

## Reference setup

Reference configuration for 24-3 Aerial release:
```
-----General--------------------------------------
Hostname                           : aerial-64tr-smc
IP address                         : 10.32.221.21
Linux distro                       : "Ubuntu 22.04.5 LTS"
Linux kernel version               : 6.5.0-1019-nvidia
-----Kernel Command Line--------------------------
Audit subsystem                    : audit=0
Clock source                       : N/A
HugePage count                     : hugepages=48
HugePage size                      : hugepagesz=512M
CPU idle time management           : idle=poll
Max Intel C-state                  : N/A
Intel IOMMU                        : N/A
IOMMU                              : N/A
Isolated CPUs                      : N/A
Corrected errors                   : N/A
Adaptive-tick CPUs                 : nohz_full=4-64
Soft-lockup detector disable       : nosoftlockup
Max processor C-state              : processor.max_cstate=0
RCU callback polling               : rcu_nocb_poll
No-RCU-callback CPUs               : rcu_nocbs=4-64
TSC stability checks               : tsc=reliable
-----CPU------------------------------------------
CPU cores                          : 72
Thread(s) per CPU core             : 1
CPU MHz:                           : N/A
CPU sockets                        : 1
-----Environment variables------------------------
CUDA_DEVICE_MAX_CONNECTIONS        : N/A
cuBB_SDK                           : N/A
-----Memory---------------------------------------
HugePage count                     : 48
Free HugePages                     : 31
HugePage size                      : 524288 kB
Shared memory size                 : 240G
-----Nvidia GPUs----------------------------------
GPU driver version                 : 560.35.03
CUDA version                       : 12.6
GPU0
  GPU product name                 : NVIDIA GH200 480GB
  GPU persistence mode             : Enabled
  Current GPU temperature          : 33 C
  GPU clock frequency              : 1980 MHz
  Max GPU clock frequency          : 1980 MHz
  GPU PCIe bus id                  : 00000009:01:00.0
-----GPUDirect topology---------------------------
        GPU0    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    NODE    NODE    NODE    0-71    0               1
NIC0    NODE     X      PIX     NODE    NODE
NIC1    NODE    PIX      X      NODE    NODE
NIC2    NODE    NODE    NODE     X      PIX
NIC3    NODE    NODE    NODE    PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3


-----Mellanox NICs--------------------------------
NIC0
  NIC product name                 : BlueField3
  NIC part number                  : 900-9D3B6-00CV-A_Ax
  NIC PCIe bus id                  : /dev/mst/mt41692_pciconf1
  NIC FW version                   : 32.41.1000
  FLEX_PARSER_PROFILE_ENABLE       : 4
  PROG_PARSE_GRAPH                 : True(1)
  ACCURATE_TX_SCHEDULER            : True(1)
  CQE_COMPRESSION                  : AGGRESSIVE(1)
  REAL_TIME_CLOCK_ENABLE           : True(1)
NIC1
  NIC product name                 : BlueField3
  NIC part number                  : 900-9D3B6-00CV-A_Ax
  NIC PCIe bus id                  : /dev/mst/mt41692_pciconf0
  NIC FW version                   : 32.41.1000
  FLEX_PARSER_PROFILE_ENABLE       : 4
  PROG_PARSE_GRAPH                 : True(1)
  ACCURATE_TX_SCHEDULER            : True(1)
  CQE_COMPRESSION                  : AGGRESSIVE(1)
  REAL_TIME_CLOCK_ENABLE           : True(1)
-----Mellanox NIC Interfaces----------------------
Interface0
  Name                             : aerial00
  Network adapter                  : mlx5_0
  PCIe bus id                      : 0000:01:00.0
  Ethernet address                 : 58:a2:e1:6a:2d:c6
  Operstate                        : up
  MTU                              : 1514
  RX flow control                  : off
  TX flow control                  : off
  PTP hardware clock               : 0
  QoS Priority trust state         : pcp
  PCIe MRRS                        : 4096 bytes
Interface1
  Name                             : aerial01
  Network adapter                  : mlx5_1
  PCIe bus id                      : 0000:01:00.1
  Ethernet address                 : 58:a2:e1:6a:2d:c7
  Operstate                        : up
  MTU                              : 1500
  RX flow control                  : off
  TX flow control                  : off
  PTP hardware clock               : 1
  QoS Priority trust state         : pcp
  PCIe MRRS                        : 512 bytes
Interface2
  Name                             : aerial02
  Network adapter                  : mlx5_2
  PCIe bus id                      : 0002:01:00.0
  Ethernet address                 : 9c:63:c0:3c:44:04
  Operstate                        : up
  MTU                              : 1500
  RX flow control                  : on
  TX flow control                  : on
  PTP hardware clock               : 2
  QoS Priority trust state         : pcp
  PCIe MRRS                        : 512 bytes
Interface3
  Name                             : aerial03
  Network adapter                  : mlx5_3
  PCIe bus id                      : 0002:01:00.1
  Ethernet address                 : 9c:63:c0:3c:44:05
  Operstate                        : up
  MTU                              : 1500
  RX flow control                  : on
  TX flow control                  : on
  PTP hardware clock               : 3
  QoS Priority trust state         : pcp
  PCIe MRRS                        : 512 bytes
-----Linux PTP------------------------------------
● ptp4l.service - Precision Time Protocol (PTP) service
     Loaded: loaded (/lib/systemd/system/ptp4l.service; enabled; vendor preset: enabled)
     Active: active (running) since Fri 2024-11-08 04:36:38 UTC; 6 days ago
       Docs: man:ptp4l
    Process: 1777 ExecStartPre=ifconfig aerial00 up (code=exited, status=0/SUCCESS)
    Process: 3517 ExecStartPre=ethtool --set-priv-flags aerial00 tx_port_ts on (code=exited, status=0/SUCCESS)
    Process: 4118 ExecStartPre=ethtool -A aerial00 rx off tx off (code=exited, status=0/SUCCESS)
    Process: 4179 ExecStartPre=ifconfig aerial01 up (code=exited, status=0/SUCCESS)
    Process: 4204 ExecStartPre=ethtool --set-priv-flags aerial01 tx_port_ts on (code=exited, status=0/SUCCESS)
    Process: 4250 ExecStartPre=ethtool -A aerial01 rx off tx off (code=exited, status=0/SUCCESS)
   Main PID: 4275 (ptp4l)
      Tasks: 1 (limit: 146900)
     Memory: 3.3M
        CPU: 7min 24.783s
     CGroup: /system.slice/ptp4l.service
             └─4275 /usr/sbin/ptp4l -f /etc/ptp.conf

Nov 14 05:43:12 aerial-64tr-smc ptp4l[4275]: [522416.958] rms    3 max    6 freq -10096 +/-  14 delay   -10 +/-   1
Nov 14 05:43:13 aerial-64tr-smc ptp4l[4275]: [522417.959] rms    4 max    8 freq -10105 +/-  19 delay   -10 +/-   0
Nov 14 05:43:14 aerial-64tr-smc ptp4l[4275]: [522418.959] rms    5 max   13 freq -10111 +/-  21 delay    -9 +/-   1
Nov 14 05:43:15 aerial-64tr-smc ptp4l[4275]: [522419.959] rms    5 max    8 freq -10097 +/-  22 delay    -9 +/-   1
Nov 14 05:43:16 aerial-64tr-smc ptp4l[4275]: [522420.959] rms    5 max   13 freq -10108 +/-  25 delay    -9 +/-   1
Nov 14 05:43:17 aerial-64tr-smc ptp4l[4275]: [522421.960] rms    5 max   11 freq -10093 +/-  23 delay    -9 +/-   0
Nov 14 05:43:18 aerial-64tr-smc ptp4l[4275]: [522422.960] rms    4 max    9 freq -10102 +/-  19 delay    -9 +/-   0
Nov 14 05:43:19 aerial-64tr-smc ptp4l[4275]: [522423.960] rms    3 max    7 freq -10111 +/-  12 delay    -9 +/-   0
Nov 14 05:43:20 aerial-64tr-smc ptp4l[4275]: [522424.960] rms    4 max    9 freq -10100 +/-  19 delay    -9 +/-   0
Nov 14 05:43:21 aerial-64tr-smc ptp4l[4275]: [522425.961] rms    5 max   13 freq -10106 +/-  23 delay    -9 +/-   0

● phc2sys.service - Synchronize system clock or PTP hardware clock (PHC)
     Loaded: loaded (/lib/systemd/system/phc2sys.service; enabled; vendor preset: enabled)
     Active: active (running) since Fri 2024-11-08 04:36:41 UTC; 6 days ago
       Docs: man:phc2sys
    Process: 4281 ExecStartPre=sleep 2 (code=exited, status=0/SUCCESS)
   Main PID: 4385 (sh)
      Tasks: 2 (limit: 146900)
     Memory: 6.0M
        CPU: 29min 8.815s
     CGroup: /system.slice/phc2sys.service
             ├─4385 /bin/sh -c "taskset -c 41 /usr/sbin/phc2sys -s /dev/ptp\$(ethtool -T aerial00 | grep PTP | awk '{print \$4}') -c CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256"
             └─4390 /usr/sbin/phc2sys -s /dev/ptp0 -c CLOCK_REALTIME -n 24 -O 0 -R 256 -u 256

Nov 14 05:43:12 aerial-64tr-smc phc2sys[4390]: [522416.884] CLOCK_REALTIME rms   13 max   29 freq  -2504 +/-  91 delay   513 +/-   7
Nov 14 05:43:13 aerial-64tr-smc phc2sys[4390]: [522417.900] CLOCK_REALTIME rms   13 max   29 freq  -2483 +/-  83 delay   513 +/-   6
Nov 14 05:43:14 aerial-64tr-smc phc2sys[4390]: [522418.916] CLOCK_REALTIME rms   10 max   24 freq  -2484 +/-  51 delay   513 +/-   7
Nov 14 05:43:15 aerial-64tr-smc phc2sys[4390]: [522419.933] CLOCK_REALTIME rms   12 max   26 freq  -2497 +/-  66 delay   513 +/-   7
Nov 14 05:43:16 aerial-64tr-smc phc2sys[4390]: [522420.949] CLOCK_REALTIME rms   12 max   27 freq  -2480 +/-  58 delay   513 +/-   7
Nov 14 05:43:17 aerial-64tr-smc phc2sys[4390]: [522421.966] CLOCK_REALTIME rms   12 max   28 freq  -2504 +/-  71 delay   513 +/-   6
Nov 14 05:43:18 aerial-64tr-smc phc2sys[4390]: [522422.982] CLOCK_REALTIME rms   11 max   26 freq  -2468 +/-  67 delay   514 +/-   7
Nov 14 05:43:19 aerial-64tr-smc phc2sys[4390]: [522424.001] CLOCK_REALTIME rms    9 max   24 freq  -2510 +/-  42 delay   513 +/-   7
Nov 14 05:43:20 aerial-64tr-smc phc2sys[4390]: [522425.020] CLOCK_REALTIME rms   10 max   22 freq  -2468 +/-  46 delay   513 +/-   7
Nov 14 05:43:21 aerial-64tr-smc phc2sys[4390]: [522426.039] CLOCK_REALTIME rms    9 max   23 freq  -2492 +/-  38 delay   513 +/-   7

-----Software Packages----------------------------
cmake                              : N/A
docker      /usr/bin               : 27.3.1
gcc         /usr/bin               : 11.4.0
git-lfs     /usr/bin               : 3.0.2
MOFED                              : N/A
meson                              : N/A
ninja                              : N/A
ptp4l       /usr/sbin              : 3.1.1-3
-----Loaded Kernel Modules------------------------
GDRCopy                            : gdrdrv
GPUDirect RDMA                     : N/A
Nvidia                             : nvidia
-----Non-persistent settings----------------------
VM swappiness                      : vm.swappiness = 0
VM zone reclaim mode               : vm.zone_reclaim_mode = 0
```