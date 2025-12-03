Build nvipc for partner
=======================
The cuPHY-CP/gt_common_libs folder includes the common libraries used in multiple modules. This guide provide instructions for packaging the code so that it can be built for L2/MAC partner integration.

Prepare source code
-------------------
Related sources are shown below, run ./pack_nvipc.sh to get a tarball of all the source code needed to build nvipc. Source code of external libs fmtlog and libyaml will be downloaded automatically.

```
.
├── CMakeLists.txt
├── external
│   ├── fmtlog
│   └── libyaml
├── nvIPC
├── nvlog
└── README.md
```

Install dependencies
--------------------
The system requires basic build tools (gcc, cmake) and libraries (libpcap, libcunit1, libnuma).

For Ubuntu, run the following command to install the dependencies:

```bash
sudo apt-get install -y build-essential cmake pkg-config libpcap-dev libcunit1-dev libnuma-dev
```

Additional feature requirement:

(1) To support CUDA memory pool, requires CUDA version >= 12 to be installed.

(2) To support fmtlog based nvlog (C++17), requires gcc, g++ version 7.0 or higher

(3) To support nvIPC build with lower versions of gcc and g++ (lower than 7.0), fmtlog will not be included.

Default build
-------------
```bash
cmake -Bbuild
cmake --build build
cmake --install build
```

Configurable options
--------------------
1. **NVIPC_CUDA_ENABLE**: default is ON, depends on CUDA version >= 12
2. **NVIPC_FMTLOG_ENABLE**: default is ON.
3. **CMAKE_BUILD_TYPE**: default is "Release". Config to "Debug" if want to debug by GDB.

Note:
May set **NVIPC_FMTLOG_ENABLE=OFF** to disable fmtlog, especially for low GCC version (lower than 7.0) platform.

# Example 1: use default configurations (Recommended)
```bash
cmake -Bbuild
```

# Example 2: disable CUDA
```bash
cmake -Bbuild -DNVIPC_CUDA_ENABLE=OFF
```

# Example 3: disable NVIPC_FMTLOG_ENABLE
```bash
cmake -Bbuild -DNVIPC_FMTLOG_ENABLE=OFF
```

# Example 4: enable GDB debug info
```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Debug
```

# Test 1: SHM IPC example

Run below 2 processes in different terminals: 1 primary process and 1 secondary process, they communicate with each other:

```bash
sudo ./build/nvIPC/tests/example/test_ipc 3 1 1    # Primary process
sudo ./build/nvIPC/tests/example/test_ipc 3 1 0    # Secondary process
```

# Test 2: SHM IPC unit test
Run below command, it will start a primary process and fork a secondary process.
The 2 processes communicate with each other. Should see all pass in console output.

```bash
sudo ./build/nvIPC/tests/cunit/nvipc_cunit 3 2
```

An nvipc_unit_test.sh script is provided to automatically run the unit test with several different cmake configurations. It can be run in the nvipc_src folder which is extracted from the tarball.

Integration instructions
------------------------
1. Refer to nvIPC/tests/example/test_ipc.c as example of how to use libnvipc.so
2. Provided nvIPC/tests/example/nvipc_secondary.yaml for L2 partner to configure NVIPC. For secondary NVIPC, only need configure "prefix" in shm_config.
3. Recommend L2 to copy the nvipc_secondary.yaml and call the load_nv_ipc_yaml_config() API to load NVIPC configurations. In addition, this API initializes NVIPC logger with "nvipc_log" configurations in yaml file.


