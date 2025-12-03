# Aerial Fronthaul Driver

Static library relying on DPDK for managing fronthaul network traffic on C/U-plane.
Aerial Fronthaul Driver implements a subset of O-RAN Fronthaul Control, User and Synchronization Plane Specification 5.0 - November 2020 (O-RAN.WG4.CUS.0-v05.00)

## Building

To build `aerial-fh` only, in the cuBB directory, run the following
```
export cuBB_SDK=$(pwd)
mkdir build && cd build
cmake ..
cd ${cuBB_SDK}/build/cuPHY-CP/aerial-fh-driver/
make -j $(nproc --all)
```

### Unit tests

First, set `UT_NIC` environment variable to the PCIe address of the NIC port to use:
```
export UT_NIC=0000:b5:00.1
```

Running the tests:
```
cd ${cuBB_SDK}/build/cuPHY-CP/aerial-fh-driver/
sudo -E ./test/ut
```

## O-RAN.WG4.CUS.0-v05.00 Support

### Supported
- Ethernet encapsulation
- Application layer fragmentation
- C-plane section types: 0, 1, 3, 5
- Static user data compression (IQ format configured via M-plane)
- Multiple sections per C-plane message
- C-plane section extensions

### Not supported
- IP/UDP encapsulation
- Radio Transport layer fragmentation
- C-plane section types: 6, 7
- Dynamic user data compression


