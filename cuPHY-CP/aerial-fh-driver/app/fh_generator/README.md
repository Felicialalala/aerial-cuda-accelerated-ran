# FH traffic generator

O-RAN fronthaul traffic generator using aerial-fh library to generate bidirectional arbitrary traffic patterns.

`fh_generator` transmits 'dummy' C-plane and U-plane packets on the eCPRI flows specified and according to the transmission windows defined.

`fh_generator` has 2 modes of operation, one as the DU and another as the RU.

1. As the DU, we read the config file to simulate a repeated slot pattern worth of ORAN traffic. The FH Generator will spawn a DL thread and a UL thread.
The DL thread will schedule the defined downlink traffic pattern from the config file, this includes scheduling ORAN UL C-plane, DL C-plane from CPU, and DL U-plane from the GPU.
The UL thread will receive kernels responsible for receiving UL traffic into GPU memory, the packets are scored based on RX time, and the number of PRBs received tracked per slot.

2. As the RU, we read the same config file to simulate the same ORAN traffic but switching the RX and TX. 
The UL thread will schedule UL traffic, while the DL thread will receive all traffic and report the time.

This will be built on top of the Aerial FH API. Future work will include exposing the DPDK API calls to have more customizable traffic patterns not tied to ORAN.

## Building

To build `fh_generator` only, in the cuBB directory, run the following
```
export cuBB_SDK=$(pwd)
mkdir build.$(uname -m) && cd build.$(uname -m)
cmake .. -DENABLE_20C=ON
cd ${cuBB_SDK}/build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator
make -j $(nproc --all)
```

Note: -DENABLE_20C=ON enables up to 32C due to DOCA limitation that Cell count needs to be a power of 2

To run with more than 32C up to 64C, we need to enable the below flag
```
cmake .. -DENABLE_64C=ON
```

## Config file generation

### Command line configs

```
Convert slot pattern in JSON format to fhgen yaml configuration

python3 ./scripts/fhgen_from_pattern.py -h
```

### Example config file generation commands:

The config file names will be appended with the non-default configurations in the directory ${cuBB_SDK}/cuPHY-CP/aerial-fh-driver/app/fh_generator/patterns/.

```
# Generate 1C test for 59C test case for DU NIC 0000:01:00.0 and RU NIC 0000:ca:00.0, with gRPC synchronization using their respective IP Addresses
python3 ./cuPHY-CP/aerial-fh-driver/app/fh_generator/scripts/fhgen_from_pattern.py -c 1 --du_nic_addrs "0000:01:00.0" --ru_nic_addrs "0000:ca:00.0" -i ./cuPHY-CP/traffic_pattern/POC2_59c.json --sfn_slot_sync_ru "10.112.208.161" --sfn_slot_sync_du "10.112.208.176"

```

## Operation
### Setup

1. Read Config file
2. Store the traffic pattern based on slots:cells:flows:PRB ranges
3. Launch UL/DL thread

## Usage example

Generate FH traffic from `./cuPHY-CP/traffic_pattern/POC2_59c_1C.yaml`

```
# DU Side
sudo -E LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/mellanox/dpdk/lib/x86_64-linux-gnu:/opt/mellanox/doca/lib/x86_64-linux-gnu ./build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator ./cuPHY-CP/traffic_pattern/POC2_59c_1C.yaml

# RU Side
sudo -E LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/mellanox/dpdk/lib/x86_64-linux-gnu:/opt/mellanox/doca/lib/x86_64-linux-gnu ./build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator ./cuPHY-CP/traffic_pattern/POC2_59c_1C.yaml -r
```