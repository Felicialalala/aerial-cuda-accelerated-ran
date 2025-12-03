# rx_pcap_capture.py

Capture fronthaul RX traffic. This script saves all packets received by a `aerial-fh` library instance into a PCAP file.

`rx_pcap_capture.py` runs a DPDK *[pdump](https://doc.dpdk.org/guides/howto/packet_capture_framework.html)* client which writes incoming FH packets into a file.

The `dpdk-pdump` client attaches to the DPDK primary process created by *aerial-fh*.

Since there can be multiple `aerial-fh` instances (and, thus, DPDK primary processes) running on the system, `rx_pcap_capture.py` must use the same `--file-prefix` as the *Aerial* application we want to sniff RX trafic on.
 

## How to use
1. Set `cuBB_SDK` environment variable to the *cuBB* root directory.
2. Set `pdump_client_thread` in a `cuphycontroller` config file to a positive value in order to launch the *pdump_client* on the CPU core specified.
3. Check (and modify if needed) the `dpdk_file_prefix` value in `cuphycontroller` config file.
4. Launch `cuphycontroller`.
5. Launch `rx_pcap_capture.py` with the same `--file-prefix` value as in *3.*

Similar instructions apply to `ru_emulator`.

## Example commands

### Display help
```
python3 rx_pcap_capture.py -h
```

### Capture RX traffic
Capture RX packets on NIC `0000:b5:00.1` into `temp.pcap` file using CPU core `1` by attaching to a running *cuphycontroller* instance with `--file-prefix=cuphycontroller`:
```
sudo python3 rx_pcap_capture.py --nic 0000:b5:00.1 --pcap ./temp.pcap --core 1 --file-prefix aerial-fh
```
