# PCAP to cuPHY tensor converter

Script for extracting user data (IQ) samples from ORAN CUS fronthaul PCAP capture file. 

`pcap_to_tensor.py` produces a separate output file with raw IQ data buffer for every slot. 

Repeating slots with the same *FrameId*, *SubframeId* and *SlotId* are differentiated by the U-plane packet RX timestamp.

`pcap_to_tensor.py` script can be configured using an input YAML file. Please use `config.yaml` as reference, it is used by default.


## Prerequisites

#### Wireshark or tshark (version 3.6.0 or newer)
Ubuntu 22.04 tshark version 3.6.2-2 available as a package:
```
sudo apt install tshark -y
tshark --version
# Output: TShark (Wireshark) 3.6.2 (Git v3.6.2 packaged as 3.6.2-2)
```

If you need a later version of that you can install one with:
```
tsharkVer=4.1.0
apt install libspeexdsp-dev qttools5-dev qttools5-dev-tools libqt5svg5-dev qtmultimedia5-dev build-essential automake autoconf libgtk2.0-dev libglib2.0-dev flex bison libpcap-dev libgcrypt20-dev cmake libc-ares-dev  liblua5.2-dev -y

wget https://1.eu.dl.wireshark.org/src/all-versions/wireshark-$tsharkVer.tar.xz -P /tmp
cd /tmp
tar Jxf wireshark-$tsharkVer.tar.xz
mkdir /tmp/build && cd $_
cmake /tmp/wireshark-$tsharkVer -DBUILD_wireshark=OFF
make -j
sudo make install
which tshark 
tshark --version
```

#### Python3 (version 3.6.9 or newer)
```
sudo apt install python3
```

#### PyShark (version 0.4.3 or newer)
```
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install -U pyshark
```

#### PyYAML (5.4.1)
```
sudo pip3 install -U pyyaml
```


## Example commands

Show help message:
```
python3 pcap_to_tensor.py -h
```

Parse `/tmp/example.pcap` PCAP file using `~/my_config.yaml` configuration:
```
python3 pcap_to_tensor.py /tmp/example.pcap --config ~/my_config.yaml
```

Use packets extracted from an nvipc and saved to `ulTTI.pcap` to extract tensors from `4layerdl.pcap` using the default configuration:
```
./pcap_to_tensor.py -f ulTTI.pcap 4layerdl.pcap
```
