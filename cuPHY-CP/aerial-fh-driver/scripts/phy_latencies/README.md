# phy_latencies.py

Display average latencies of main aerial-fh and cuphydriver operations (e.g. C-plane TX, U-plane packet preparation, PDSCH GPU processing etc...). 
The script takes a cuPHY-CP PHY log as input and computes average delays with standard deviation.

## Usage

### Display help
```
python3 ./phy_latencies.py -h
```

### PHY latencies without Cell aggregation
Display PHY latencies from `phy.log` for each cell separately. Show 95th percentile of each latency type:
```
python3 phy_latencies.py phy.log --percentile 95
```

### PHY latencies with Cell aggregation
Display PHY latencies from `phy.log` with cell aggregation. Save latency CDFs into PNG files. Prefix each generated plot filename and title with `Run A`:
```
python3 phy_latencies.py phy.log -a -p save -t 'Run A'
```
