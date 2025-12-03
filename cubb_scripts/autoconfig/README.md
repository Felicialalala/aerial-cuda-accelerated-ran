# Autoconfig script
The script is aimed at automating the generation of cuphycontroller configurations from FAPI Test Vectors.

## Dependencies
<p>pyyaml (https://pypi.org/project/PyYAML/) <p>
<p>h5py (https://docs.h5py.org/en/latest/build.html)<p>

## Execution command
There are 2 required parameters to be passed to the script - 
- -i : Path to the input directory where all the test vectors are located, configs corresponding to these will be generated. <input_dir>
- -t : Path to the template yaml file using which the configs will be generated. <config_template.yaml>

### Usage
```
python3 auto_controllerConfig.py -i <input_dir> -t <config_template.yaml>
```
### Example
```
python3 auto_controllerConfig.py -i /path/to/GPU_test_input -t ../../cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF.yaml 
```
The above command produces cuphycontroller_nrSim_SCF_xxxx.yaml and out.txt with every line  = xxxx which is the name of the TC 
and 
l2_adapter_config_nrSim_SCF_mu_X.yaml where X represents the value of mu for X in range(0,5)

Note - The script currently only supports FAPI TVs

### Optional Command line arguments
- -o or --output_dir flag can be used to generate the config files in the specified directory. By default the are generated in the
directory passed as input_dir using the "-i" flag

Example Usage :
```
python3 auto_controllerConfig.py -i /path/to/GPU_test_input -t ../../cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF.yaml -o config_output_dir/
```

- -a or --all_cells flag can be used along with an boolean argument "true/false" to make the modification in fields for all the cells. 
By default it is only done for the first cell.

Example Usage :
```
python3 auto_controllerConfig.py -i /path/to/GPU_test_input -t ../../cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF.yaml -a true
```

## Steps to Execute E2E Tests

In all the commands below $cuBB_SDK refers to the cuBB repo location cloned using steps found [here](https://confluence.nvidia.com/display/5GV/Aerial+2021-3+Release+QA+Guide#Aerial2021-3ReleaseQAGuide-Usingrepotooltosyncandbuild)

**Step 1**
Generate cuPHY TVs using 5GModel using the following **MATLAB** command
```
cd('nr_matlab'); startup; [nTC, errCnt] = runRegression({'TestVector'}, {'allChannels'}, 'compact');
```

**Step 2** 
Generate launch_patterns for the test-vectors using the auto_lp.py script
```
cd $cuBB_SDK
cd cubb_scripts
python3 auto_lp.py -i ../5GModel/nr_matlab/GPU_test_input -t launch_pattern_nrSim.yaml
```

**Step 3**
Copy the launch pattern and TV files to TestVectors repo
```
cd $cuBB_SDK
cp ./5GModel/nr_matlab/GPU_test_input/TVnr_* ./testVectors/.
cp ./5GModel/nr_matlab/GPU_test_input/launch_pattern* ./testVectors/multi-cell/.
```

**Step 4**
Generate cuPHY-Controller Config YAML files using the autoconfig.py script 
```
cd $cuBB_SDK
cd cubb_scripts/autoconfig 
python3 auto_controllerConfig.py -i ../../5GModel/nr_matlab/GPU_test_input/ -t ../../cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF.yaml -o ../../cuPHY-CP/cuphycontroller/config
```
The above command produces -
- cuphy configuration files named **cuphycontroller_nrSim_SCF_xxxx.yaml**
- out.txt with every line xxxx which is the name of the TV file
- **l2_adapter_config_nrSim_SCF_mu_X.yaml** where X represents the value of mu for X in range(0,5)

for all the TV files present at the input location (In this case, <cuBB repo location>/5GModel/nr_matlab/GPU_test_input/)
at the location specificed using the optional "-o" flag. 
In this case the above mentioned files will be generated inside the cuphycontroller config directory : <cuBB repo location>/cuPHY-CP/cuphycontroller/config

**Step 5**
Change the configs for cuphycontroller, testMAC and RU-Emulator using the steps [here](https://confluence.nvidia.com/display/5GV/Aerial+2021-3+Release+QA+Guide#Aerial2021-3ReleaseQAGuide-ConfigFileModifications)

**Step 6**
Run the sample test cases
```
# PBCH
sudo -E ./cuphycontroller_scf nrSim_SCF_1006
sudo ./test_mac nrSim 1006 --channels PBCH
sudo ./ru_emulator nrSim 1006 --channels PBCH
# Expect RU Emulator to report 100 PBCH per second

# PUSCH
sudo -E ./cuphycontroller_scf nrSim_SCF_7103
sudo ./test_mac nrSim 7103 --channels PUSCH
sudo ./ru_emulator nrSim 7103 --channels PUSCH
# Expect testMAC to report 100 PUSCH per second
```
More sample cases can be found [here](https://confluence.nvidia.com/display/5GV/Aerial+2021-3+Release+QA+Guide#Aerial2021-3ReleaseQAGuide-Exampletestcases)
