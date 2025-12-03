# aerial_mcore: Python API bindings to the Aerial 5GModel

This page shows how to generate the Aerial Test Vector .h5 files using the Aerial 5GModel Python bindings API.


Quickstart
----------

Step 1. From within the Aerial development container, pip install aerial_mcore.

```
sudo -H pip install ${cuBB_SDK}/5GModel/aerial_mcore/aerial_pkg/dist/aerial_mcore-*.whl
```

Step 2. Generate the TVs.  Run this inside the Aerial development container:
```
cd ${cuBB_SDK}/5GModel/aerial_mcore/examples
source ../scripts/setup.sh
export REGRESSION_MODE=1
python3 ./example_5GModel_regression.py allChannels
ls -alF GPU_test_input/
```

Example output is below:
```
$  cd ${cuBB_SDK}/5GModel/aerial_mcore/examples
$  source ../scripts/setup.sh
[Aerial Python]$  export REGRESSION_MODE=1
[Aerial Python]$  python3 ./example_5GModel_regression.py allChannels

Run genCfgTemplate ...
Test runSim for DL ...
Read config from cfg_template_DL.yaml

Channel: ssb pdcch pdsch csirs prach pucch pusch srs
----------------------------------------------------
Alloc:    1    1     1     1     0     0     0    0

==> PASS

Test runSim for UL ...
Read config from cfg_template_UL.yaml

Channel: ssb pdcch pdsch csirs prach pucch pusch srs
----------------------------------------------------
Alloc:    0    0     0     0     1     1     1    1

PUSCH TB detected
PRACH detected
PUCCH UCI detected
==> PASS
...


[Aerial Python]$  ls -alF GPU_test_input/
total 12406224
drwxr-xr-x 2 aerial   1009    196608 Mar  1 22:40 ./
drwxrwxr-x 3 aerial   1009      4096 Mar  2 17:31 ../
-rw-r--r-- 1 aerial aerial  35195447 Mar  1 22:29 TV_cuphy_F01-DS-01_slot0_MIMO4x4_PRB106_DataSyms11_qam256.h5
-rw-r--r-- 1 aerial aerial      1100 Mar  1 22:29 TV_cuphy_F01-DS-01_slot0_MIMO4x4_PRB106_DataSyms11_qam256.yaml
-rw-r--r-- 1 aerial aerial  23275193 Mar  1 22:29 TV_cuphy_F01-DS-39_slot0_MIMO4x4_PRB72_DataSyms11_qam256.h5
-rw-r--r-- 1 aerial aerial      1099 Mar  1 22:29 TV_cuphy_F01-DS-39_slot0_MIMO4x4_PRB72_DataSyms11_qam256.yaml
...
-rw-r--r-- 1 aerial aerial   1479168 Mar  2 17:31 template_UL_PUCCH_F0_gNB_CUPHY_s0p1.h5
-rw-r--r-- 1 aerial aerial   7253384 Mar  2 17:31 template_UL_PUSCH_gNB_CUPHY_s0p0.h5
-rw-r--r-- 1 aerial aerial    157327 Mar  2 17:31 template_UL_SRS_gNB_CUPHY_s0p0.h5
-rw-r--r-- 1 aerial aerial   3468690 Mar  2 17:31 template_UL_gNB_FAPI_s0.h5
```
