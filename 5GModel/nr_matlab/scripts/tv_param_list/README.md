# Description
This script is designed to create an Excel file that shows all parameters of given test vectors. The user can specify a directory where test vectors are stored using the command line argument `-i <directory>`.

# Usage
To use the script, run the following command:

```python
python3 tv_param_list.py -i <directory>
```

where `<directory>` is the path to the directory where the test vectors are stored.

# Requirements
* Python 3.x
* Pandas
* Numpy
* xlsxwriter

# Output
The script creates an Excel file named `tv_param_list.xlsx` that contains all parameters of the given test vectors.

# Example
To run the script on the test vectors located in the directory /path/to/test_vectors, use the following command:

```python3
$ ./tv_param_list.py -i /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/
[1/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_0677_gNB_FAPI_s0.h5
[2/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2895_gNB_FAPI_s0.h5
[3/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_ULMIX_2258_gNB_FAPI_s0.h5
[4/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2095_gNB_FAPI_s0.h5
[5/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2643_gNB_FAPI_s0.h5
(snip)
[6001/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2718_gNB_FAPI_s0.h5
[6002/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_3961_gNB_FAPI_s0.h5
[6003/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2294_gNB_FAPI_s0.h5
[6004/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_2004_gNB_FAPI_s0.h5
[6005/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_ULMIX_1301_gNB_FAPI_s0.h5
[6006/6006] /opt/nvidia/cuBB/5GModel/aerial_mcore/examples/GPU_test_input/TVnr_DLMIX_1853_gNB_FAPI_s0.h5
            TC  PBCH PDCCH_DL PDSCH CSI_RS PRACH PUCCH PUSCH   SRS PDCCH_UL  BFW
0         None  None     None  None   None  None  None  None  None     None  NaN
0         0677   NaN      NaN   NaN    NaN     1    24     6   NaN      NaN  NaN
0   DLMIX_2895   NaN        1     6      2   NaN   NaN   NaN   NaN        1  NaN
0   ULMIX_2258   NaN      NaN   NaN    NaN     4    24     6   NaN      NaN  NaN
0   DLMIX_2095   NaN        1     6      1   NaN   NaN   NaN   NaN        1  NaN
..         ...   ...      ...   ...    ...   ...   ...   ...   ...      ...  ...
0         3961   NaN      NaN     1    NaN   NaN   NaN   NaN   NaN      NaN  NaN
0   DLMIX_2294   NaN        1     6      3   NaN   NaN   NaN   NaN        1  NaN
0   DLMIX_2004     2        1     6    NaN   NaN   NaN   NaN   NaN        1  NaN
0   ULMIX_1301   NaN      NaN   NaN    NaN   NaN    24     6   NaN      NaN  NaN
0   DLMIX_1853   NaN        1     6    NaN   NaN   NaN   NaN   NaN      NaN  NaN

[6007 rows x 11 columns]
```