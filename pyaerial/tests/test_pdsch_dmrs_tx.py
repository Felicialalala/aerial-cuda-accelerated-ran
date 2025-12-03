# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dmrs_tx.py."""
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.pdsch import DmrsTx
from aerial.phy5g.config import PdschDmrsConfig
from aerial.util.fapi import dmrs_fapi_to_bit_array


test_case_numbers = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013,
                     3014, 3015, 3018, 3019, 3020, 3021, 3022, 3023, 3026, 3027, 3028, 3029, 3030,
                     3031, 3033, 3034, 3035, 3036, 3037, 3038, 3040, 3041, 3042, 3043, 3044, 3046,
                     3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3058, 3059, 3060,
                     3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073,
                     3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086,
                     3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099,
                     3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112,
                     3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125,
                     3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138,
                     3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151,
                     3152, 3153, 3154, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210,
                     3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223,
                     3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236,
                     3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3246, 3247, 3248, 3249, 3250,
                     3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3264,
                     3265, 3266, 3267, 3268, 3269, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278,
                     3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3292,
                     3293, 3294, 3295, 3297, 3298, 3299, 3301, 3302, 3303, 3304, 3305, 3306, 3307,
                     3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320,
                     3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333,
                     3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3343, 3344, 3350, 3351, 3352,
                     3353, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412,
                     3413, 3422, 3501, 3502, 3504, 3505, 3506, 3509, 3510, 3513, 3521, 3801, 3802,
                     3803, 3804, 3805, 3901, 3902, 3903, 3904, 3906, 3907, 3908, 3909, 3910, 3911,
                     3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924,
                     3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937,
                     3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950,
                     3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963,
                     3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976,
                     3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989,
                     3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999]


# pylint: disable=too-many-locals
all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, cupy",
    all_cases,
    ids=[f"{test_case_number} - cuPy: {cupy}" for test_case_number, cupy in all_cases]
)
def test_dmrs_tx_run(cuda_stream, test_case_number, cupy):
    """Test running DMRS Tx against Aerial test vectors.

    There are no separate test vectors for DMRS alone, so the test method here
    is to just check that something gets written, and that what gets written is
    the same as in the reference. This is not a 100% proof of correctness, as it
    allows to pass with only part of the DMRS getting written, but it's the best
    we can do with the existing test vectors, without having to write code that
    determines the actual DMRS locations.
    """
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PDSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    num_prb_dl_bwp = input_file["cellStat_pars"]["nPrbDlBwp"][0]
    slot = input_file["cellDyn_pars"]["slotNum"][0]
    num_ues = len(input_file["ue_pars"])

    if np.array(input_file["tb0_PM_W"]).size > 0:
        precoding_matrices = [
            np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["re"]) +
            1j * np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["im"])
            for ue_idx in range(num_ues)
        ]
    else:
        precoding_matrices = [None for _ in range(num_ues)]

    dmrs_params = []
    for ue_idx in range(num_ues):

        ue_grp_idx = input_file["ue_pars"]["ueGrpIdx"][ue_idx]

        dmrs_sym_loc_bmsk = input_file["ueGrp_pars"]["dmrsSymLocBmsk"][ue_grp_idx]
        dmrs_syms = dmrs_fapi_to_bit_array(dmrs_sym_loc_bmsk)
        start_sym = input_file["ueGrp_pars"]["pdschStartSym"][ue_grp_idx]
        num_pdsch_syms = input_file["ueGrp_pars"]["nPdschSym"][ue_grp_idx]
        resource_alloc = input_file["ueGrp_pars"]["resourceAlloc"][ue_grp_idx]
        start_prb = input_file["ueGrp_pars"]["startPrb"][ue_grp_idx]
        num_prbs = input_file["ueGrp_pars"]["nPrb"][ue_grp_idx]
        prb_bitmap = input_file["ueGrp_pars"]["rbBitmap"][ue_grp_idx]
        num_dmrs_cdm_grps_no_data = input_file["dmrs_pars"]["nDmrsCdmGrpsNoData"][ue_grp_idx]

        scid = input_file["ue_pars"]["scid"][ue_idx]
        dmrs_scrm_id = input_file["ue_pars"]["dmrsScramId"][ue_idx]
        layers = input_file["ue_pars"]["nUeLayers"][ue_idx]
        dmrs_ports = input_file["ue_pars"]["dmrsPortBmsk"][ue_idx]
        beta_qam = input_file["ue_pars"]["beta_qam"][ue_idx]
        beta_dmrs = input_file["ue_pars"]["beta_dmrs"][ue_idx]
        ref_point = input_file["ue_pars"]["refPoint"][ue_idx]
        bwp_start = input_file["ue_pars"]["BWPStart"][ue_idx]

        dmrs_params.append(PdschDmrsConfig(
            cell_index_in_cell_group=0,
            num_bwp_prbs=num_prb_dl_bwp,
            num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
            dmrs_scrm_id=dmrs_scrm_id,
            resource_alloc=resource_alloc,
            prb_bitmap=prb_bitmap,
            start_prb=start_prb,
            num_prbs=num_prbs,
            dmrs_syms=dmrs_syms,
            start_sym=start_sym,
            num_pdsch_syms=num_pdsch_syms,
            scid=scid,
            layers=layers,
            dmrs_ports=dmrs_ports,
            bwp_start=bwp_start,
            ref_point=ref_point,
            beta_qam=beta_qam,
            beta_dmrs=beta_dmrs,
            precoding_matrix=precoding_matrices[ue_idx]
        ))

    dmrs_tx = DmrsTx(cuda_stream=cuda_stream, max_num_tbs=num_ues)

    ref_buffer = np.array(input_file["Xtf"])
    ref_buffer = ref_buffer["re"] + 1j * ref_buffer["im"]
    if ref_buffer.ndim == 2:
        ref_buffer = ref_buffer[None]
    ref_buffer = ref_buffer.transpose(2, 1, 0).astype(np.complex64)

    if cupy:
        zeros_input = cp.zeros_like(ref_buffer)
    else:
        zeros_input = np.zeros_like(ref_buffer)

    tx_buffer = dmrs_tx.run(slot=slot,
                            tx_buffers=[zeros_input],
                            dmrs_params=dmrs_params)[0]

    if cupy:
        tx_buffer = tx_buffer.get()
        zeros_input = zeros_input.get()

    # Compare the non-zero parts with the reference buffer.
    # Check also that something got actually written.
    dmrs_loc = np.where(tx_buffer != zeros_input)
    assert dmrs_loc[0].size > 0
    assert np.allclose(ref_buffer[dmrs_loc], tx_buffer[dmrs_loc], atol=1e-3)
