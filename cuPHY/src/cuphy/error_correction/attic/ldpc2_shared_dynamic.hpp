/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if !defined(LDPC2_SHARED_DYNAMIC_HPP_INCLUDED__)
#define LDPC2_SHARED_HPP_DYNAMIC_INCLUDED__

#include "ldpc2.hpp"

namespace ldpc2
{

cuphyStatus_t decode_ldpc2_shared_dynamic_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_SHARED_DYNAMIC_HPP_INCLUDED__)
