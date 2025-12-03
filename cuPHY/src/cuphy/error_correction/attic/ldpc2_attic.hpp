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

#if !defined(LDPC_2_ATTIC_HPP_INCLUDED_)
#define LDPC_2_ATTIC_HPP_INCLUDED_

////////////////////////////////////////////////////////////////////////
// ldpc
// Exported functions called by cuPHY LDPC code
namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
cuphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);
    
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
cuphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     cuphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     cuphyLDPCDiagnostic_t* diag,
                                     cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp()
cuphyStatus_t decode_ldpc2_reg_index_fp(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2()
cuphyStatus_t decode_ldpc2_reg_index_fp_x2(decoder&               dec,
                                           LDPC_output_t&         tDst,
                                           const_tensor_pair&     tLLR,
                                           const LDPC_config&     config,
                                           float                  normalization,
                                           cuphyLDPCResults_t*    results,
                                           void*                  workspace,
                                           cuphyLDPCDiagnostic_t* diag,
                                           cudaStream_t           strm);
////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_fp_x2_workspace_size(const decoder&     dec,
                                                                    const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address()
cuphyStatus_t decode_ldpc2_global_address(decoder&               dec,
                                          LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          cuphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          cuphyLDPCDiagnostic_t* diag,
                                          cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_address_workspace_size(const decoder&     dec,
                                                                   const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index()
cuphyStatus_t decode_ldpc2_global_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index()
cuphyStatus_t decode_ldpc2_shared_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index()
cuphyStatus_t decode_ldpc2_shared_cluster_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                cuphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                cuphyLDPCDiagnostic_t* diag,
                                                cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_cluster_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2()
cuphyStatus_t decode_ldpc2_shared_index_fp_x2(decoder&               dec,
                                              LDPC_output_t&         tDst,
                                              const_tensor_pair&     tLLR,
                                              const LDPC_config&     config,
                                              float                  normalization,
                                              cuphyLDPCResults_t*    results,
                                              void*                  workspace,
                                              cuphyLDPCDiagnostic_t* diag,
                                              cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_fp_x2_workspace_size(const decoder&     dec,
                                                                       const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index()
cuphyStatus_t decode_ldpc2_shared_dynamic_index(decoder&               dec,
                                                LDPC_output_t&         tDst,
                                                const_tensor_pair&     tLLR,
                                                const LDPC_config&     config,
                                                float                  normalization,
                                                cuphyLDPCResults_t*    results,
                                                void*                  workspace,
                                                cuphyLDPCDiagnostic_t* diag,
                                                cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_dynamic_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index()
cuphyStatus_t decode_ldpc2_split_index(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_index_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index()
cuphyStatus_t decode_ldpc2_split_dynamic_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               cuphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               cuphyLDPCDiagnostic_t* diag,
                                               cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_dynamic_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index()
cuphyStatus_t decode_ldpc2_split_cluster_index(decoder&               dec,
                                               LDPC_output_t&         tDst,
                                               const_tensor_pair&     tLLR,
                                               const LDPC_config&     config,
                                               float                  normalization,
                                               cuphyLDPCResults_t*    results,
                                               void*                  workspace,
                                               cuphyLDPCDiagnostic_t* diag,
                                               cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_cluster_index_workspace_size(const decoder&     dec,
                                                                        const LDPC_config& cfg);


////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index()
cuphyStatus_t decode_ldpc2_reg_index(decoder&               dec,
                                     LDPC_output_t&         tDst,
                                     const_tensor_pair&     tLLR,
                                     const LDPC_config&     config,
                                     float                  normalization,
                                     cuphyLDPCResults_t*    results,
                                     void*                  workspace,
                                     cuphyLDPCDiagnostic_t* diag,
                                     cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_index_workspace_size(const decoder&     dec,
                                                              const LDPC_config& cfg);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address()
cuphyStatus_t decode_ldpc2_reg_address(decoder&               dec,
                                       LDPC_output_t&         tDst,
                                       const_tensor_pair&     tLLR,
                                       const LDPC_config&     config,
                                       float                  normalization,
                                       cuphyLDPCResults_t*    results,
                                       void*                  workspace,
                                       cuphyLDPCDiagnostic_t* diag,
                                       cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_reg_address_workspace_size(const decoder&     dec,
                                                                const LDPC_config& cfg);

} // namespace ldpc

#endif // !defined(LDPC_2_ATTIC_HPP_INCLUDED_)
