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

#include "blockFP_test_kernels.h"
#include <CLI/CLI.hpp> // Assuming this is in the include path
#include <iostream>
#include <string>
#include <cstdint>


int main(int argc, char **argv) {
    CLI::App app{"BlockFP Compression Test Utility"};
    app.footer("This utility tests Block Floating Point compression and decompression kernels.");

    // Default values match those in the original main function
    uint32_t nfreqs = 10;
    uint8_t compbits = 9;
    float beta = 10000.0f;
    int16_t beam_id_offset = -1; // Default: -1 (beam_id not packed)
    std::string filename; // Default: no input file, generate random data
    bool verbose = false;

    app.add_option("-f,--file", filename, "Path to the input HDF5 file. If not provided, random data is generated.");
    app.add_option("-b,--beamid", beam_id_offset, "Beam ID offset. If >= 0, Beam ID is packed. Default: -1");
    app.add_flag("-v,--verbose", verbose, "Enable verbose output to print detailed information and diagnostics.");
    app.add_option("--nfreqs", nfreqs, "Number of frequency groups (PRB groups). Default: 10");
    app.add_option("--compbits", compbits, "Number of compression bits (e.g., 8, 9, ...). 16 for uncompressed, 32 for FP pass-through. Default: 9");
    app.add_option("--beta", beta, "Scaling factor beta for compression. Default: 10000.0");

    CLI11_PARSE(app, argc, argv);

    if (verbose) {
        std::cout << "Starting BlockFP Test with parameters from CLI:" << std::endl;
        std::cout << "  Number of Frequencies (nfreqs): " << nfreqs << std::endl;
        std::cout << "  Compression Bits (compbits): " << static_cast<int>(compbits) << std::endl;
        std::cout << "  Scaling Factor (beta): " << beta << std::endl;
        std::cout << "  Beam ID Offset (beam_id_offset): " << beam_id_offset << std::endl;
        if (!filename.empty()) {
            std::cout << "  Input HDF5 File (filename): " << filename << std::endl;
        } else {
            std::cout << "  Input HDF5 File (filename): Not provided, will use random data." << std::endl;
        }
        std::cout << "  Verbose Mode: Enabled" << std::endl;
    }

    // Call the CUDA processing function defined in the .cu file
    execute_blockfp_processing(nfreqs, compbits, beta, beam_id_offset, filename, verbose);

    return 0;
} 