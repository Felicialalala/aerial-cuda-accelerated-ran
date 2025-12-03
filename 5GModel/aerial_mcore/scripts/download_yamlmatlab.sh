#!/bin/bash
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

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
MODEL_ROOT=$(dirname $PROJECT_ROOT)
echo $SCRIPT starting...

rm -rf $MODEL_ROOT/nr_matlab/yamlmatlab
cd $MODEL_ROOT/nr_matlab && git clone https://github.com/jerelbn/yamlmatlab.git
cd yamlmatlab && git checkout e011be81a77d2bbcb5a88c244789f2211aadeb59
cd external && rm snakeyaml-1.9.jar && wget -q https://repo1.maven.org/maven2/org/yaml/snakeyaml/2.5/snakeyaml-2.5.jar && md5sum -c $SCRIPT_DIR/snakeyaml-2.5.jar.md5
find $MODEL_ROOT/nr_matlab/yamlmatlab -type f -name '*.m' -exec sed -i 's/snakeyaml-1.9.jar/snakeyaml-2.5.jar/g' {} \;
