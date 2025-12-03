% SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% MATLAB class wrapper to an underlying instance of a wrapper class
% for library function calls.
classdef cuphy < handle
    properties (SetAccess = private, Hidden = false)
        objectHandle;
    end
    methods
        %% -------------------------------------------------------------
        %% Constructor: Create a new instance of the internal wrapper class
        function this = cuphy(varargin)
            this.objectHandle = cuphy_mex('create', varargin{:});
        end
        %% -------------------------------------------------------------
        %% Destructor: Clean up the internal wrapper instance
        function delete(this)
            cuphy_mex('delete', this.objectHandle);
        end
        %% -------------------------------------------------------------
        %% Perform 1D MMSE channel estimation
        function varargout = channelEstMMSE1D(this, varargin)
          [varargout{1:nargout}] = cuphy_mex('channelEstMMSE1D', this.objectHandle, varargin{:});
        end
    end
end
