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

function S1 = updateStruct(S1, S2)

S1 = recfun(S1, S2);

% S1 = recfun_cell2array(S2);

end



function [S1] = recfun(S1, S2)

names = fieldnames(S2);
for k = 1:numel(names)    
    name = names{k};
    if isfield(S1, names{k})     
        if isstruct(S1.(names{k}))&& isstruct(S2.(names{k}))
            S1.(names{k}) = recfun(S1.(names{k}), S2.(names{k}));
        elseif ~isstruct(S1.(names{k}))&& ~isstruct(S2.(names{k}))
            s2field = S2.(names{k});
            if iscell(s2field)
                len = [];
                for n = 1:length(s2field)
                    len(n) = length(s2field{n});
                end
                if sum(abs(len-1)) == 0
                    for n = 1:length(s2field)                        
                        if isstruct(s2field{n})                      
                            if length(S1.(names{k})) < n
                                S1.(names{k}){n} = recfun(s2field{n}, s2field{n});
                            else
                                S1.(names{k}){n} = recfun(S1.(names{k}){n}, s2field{n});
                            end
                        else
                            if n == 1
                                S1.(names{k})= [];
                            end
                            S1.(names{k})(n) = s2field{n};
                        end                        
                    end
                else
                    S1.(names{k})  = s2field;
                end
            else
                S1.(names{k})  = s2field;
            end
        else
            fprintf('%s is not found\n', names{k});
        end                        
    else
        fprintf('%s is not found\n', names{k});
        continue;      
    end
end

end


% 
% function [S1] = recfun(S1, S2)
% 
% names = fieldnames(S2);
% for k = 1:numel(names)    
%     name = names{k}
%     if isfield(S1, names{k})     
%         if isstruct(S1.(names{k}))&& isstruct(S2.(names{k}))
%             S1.(names{k}) = recfun(S1.(names{k}), S2.(names{k}));
%         elseif ~isstruct(S1.(names{k}))&& ~isstruct(S2.(names{k}))
%             s2field = S2.(names{k});
%             if iscell(s2field)
%                 for n = 1:length(s2field)
%                     if isstruct(s2field{n})
%                         if length(S1.(names{k})) < n
%                             S1.(names{k}){n} = s2field{n};
%                         else
%                             S1.(names{k}){n} = recfun(S1.(names{k}){n}, s2field{n});
%                         end
%                     else
%                         S1.(names{k}){n} = s2field{n};
%                     end
%                 end
%             else
%                 S1.(names{k})  = s2field;
%             end
%         else
%             fprintf('%s is not found\n', names{k});
%         end                        
%     else
%         fprintf('%s is not found\n', names{k});
%         continue;      
%     end
% end
% 
% end

% 
% 
% function [S2] = recfun_cell2array(S1)
% 
% names = fieldnames(S1);
% for k = 1:numel(names)
%     s1field = S1.(names{k});
%     if iscell(S1.(names{k}))
%         len = [];
%         for n = 1:length(S1.(names{k}))
%             len(n) = length(S1.(names{k}){n});
%         end
%         if sum(abs(len-1)) == 0
%             for n = 1:length(S1.(names{k}))
%                 if isstruct(S1.(names{k}){n})
%                     S2.(names{k}){n} = recfun_cell2array(S1.(names{k}){n});
%                 else
%                     S2.(names{k})(n) = s1field{n};
%                 end
%             end
%         else
%             S2.(names{k})  = s1field;
%         end
%     else
%         S2.(names{k})  = s1field;
%     end
% end
% 
% end
