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

function  [metrics] = compare(x_ref, x_meas, metric_name, plot_fig, pruneZeros, ref_src_name, meas_src_name)

    fig_name = metric_name;
    if(nargin < 4)       
       plot_fig = 0;
    end
    if(nargin < 5)
       pruneZeros = 1;       
    end
    if(nargin < 6)
       ref_src_name = 'MATLAB';       
    end
    if(nargin < 7)
        meas_src_name = 'GPU';
    end

    % flatten
    x_ref = x_ref(:);
    x_meas = x_meas(:);
	
    metrics.x_ref = x_ref(:);
    metrics.x_meas = x_meas(:);

    % find indices of ref and meas vectors which are below machine precision
    if(~isreal(x_ref))
        idxs_x_ref = union(find(abs(real(x_ref)) < eps), find(abs(imag(x_ref)) < eps));
        idxs_x_meas = union(find(abs(real(x_meas)) < eps), find(abs(imag(x_meas)) < eps));
    else
        idxs_x_ref = find(abs(x_ref) < eps);
        idxs_x_meas = find(abs(x_meas) < eps);
    end
    if ~isempty(idxs_x_ref) || ~isempty(idxs_x_meas)
        if ~pruneZeros
            warning('Found entries below machine epsilon in reference and/or measured vectors');        
        end
        if pruneZeros
            fprintf('Pruning out entries below machine epsilon common to both reference and measured vectors\n');
            % find the common indices which are below machine precision
            idxs_null = intersect(idxs_x_ref, idxs_x_meas);
            if ~isempty(idxs_null)
                x_ref(idxs_null) = [];
                x_meas(idxs_null) = [];
                fig_name = [fig_name ' (zeros pruned out)'];
            end
        end        
    end
      
    metrics.err_pwr = abs(metrics.x_ref - metrics.x_meas).^2;
    metrics.sig_pwr = abs(metrics.x_ref).^2;
    
    if ~pruneZeros
        % clamp zeros to machine epsilon
        if min(metrics.err_pwr) < eps
            idx = find(metrics.err_pwr < eps);
            metrics.err_pwr(idx) = eps;
        end

        % clamp zeros to machine epsilon
        if min(metrics.sig_pwr) < eps
            idx = find(metrics.sig_pwr < eps);
            metrics.sig_pwr(idx) = eps;
        end
    end

    % Compute relative error power
    metrics.rel_err_pwr_db = 10*log10(metrics.err_pwr./metrics.sig_pwr);
    % Peak relative error power
    metrics.peak_rel_err_pwr_db = max(metrics.rel_err_pwr_db(:));
    
    % Average relative error power
    metrics.avg_rel_err_pwr_db = 10*log10(mean(metrics.err_pwr./metrics.sig_pwr));
    
    % SNR
    metrics.snr_db = 10*log10(metrics.sig_pwr./metrics.err_pwr);
    %metrics.max_rel_err = max(metrics.rel_err);

    fprintf('==============================================================\n');
    fprintf('Metric: %s\n', fig_name);   
    fprintf('Peak relative error power    %6.3f db\n', metrics.peak_rel_err_pwr_db);
    fprintf('Average relative error power %6.3f db\n', metrics.avg_rel_err_pwr_db);
    %fprintf('Min SNR                      %6.3f db\n', min(metrics.snr_db));
    if plot_fig
        figure;        
        if(~isreal(x_ref))
            x_ref_re = real(x_ref);
            x_ref_im = imag(x_ref);

            x_meas_re = real(x_meas);
            x_meas_im = imag(x_meas);

            tiledlayout(3,1)
            ax1 = nexttile;
            %plot(snr_db, '.k'); title('SNR in dB');        
            plot(metrics.rel_err_pwr_db, '.k'); title(sprintf('%s\nRelative error power - Avg %4.2fdB Peak %4.2fdB', fig_name, metrics.avg_rel_err_pwr_db, metrics.peak_rel_err_pwr_db));

            ax2 = nexttile;
            plot(x_ref_re, 'ob'); hold on; plot(x_meas_re, 'xr');
            legend(ref_src_name, meas_src_name); title(sprintf('Real component %s vs %s', ref_src_name, meas_src_name));

            ax3 = nexttile;
            plot(x_ref_im, 'ob'); hold on; plot(x_meas_im, 'xr');                        
            legend(ref_src_name, meas_src_name); title(sprintf('Imag component %s vs %s', ref_src_name, meas_src_name));
            linkaxes([ax1 ax2 ax3],'x')
        else
            tiledlayout(2,1);
            ax1 = nexttile;
            plot(metrics.rel_err_pwr_db, '.k'); title(sprintf('%s\nRelative error power - Avg %4.2fdB Peak %4.2fdB', fig_name, metrics.avg_rel_err_pwr_db, metrics.peak_rel_err_pwr_db)); 

            ax2 = nexttile;
            plot(x_ref, 'ob'); hold on; plot(real(x_meas), 'xr');                        
            legend(ref_src_name, meas_src_name); title(sprintf('%s vs %s', ref_src_name, meas_src_name));     
            linkaxes([ax1 ax2],'x')
        end
    end
end
