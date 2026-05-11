%% Stimulation-only simulation summary
% Python generates stimulation-only outputs from the early/front segment of the
% model, without CS or door inputs. This script z-scores activity relative to
% the pre-stimulation baseline and summarizes the selected cluster response.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

C = load(cfg.files.clusteringOut, 'clusterResult');
clusterResult = C.clusterResult;
targetCluster = cfg.cluster.claustrumLikeCluster;
targetNeurons = clusterResult.idx == targetCluster;

control = analyze_stimulation_file(cfg.files.stimulation, cfg, targetNeurons, 'control');
stimSummary = struct();
stimSummary.control = control;
stimSummary.targetCluster = targetCluster;
stimSummary.targetNeurons = find(targetNeurons);

if exist(cfg.files.stimulationNbqx, 'file')
    stimSummary.nbqx = analyze_stimulation_file(cfg.files.stimulationNbqx, cfg, targetNeurons, 'nbqx');
else
    warning('NBQX stimulation file was not found. Saving control-only summary.');
end

save(cfg.files.stimOut, 'stimSummary', 'cfg');
fprintf('Saved stimulation summary to %s\n', cfg.files.stimOut);

plot_stimulation_summary(stimSummary, cfg);

function out = analyze_stimulation_file(filename, cfg, targetNeurons, label)
S = load(filename);
hidden = S.hidden_activity_escape;
modelOutput = S.model_output_escape;
[nTrials, totalSize, networkSize] = size(hidden);
stimStart = min(cfg.stimulation.startBin, totalSize);
baselineLength = min(cfg.stimulation.baselineLength, stimStart - 1);
baselineIdx = (stimStart - baselineLength):(stimStart - 1);

zTrial = nan(nTrials, totalSize, networkSize);
for tr = 1:nTrials
    activity = reshape(hidden(tr, :, :), totalSize, networkSize);
    baseline = mean(activity(baselineIdx, :), 1, 'omitnan');
    sigma = std(activity(baselineIdx, :), 0, 1, 'omitnan');
    sigma(sigma == 0 | isnan(sigma)) = eps;
    zTrial(tr, :, :) = (activity - baseline) ./ sigma;
end
zTrial(~isfinite(zTrial)) = 0;
zMean = squeeze(mean(zTrial, 1, 'omitnan'));

out = struct();
out.label = label;
out.filename = filename;
out.zTrial = zTrial;
out.zMean = zMean;
out.clusterResponse = zMean(:, targetNeurons);
out.meanClusterResponse = mean(out.clusterResponse, 2, 'omitnan');
out.modelOutput = modelOutput;
out.stimStart = stimStart;
out.baselineIdx = baselineIdx;
end

function plot_stimulation_summary(stimSummary, cfg)
control = stimSummary.control;
figure('Name', 'Stimulation-only cluster heatmap');
imagesc(max(control.clusterResponse(cfg.stimulation.startBin-2:min(55, size(control.clusterResponse, 1)), :)', 0));
xlabel('Time bin'); ylabel('Selected cluster neuron');
title('Stimulation-only response, selected cluster');
colorbar;

figure('Name', 'Mean stimulation-only response'); hold on;
plot(control.meanClusterResponse, 'LineWidth', 2, 'DisplayName', 'control');
if isfield(stimSummary, 'nbqx')
    plot(stimSummary.nbqx.meanClusterResponse, 'LineWidth', 2, 'DisplayName', 'NBQX/AP5-like perturbation');
end
xline(control.stimStart, '--', 'stimulation');
xlabel('Time bin'); ylabel('Mean z-score');
title('Mean stimulation-only response');
legend('Location', 'best'); box off;
end
