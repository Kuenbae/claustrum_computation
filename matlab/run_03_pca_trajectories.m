%% PCA-based trajectory visualization
% This script builds time-normalized trial trajectories for CS + Open,
% Open-only, and CS-only conditions, performs PCA on the combined population
% activity matrix, and plots condition-averaged trajectories in the first
% three PCs.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

mainData = load(cfg.files.main);
csOnlyData = load(cfg.files.csOnly);
C = load(cfg.files.clusteringOut, 'clusterResult');
clusterResult = C.clusterResult;
targetNeurons = clusterResult.idx == cfg.cluster.claustrumLikeCluster;

zOpen = compute_escape_zscores(mainData, cfg);
zCsOnly = compute_escape_zscores(csOnlyData, cfg, ...
    'FixedCrossAbs', cfg.timing.totalSize, ...
    'FixedCrossLatency', cfg.pid.meanCrossBins);

fixedEndFull = cfg.pid.analysisStartOffset + cfg.pid.fixedPreOpenBins;
cropStart = cfg.pid.analysisStartOffset + 1;
postBins = cfg.pid.meanCrossBins;

Y_CS_open = time_normalize_activity(zOpen.z_score_trial_CS, zOpen.cross_cs, targetNeurons, fixedEndFull, postBins);
Y_open_only = time_normalize_activity(zOpen.z_score_trial_neu, zOpen.cross_Neu, targetNeurons, fixedEndFull, postBins);
Y_CS_only = time_normalize_activity(zCsOnly.z_score_trial_CS, ones(size(zCsOnly.z_score_trial_CS, 1), 1) * postBins, targetNeurons, fixedEndFull, postBins);

Y_CS_open = Y_CS_open(cropStart:end, :, :);
Y_open_only = Y_open_only(cropStart:end, :, :);
Y_CS_only = Y_CS_only(cropStart:end, :, :);

conditions = {'CS + Open', 'Open only', 'CS only'};
trialCounts = [size(Y_CS_open, 2), size(Y_open_only, 2), size(Y_CS_only, 2)];
Yall = cat(2, Y_CS_open, Y_open_only, Y_CS_only);           % time x trials x neurons
YtrialMajor = permute(Yall, [2, 1, 3]);                    % trials x time x neurons
[nTrials, nTime, nNeurons] = size(YtrialMajor);
X = reshape(YtrialMajor, nTrials * nTime, nNeurons);

[coeff, score, latent, ~, explained, mu] = pca(X);
score3 = reshape(score(:, 1:3), nTrials, nTime, 3);

trajectory = struct();
startIdx = 1;
figure('Name', 'PCA trajectories'); hold on;
for c = 1:numel(conditions)
    theseTrials = startIdx:(startIdx + trialCounts(c) - 1);
    meanTraj = squeeze(mean(score3(theseTrials, :, :), 1, 'omitnan'));
    semTraj = squeeze(std(score3(theseTrials, :, :), 0, 1, 'omitnan')) ./ sqrt(numel(theseTrials));
    trajectory(c).name = conditions{c};
    trajectory(c).mean = meanTraj;
    trajectory(c).sem = semTraj;
    trajectory(c).trialScores = score3(theseTrials, :, :);
    plot3(meanTraj(:, 1), meanTraj(:, 2), meanTraj(:, 3), 'LineWidth', 2, 'DisplayName', conditions{c});
    scatter3(meanTraj(1, 1), meanTraj(1, 2), meanTraj(1, 3), 40, 'filled');
    scatter3(meanTraj(end, 1), meanTraj(end, 2), meanTraj(end, 3), 40, 'filled');
    startIdx = startIdx + trialCounts(c);
end
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
title('Time-normalized RNN population trajectories');
legend('Location', 'best'); grid on; view(3);

pcaModel = struct('coeff', coeff, 'latent', latent, 'explained', explained, 'mu', mu);
save(cfg.files.trajectoryOut, 'trajectory', 'pcaModel', 'cfg');
fprintf('Saved PCA trajectory outputs to %s\n', cfg.files.trajectoryOut);
