%% Cross-temporal decoding for high- and low-synergy neurons
% Divides neurons in the claustrum-like cluster by synergy and trains a
% time-bin-specific decoder for CS + Open versus component conditions.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

if ~exist(cfg.files.publicRnnActivity, 'file') || ~exist(cfg.files.publicPidSynergy, 'file')
    error('Missing public activity or PID synergy input. Download the standardized data bundle.');
end

D = load(cfg.files.publicRnnActivity);
P = load(cfg.files.publicPidSynergy);
clusterResult = load_cluster_result(cfg, cfg.files.publicClusterOut);
targetClusterLabel = cfg.cluster.claustrumLikeCluster;
targetNeurons = clusterResult.idx(:) == targetClusterLabel;
targetNeuronIdx = find(targetNeurons);

pidField = sprintf('PILValsMat%d', targetClusterLabel);
if ~isfield(P.PIValsResults, pidField)
    error('PIValsResults does not contain field %s.', pidField);
end
pidValues = P.PIValsResults.(pidField);
synergyMatrix = squeeze(pidValues(:, 4, :));
meanSynergy = mean(synergyMatrix, 1, 'omitnan');
[~, order] = sort(meanSynergy, 'descend');
nTop = max(1, round(numel(order) * 0.25));

neuronSets = struct();
neuronSets(1).name = 'high_synergy';
neuronSets(1).indices = targetNeuronIdx(order(1:nTop));
neuronSets(2).name = 'low_synergy';
neuronSets(2).indices = targetNeuronIdx(order(end - nTop + 1:end));

figDir = fullfile(cfg.figureDir, 'cross_temporal_decoding');
if ~isfolder(figDir), mkdir(figDir); end

crossTemporal = struct();
for s = 1:numel(neuronSets)
    neurons = neuronSets(s).indices;
    meanCrossCS = round(mean(D.cross_cs));
    normCS = normalize_open_to_cross(D.z_score_trial_CS(:, cfg.trajectory.cropStartBin:end, neurons), ...
        D.cross_cs(:), cfg.trajectory.fixedPreOpenBins, cfg.trajectory.postOpenBins);
    normOpenOnly = normalize_open_to_cross(D.z_score_trial_neu(:, cfg.trajectory.cropStartBin:end, neurons), ...
        D.cross_Neu(:), cfg.trajectory.fixedPreOpenBins, cfg.trajectory.postOpenBins);
    normCsOnly = normalize_open_to_cross(D.z_score_trial_noopen(:, cfg.trajectory.cropStartBin:end, neurons), ...
        repmat(meanCrossCS, size(D.z_score_trial_noopen, 1), 1), cfg.trajectory.fixedPreOpenBins, cfg.trajectory.postOpenBins);

    activity = cat(1, normCS, normOpenOnly, normCsOnly);
    labels = [ones(size(normCS, 1), 1); zeros(size(normOpenOnly, 1) + size(normCsOnly, 1), 1)];
    [nTrials, nBins, nNeurons] = size(activity);
    decodeAcc = nan(nBins, nBins);
    weightsAbs = nan(nNeurons, nBins);

    for trainBin = 1:nBins
        Xtrain = squeeze(activity(:, trainBin, :));
        [model, ~] = train_linear_discriminant(Xtrain, labels);

        classNames = model.ClassificationDiscriminant.ClassNames;
        class0 = find(classNames == 0, 1);
        class1 = find(classNames == 1, 1);
        if ~isempty(class0) && ~isempty(class1)
            coefStruct = model.ClassificationDiscriminant.Coeffs(class0, class1);
            weightsAbs(:, trainBin) = abs(coefStruct.Linear(:));
        end

        for testBin = 1:nBins
            Xtest = squeeze(activity(:, testBin, :));
            yhat = model.predictFcn(Xtest);
            decodeAcc(trainBin, testBin) = mean(labels(:) == yhat(:)) * 100;
        end
    end

    crossTemporal(s).name = neuronSets(s).name;
    crossTemporal(s).neurons = neurons;
    crossTemporal(s).decodeAccuracy = decodeAcc;
    crossTemporal(s).weightsAbs = weightsAbs;

    fig = figure('Name', ['Cross-temporal decoding: ' neuronSets(s).name]);
    imagesc(decodeAcc);
    axis square; colorbar; caxis([50 100]);
    set(gca, 'YDir', 'normal');
    xlabel('Test time bin'); ylabel('Train time bin');
    title(strrep(neuronSets(s).name, '_', ' '));
    hold on;
    xline(cfg.trajectory.fixedPreOpenBins, 'w--', 'LineWidth', 1.5);
    yline(cfg.trajectory.fixedPreOpenBins, 'w--', 'LineWidth', 1.5);
    saveas(fig, fullfile(figDir, sprintf('%s_heatmap.jpg', neuronSets(s).name)));
end

save(cfg.files.crossTemporalOut, 'crossTemporal', 'cfg');
fprintf('Saved cross-temporal decoding output to %s\n', cfg.files.crossTemporalOut);
