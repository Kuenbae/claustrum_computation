%% Population clustering with manuscript t-SNE parameters
% This script computes epoch-wise z-scored RNN activity, embeds neurons with
% t-SNE using perplexity = 24 and exaggeration = 48, and clusters the embedded
% points with the gap statistic followed by k-means.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

S = load(cfg.files.main);
z = compute_escape_zscores(S, cfg);

% Rows are task epochs and columns are neurons. Baseline rows are excluded
% before t-SNE, matching the population analysis described in the manuscript.
X = [z.z_score_CS(2:end, :); z.z_score_neu(2:end, :)]';

rng(cfg.tsne.rngSeed, 'twister');
opts = statset('MaxIter', cfg.tsne.maxIter);
Y = tsne(X, ...
    'Algorithm', 'exact', ...
    'Distance', cfg.tsne.distance, ...
    'NumDimensions', cfg.tsne.numDimensions, ...
    'Perplexity', cfg.tsne.perplexity, ...
    'Exaggeration', cfg.tsne.exaggeration, ...
    'LearnRate', cfg.tsne.learnRate, ...
    'Standardize', true, ...
    'Options', opts, ...
    'Verbose', 1);

clusterEval = evalclusters(Y, 'kmeans', 'gap', 'KList', 1:cfg.cluster.maxClusters);
clusterNum = max(clusterEval.OptimalK, 2);
idx = kmeans(Y, clusterNum, 'Replicates', cfg.cluster.kmeansReplicates, 'Display', 'off');

clusterResult = struct();
clusterResult.Y = Y;
clusterResult.idx = idx;
clusterResult.cluster_num = clusterNum;
clusterResult.perplexity = cfg.tsne.perplexity;
clusterResult.exaggeration = cfg.tsne.exaggeration;
clusterResult.rngSeed = cfg.tsne.rngSeed;
clusterResult.distance = cfg.tsne.distance;
clusterResult.epochMatrix = X;
clusterResult.gapCriterion = clusterEval;

save(cfg.files.clusteringOut, 'clusterResult', 'z', 'cfg');
fprintf('Saved fixed-parameter clustering to %s\n', cfg.files.clusteringOut);

figure('Name', 'Fixed t-SNE clustering');
scatter3(Y(:, 1), Y(:, 2), Y(:, 3), 60, idx, 'filled');
xlabel('t-SNE 1'); ylabel('t-SNE 2'); zlabel('t-SNE 3');
title(sprintf('t-SNE clustering: perplexity = %d, exaggeration = %d', ...
    cfg.tsne.perplexity, cfg.tsne.exaggeration));
grid on; view(3);
