%% RNN PCA trajectory visualization from standardized public activity data
% Visualizes CS + Open, Open-only, and CS-only trajectories for the three
% published RNN clusters. Also visualizes the inhibition condition for the
% claustrum-like cluster when perturbation data are available.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

if ~exist(cfg.files.publicRnnActivity, 'file')
    error('Missing %s. Download the standardized data bundle and place its results/ files in this repository.', cfg.files.publicRnnActivity);
end
D = load(cfg.files.publicRnnActivity);
clusterResult = load_cluster_result(cfg, cfg.files.publicClusterOut);

figDir = fullfile(cfg.figureDir, 'pca_trajectories');
if ~isfolder(figDir), mkdir(figDir); end

trajectory = struct();
for publishedCluster = 1:numel(cfg.cluster.publishedOrder)
    targetClusterLabel = cfg.cluster.publishedOrder(publishedCluster);
    targetNeurons = clusterResult.idx(:) == targetClusterLabel;
    fprintf('Published Cluster %d uses k-means label %d (%d neurons).\n', ...
        publishedCluster, targetClusterLabel, nnz(targetNeurons));

    groupNames = {'cs', 'openonly', 'csonly'};
    displayNames = {'CS + Open', 'Open only', 'CS only'};
    activityGroups = {D.z_score_trial_CS, D.z_score_trial_neu, D.z_score_trial_noopen};
    crossGroups = {D.cross_cs(:), D.cross_Neu(:), repmat(round(mean(D.cross_cs)), size(D.z_score_trial_noopen, 1), 1)};

    [trajStruct, pcaModel] = compute_group_pca_trajectories(activityGroups, crossGroups, targetNeurons, cfg);
    trajectory(publishedCluster).publishedCluster = publishedCluster;
    trajectory(publishedCluster).clusterLabel = targetClusterLabel;
    trajectory(publishedCluster).groups = trajStruct;
    trajectory(publishedCluster).pcaModel = pcaModel;

    [viewAngle, camPos, camTgt, figPos, axesPosition] = get_view_settings(publishedCluster);

    allMean = cat(1, trajStruct.mean);
    axisLimits = padded_axis_limits(allMean, 0.05);

    for g = 1:numel(groupNames)
        fig = figure('Name', sprintf('Cluster %d: %s', publishedCluster, displayNames{g}), 'Position', figPos);
        plot_rnn_trajectory_with_sem(trajStruct(g).mean, trajStruct(g).sem, groupNames{g}, cfg);
        apply_trajectory_axes(gca, axisLimits, viewAngle, camPos, camTgt, axesPosition);
        title(sprintf('Published Cluster %d: %s', publishedCluster, displayNames{g}));
        saveas(fig, fullfile(figDir, sprintf('cluster_%d_%s.jpg', publishedCluster, groupNames{g})));
    end
end

%% Optional inhibition trajectory comparison for the claustrum-like cluster
if exist(cfg.files.publicPerturbations, 'file')
    P = load(cfg.files.publicPerturbations);
    targetClusterLabel = cfg.cluster.claustrumLikeCluster;
    targetNeurons = clusterResult.idx(:) == targetClusterLabel;
    activityGroups = {D.z_score_trial_CS, P.z_score_trial_CS_inhibition};
    crossGroups = {D.cross_cs(:), P.crossing_cs_inhibition(:)};
    [inhTraj, inhPcaModel] = compute_group_pca_trajectories(activityGroups, crossGroups, targetNeurons, cfg);

    fig = figure('Name', 'Claustrum-like cluster inhibition trajectory', 'Position', get(0, 'ScreenSize'));
    hold on;
    plot_rnn_trajectory_with_sem(inhTraj(1).mean, inhTraj(1).sem, 'cs', cfg);
    plot_rnn_trajectory_with_sem(inhTraj(2).mean, inhTraj(2).sem, 'inhibition', cfg);
    title('Claustrum-like cluster: control and inhibition trajectories');
    grid on; view(3);
    saveas(fig, fullfile(figDir, 'cluster_1_inhibition_comparison.jpg'));

    trajectoryInhibition = struct('groups', inhTraj, 'pcaModel', inhPcaModel, 'clusterLabel', targetClusterLabel); %#ok<NASGU>
end

save(cfg.files.trajectoryVisualizationOut, 'trajectory', 'cfg');
fprintf('Saved trajectory visualization output to %s\n', cfg.files.trajectoryVisualizationOut);

%% Local helper functions
function [trajStruct, pcaModel] = compute_group_pca_trajectories(activityGroups, crossGroups, targetNeurons, cfg)
    cropStart = cfg.trajectory.cropStartBin;
    fixedBins = cfg.trajectory.fixedPreOpenBins;
    postBins = cfg.trajectory.postOpenBins;

    cropped = cell(size(activityGroups));
    minTime = inf;
    for i = 1:numel(activityGroups)
        cropped{i} = activityGroups{i}(:, cropStart:end, targetNeurons);
        minTime = min(minTime, size(cropped{i}, 2));
    end
    for i = 1:numel(cropped)
        cropped{i} = cropped{i}(:, 1:minTime, :);
    end

    allActivity = cat(1, cropped{:});
    groupCounts = cellfun(@(x) size(x, 1), cropped);
    [nTotalTrials, nTime, nNeurons] = size(allActivity);
    flatActivity = reshape(allActivity, nTotalTrials * nTime, nNeurons);
    [coeff, score, latent, ~, explained, mu] = pca(flatActivity);
    score3D = reshape(score(:, 1:3), nTotalTrials, nTime, 3);

    pcaModel = struct('coeff', coeff, 'latent', latent, 'explained', explained, 'mu', mu);
    startIdx = 1;
    trajStruct = struct('trialScores', {}, 'mean', {}, 'sem', {});
    for i = 1:numel(groupCounts)
        trialIdx = startIdx:(startIdx + groupCounts(i) - 1);
        scoreGroup = score3D(trialIdx, :, :);
        normGroup = normalize_open_to_cross(scoreGroup, crossGroups{i}, fixedBins, postBins, 'Smooth', cfg.trajectory.smoothing);
        [avgTraj, semTraj] = compute_avg_sem(normGroup);
        trajStruct(i).trialScores = normGroup;
        trajStruct(i).mean = avgTraj;
        trajStruct(i).sem = semTraj;
        startIdx = startIdx + groupCounts(i);
    end
end

function plot_rnn_trajectory_with_sem(avgTraj, semTraj, groupName, cfg)
    hold on;
    segmentColors = get_group_colors(groupName, 'seg');
    markerColors = get_group_colors(groupName, 'marker');
    idxStart = 1;
    idxCSon = 30;
    idxCSoff = 50;
    idxOpen = cfg.trajectory.fixedPreOpenBins;
    idxFinal = size(avgTraj, 1);
    idxSeg = {idxStart:idxCSon, idxCSon:idxCSoff, idxCSoff:idxOpen, idxOpen:idxFinal};
    eventIdx = [idxStart, idxCSon, idxCSoff, idxOpen, idxFinal];

    for k = 1:4
        h = plot3(avgTraj(idxSeg{k}, 1), avgTraj(idxSeg{k}, 2), avgTraj(idxSeg{k}, 3), 'LineWidth', 4);
        h.Color = [segmentColors{k}, 1];
    end
    for k = 1:5
        if strcmp(groupName, 'openonly') && (k == 2 || k == 3)
            continue;
        end
        if strcmp(groupName, 'csonly') && k == 4
            continue;
        end
        scatter3(avgTraj(eventIdx(k), 1), avgTraj(eventIdx(k), 2), avgTraj(eventIdx(k), 3), ...
            300, 'filled', 'MarkerFaceColor', markerColors{k});
    end

    [tubeX, tubeY, tubeZ] = generate_sem_tube(avgTraj, semTraj, 20);
    surf(tubeX, tubeY, tubeZ, 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end

function axisLimits = padded_axis_limits(points, margin)
    mins = min(points, [], 1);
    maxs = max(points, [], 1);
    span = max(maxs - mins, eps);
    axisLimits = [mins(1) - margin * span(1), maxs(1) + margin * span(1), ...
                  mins(2) - margin * span(2), maxs(2) + margin * span(2), ...
                  mins(3) - margin * span(3), maxs(3) + margin * span(3)];
end

function apply_trajectory_axes(ax, axisLimits, viewAngle, camPos, camTgt, axesPosition)
    grid(ax, 'on');
    axis(ax, axisLimits);
    view(ax, viewAngle);
    ax.CameraPosition = camPos;
    ax.CameraTarget = camTgt;
    set(ax, 'Units', 'normalized', 'Position', axesPosition, ...
        'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
    xlabel(ax, 'PC1'); ylabel(ax, 'PC2'); zlabel(ax, 'PC3');
end
