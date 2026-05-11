%% Nonlinear prediction of CS + Open trajectories from component conditions
% Compares linear regression and a small MLP for reconstructing post-open
% trajectories from Open-only and CS-only component trajectories.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

requiredInputs = {cfg.files.publicRnnActivity, cfg.files.publicPerturbations, cfg.files.publicClusterOut};
for i = 1:numel(requiredInputs)
    if ~exist(requiredInputs{i}, 'file')
        error('Missing required input: %s', requiredInputs{i});
    end
end

D = load(cfg.files.publicRnnActivity);
P = load(cfg.files.publicPerturbations);
clusterResult = load_cluster_result(cfg, cfg.files.publicClusterOut);
targetClusterLabel = cfg.cluster.claustrumLikeCluster;

meanCrossCS = round(mean(D.cross_cs));
csOnlyCross = repmat(meanCrossCS, size(D.z_score_trial_noopen, 1), 1);

conditions = struct([]);
conditions(1).name = 'CS + Open';
conditions(1).activity = D.z_score_trial_CS;
conditions(1).cross = D.cross_cs(:);
conditions(2).name = 'Inhibition';
conditions(2).activity = P.z_score_trial_CS_inhibition;
conditions(2).cross = P.crossing_cs_inhibition(:);
conditions(3).name = '180 s interval';
conditions(3).activity = P.z_score_trial_CS_interval180;
conditions(3).cross = P.crossing_cs_interval180(:);

nonlinearity = struct();
figDir = fullfile(cfg.figureDir, 'nonlinear_prediction');
if ~isfolder(figDir), mkdir(figDir); end

for i = 1:numel(conditions)
    [figs, stats] = analyze_rss_pca( ...
        clusterResult, targetClusterLabel, ...
        conditions(i).activity, D.z_score_trial_neu, D.z_score_trial_noopen, ...
        conditions(i).cross, D.cross_Neu(:), csOnlyCross, cfg, conditions(i).name);
    nonlinearity(i).condition = conditions(i).name;
    nonlinearity(i).stats = stats;
    saveas(figs.rss, fullfile(figDir, sprintf('rss_%02d_%s.jpg', i, matlab.lang.makeValidName(conditions(i).name))));
end

means = arrayfun(@(x) x.stats.meanValue, nonlinearity);
sems = arrayfun(@(x) x.stats.semValue, nonlinearity);
fig = figure('Name', 'Linear error from nonlinearity'); hold on;
bar(1:numel(means), means);
errorbar(1:numel(means), means, sems, 'k.', 'LineWidth', 1.5);
set(gca, 'XTick', 1:numel(means), 'XTickLabel', {conditions.name});
ylabel('Normalized linear - MLP RSS');
box off;
saveas(fig, fullfile(figDir, 'summary_linear_error_from_nonlinearity.jpg'));

save(cfg.files.nonlinearOut, 'nonlinearity', 'cfg');
fprintf('Saved nonlinear trajectory-prediction output to %s\n', cfg.files.nonlinearOut);
