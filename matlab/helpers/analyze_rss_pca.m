function [figs, stats] = analyze_rss_pca(clusterResult, targetClusterLabel, targetActivity, openOnlyActivity, csOnlyActivity, targetCross, openOnlyCross, csOnlyCross, cfg, conditionName)
%ANALYZE_RSS_PCA Compare linear and MLP trajectory reconstructions.
%
% targetActivity  : trials x time x neurons for the condition being predicted
% openOnlyActivity: trials x time x neurons for the door-opening-only component
% csOnlyActivity  : trials x time x neurons for the CS-only component

if nargin < 10 || isempty(conditionName)
    conditionName = 'target';
end

targetNeurons = clusterResult.idx(:) == targetClusterLabel;
if ~any(targetNeurons)
    error('No neurons found for cluster label %d.', targetClusterLabel);
end

cropStart = cfg.trajectory.cropStartBin;
fixedPreOpenBins = cfg.trajectory.fixedPreOpenBins;
postOpenBins = cfg.trajectory.postOpenBins;

A = targetActivity(:, cropStart:end, targetNeurons);
B = openOnlyActivity(:, cropStart:end, targetNeurons);
C = csOnlyActivity(:, cropStart:end, targetNeurons);
minTime = min([size(A, 2), size(B, 2), size(C, 2)]);
A = A(:, 1:minTime, :);
B = B(:, 1:minTime, :);
C = C(:, 1:minTime, :);

allActivity = cat(1, A, B, C);
[nTotalTrials, nTime, nNeurons] = size(allActivity);
flatActivity = reshape(allActivity, nTotalTrials * nTime, nNeurons);
[~, score] = pca(flatActivity);
score3D = reshape(score(:, 1:3), nTotalTrials, nTime, 3);

nTarget = size(A, 1);
nOpenOnly = size(B, 1);
nCsOnly = size(C, 1);
scoreTarget = score3D(1:nTarget, :, :);
scoreOpenOnly = score3D(nTarget + (1:nOpenOnly), :, :);
scoreCsOnly = score3D(nTarget + nOpenOnly + (1:nCsOnly), :, :);

normTarget = normalize_open_to_cross(scoreTarget, targetCross, fixedPreOpenBins, postOpenBins, 'Smooth', cfg.trajectory.smoothing);
normOpenOnly = normalize_open_to_cross(scoreOpenOnly, openOnlyCross, fixedPreOpenBins, postOpenBins, 'Smooth', cfg.trajectory.smoothing);
normCsOnly = normalize_open_to_cross(scoreCsOnly, csOnlyCross, fixedPreOpenBins, postOpenBins, 'Smooth', cfg.trajectory.smoothing);

trimStart = fixedPreOpenBins + 1;
targetTrim = normTarget(:, trimStart:end, :);
openOnlyMean = squeeze(mean(normOpenOnly(:, trimStart:end, :), 1, 'omitnan'));
csOnlyMean = squeeze(mean(normCsOnly(:, trimStart:end, :), 1, 'omitnan'));

rng(0, 'twister');
nBins = size(targetTrim, 2);
nTrials = size(targetTrim, 1);
regularizationCandidates = cfg.rss.regularizationCandidates;
optList = zeros(1, cfg.rss.lambdaSweeps);

for sweepIdx = 1:cfg.rss.lambdaSweeps
    rssMlp = zeros(numel(regularizationCandidates), nTrials, nBins);
    for tr = 1:nTrials
        thisTraj = squeeze(targetTrim(tr, :, :));
        for candidateIdx = 1:numel(regularizationCandidates)
            rssMlp(candidateIdx, tr, :) = rss_trial_mlp(thisTraj, openOnlyMean, csOnlyMean, cfg.rss.hiddenSize, regularizationCandidates(candidateIdx));
        end
    end
    meanRss = squeeze(mean(rssMlp, [2 3], 'omitnan'));
    [~, bestIdx] = min(meanRss);
    optList(sweepIdx) = regularizationCandidates(bestIdx);
end
optRegularization = mode(optList);

rssLinear = zeros(nTrials, nBins);
rssMlpAll = zeros(cfg.rss.mlpRepeats, nTrials, nBins);
for tr = 1:nTrials
    thisTraj = squeeze(targetTrim(tr, :, :));
    rssLinear(tr, :) = rss_trial_linear(thisTraj, openOnlyMean, csOnlyMean);
    for repIdx = 1:cfg.rss.mlpRepeats
        rssMlpAll(repIdx, tr, :) = rss_trial_mlp(thisTraj, openOnlyMean, csOnlyMean, cfg.rss.hiddenSize, optRegularization);
    end
end

rssMlpMean = squeeze(mean(rssMlpAll, 1, 'omitnan'));
rssDiff = rssLinear - rssMlpMean;
rssDiffNorm = rssDiff ./ mean(rssLinear, 2, 'omitnan');
meanRssDiffNorm = mean(rssDiffNorm, 2, 'omitnan');

stats = struct();
stats.condition = conditionName;
stats.clusterLabel = targetClusterLabel;
stats.optRegularization = optRegularization;
stats.rssLinear = rssLinear;
stats.rssMlpMean = rssMlpMean;
stats.rssDiff = rssDiff;
stats.rssDiffNorm = rssDiffNorm;
stats.meanRssDiffNorm = meanRssDiffNorm;
stats.meanValue = mean(meanRssDiffNorm, 'omitnan');
stats.semValue = std(meanRssDiffNorm, 0, 1, 'omitnan') ./ sqrt(numel(meanRssDiffNorm));

figs = struct();
figs.rss = figure('Name', ['RSS comparison: ' conditionName]);
ax = axes(figs.rss); hold(ax, 'on');
t = 1:nBins;
meanLinear = mean(rssLinear, 1, 'omitnan');
semLinear = std(rssLinear, 0, 1, 'omitnan') ./ sqrt(nTrials);
meanMlp = mean(rssMlpMean, 1, 'omitnan');
semMlp = std(rssMlpMean, 0, 1, 'omitnan') ./ sqrt(nTrials);
meanDiff = mean(rssDiff, 1, 'omitnan');
semDiff = std(rssDiff, 0, 1, 'omitnan') ./ sqrt(nTrials);
h1 = line_sem(ax, t, meanLinear, semLinear, [0.0 0.6 0.0]);
h2 = line_sem(ax, t, meanMlp, semMlp, [0.4 1.0 0.4]);
h3 = line_sem(ax, t, meanDiff, semDiff, [0 0 0]);
xlabel(ax, 'Post-open time bin');
ylabel(ax, 'RSS');
title(ax, ['Trajectory reconstruction: ' conditionName]);
legend(ax, [h1 h2 h3], {'Linear', 'MLP', 'Linear - MLP'}, 'Location', 'best');
box(ax, 'off');
end
