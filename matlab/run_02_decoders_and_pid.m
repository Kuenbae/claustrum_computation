%% Time-resolved decoders and PID analysis
% Decoder scores are used as effective CS and door-opening source variables.
% PID is then computed for each neuron in the selected cluster.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

mainData = load(cfg.files.main);
csOnlyData = load(cfg.files.csOnly);
if exist(cfg.files.clusteringOut, 'file')
    C = load(cfg.files.clusteringOut, 'clusterResult');
    clusterResult = C.clusterResult;
else
    error('Run matlab/run_01_population_clustering.m before PID analysis.');
end

zOpen = compute_escape_zscores(mainData, cfg);
zCsOnly = compute_escape_zscores(csOnlyData, cfg, ...
    'FixedCrossAbs', cfg.timing.totalSize, ...
    'FixedCrossLatency', cfg.pid.meanCrossBins);

targetNeurons = clusterResult.idx == cfg.cluster.claustrumLikeCluster;
targetNeuronIdx = find(targetNeurons);
fprintf('Selected cluster %d with %d neurons.\n', cfg.cluster.claustrumLikeCluster, numel(targetNeuronIdx));

fixedEndFull = cfg.pid.analysisStartOffset + cfg.pid.fixedPreOpenBins;
cropStart = cfg.pid.analysisStartOffset + 1;
postBins = cfg.pid.meanCrossBins;

%% CS decoder: CS + Open versus Open-only
activityCS = cat(1, zOpen.z_score_trial_CS, zOpen.z_score_trial_neu);
labelsCS = [ones(size(zOpen.z_score_trial_CS, 1), 1); zeros(size(zOpen.z_score_trial_neu, 1), 1)];
crossCS = [zOpen.cross_cs(:); zOpen.cross_Neu(:)];
activityCS_tn = time_normalize_activity(activityCS, crossCS, targetNeurons, fixedEndFull, postBins);
accuracyCS_full = decode_by_time(activityCS_tn, labelsCS);
accuracyCS = accuracyCS_full(cropStart:end);

%% CS decoder for CS-only versus no-CS/no-door trials
activityCsOnlyCS = cat(1, zCsOnly.z_score_trial_CS, zCsOnly.z_score_trial_neu);
labelsCsOnlyCS = [ones(size(zCsOnly.z_score_trial_CS, 1), 1); zeros(size(zCsOnly.z_score_trial_neu, 1), 1)];
crossCsOnlyCS = ones(size(activityCsOnlyCS, 1), 1) * cfg.pid.meanCrossBins;
activityCsOnlyCS_tn = time_normalize_activity(activityCsOnlyCS, crossCsOnlyCS, targetNeurons, fixedEndFull, postBins);
accuracyCsOnlyCS_full = decode_by_time(activityCsOnlyCS_tn, labelsCsOnlyCS);
accuracyCsOnlyCS = accuracyCsOnlyCS_full(cropStart:end);

%% Door decoder in CS trials: CS + Open versus CS-only
nDoorCS = min(size(zOpen.z_score_trial_CS, 1), size(zCsOnly.z_score_trial_CS, 1));
activityDoorCS = cat(1, zOpen.z_score_trial_CS(1:nDoorCS, :, :), zCsOnly.z_score_trial_CS(1:nDoorCS, :, :));
labelsDoorCS = [ones(nDoorCS, 1); zeros(nDoorCS, 1)];
crossDoorCS = [zOpen.cross_cs(1:nDoorCS); ones(nDoorCS, 1) * cfg.pid.meanCrossBins];
activityDoorCS_tn = time_normalize_activity(activityDoorCS, crossDoorCS, targetNeurons, fixedEndFull, postBins);
accuracyDoorCS_full = decode_by_time(activityDoorCS_tn, labelsDoorCS);
accuracyDoorCS = accuracyDoorCS_full(cropStart:end);

%% Door decoder in no-CS trials: Open-only versus no-CS/no-door
nDoorNoCS = min(size(zOpen.z_score_trial_neu, 1), size(zCsOnly.z_score_trial_neu, 1));
activityDoorNoCS = cat(1, zOpen.z_score_trial_neu(1:nDoorNoCS, :, :), zCsOnly.z_score_trial_neu(1:nDoorNoCS, :, :));
labelsDoorNoCS = [ones(nDoorNoCS, 1); zeros(nDoorNoCS, 1)];
crossDoorNoCS = [zOpen.cross_Neu(1:nDoorNoCS); ones(nDoorNoCS, 1) * cfg.pid.meanCrossBins];
activityDoorNoCS_tn = time_normalize_activity(activityDoorNoCS, crossDoorNoCS, targetNeurons, fixedEndFull, postBins);
accuracyDoorNoCS_full = decode_by_time(activityDoorNoCS_tn, labelsDoorNoCS);
accuracyDoorNoCS = accuracyDoorNoCS_full(cropStart:end);

%% Convert decoder accuracies to PID source strengths
csScore = decoder_score(accuracyCS, cfg.pid.decoderSmoothWindow, cfg.pid.roundDigits);
csOnlyCSScore = decoder_score(accuracyCsOnlyCS, cfg.pid.decoderSmoothWindow, cfg.pid.roundDigits);
doorScoreCS = decoder_score(accuracyDoorCS, cfg.pid.decoderSmoothWindow, cfg.pid.roundDigits);
doorScoreNoCS = decoder_score(accuracyDoorNoCS, cfg.pid.decoderSmoothWindow, cfg.pid.roundDigits);
sources = build_pid_templates(cfg, csScore, csOnlyCSScore, doorScoreCS, doorScoreNoCS);

%% Build time-normalized neural responses for the four PID conditions
nCS = min(120, min(size(zOpen.z_score_trial_CS, 1), size(zCsOnly.z_score_trial_CS, 1)));
nCsOnly = nCS;
nNone = min(nCS, size(zCsOnly.z_score_trial_neu, 1));
nOpenOnly = size(zOpen.z_score_trial_neu, 1);

Y_CS = time_normalize_activity(zOpen.z_score_trial_CS(1:nCS, :, :), zOpen.cross_cs(1:nCS), targetNeuronIdx, fixedEndFull, postBins);
Y_openOnly = time_normalize_activity(zOpen.z_score_trial_neu(1:nOpenOnly, :, :), zOpen.cross_Neu(1:nOpenOnly), targetNeuronIdx, fixedEndFull, postBins);
Y_csOnly = time_normalize_activity(zCsOnly.z_score_trial_CS(1:nCsOnly, :, :), ones(nCsOnly, 1) * postBins, targetNeuronIdx, fixedEndFull, postBins);
Y_none = time_normalize_activity(zCsOnly.z_score_trial_neu(1:nNone, :, :), ones(nNone, 1) * postBins, targetNeuronIdx, fixedEndFull, postBins);

Y_CS = Y_CS(cropStart:end, :, :);
Y_openOnly = Y_openOnly(cropStart:end, :, :);
Y_csOnly = Y_csOnly(cropStart:end, :, :);
Y_none = Y_none(cropStart:end, :, :);

Y = cell(1, numel(targetNeuronIdx));
for n = 1:numel(targetNeuronIdx)
    Y{n} = round([squeeze(Y_CS(:, :, n)), squeeze(Y_openOnly(:, :, n)), ...
        squeeze(Y_csOnly(:, :, n)), squeeze(Y_none(:, :, n))], cfg.pid.roundDigits);
end

X1 = [repmat(sources.X1_CS, 1, nCS), ...
      repmat(sources.X1_openOnly, 1, nOpenOnly), ...
      repmat(sources.X1_csOnly, 1, nCsOnly), ...
      repmat(sources.X1_none, 1, nNone)];
X2 = [repmat(sources.X2_CS, 1, nCS), ...
      repmat(sources.X2_openOnly, 1, nOpenOnly), ...
      repmat(sources.X2_csOnly, 1, nCsOnly), ...
      repmat(sources.X2_none, 1, nNone)];

pid = struct();
if exist('quickPID', 'file') == 2
    pid = run_quickpid_by_neuron(Y, X1, X2, targetNeuronIdx, cfg.pid.fixLen);
else
    warning('quickPID was not found on the MATLAB path. Decoder outputs were saved, but PID was not run.');
end

decoders = struct();
decoders.accuracyCS = accuracyCS;
decoders.accuracyCsOnlyCS = accuracyCsOnlyCS;
decoders.accuracyDoorCS = accuracyDoorCS;
decoders.accuracyDoorNoCS = accuracyDoorNoCS;
decoders.csScore = csScore;
decoders.csOnlyCSScore = csOnlyCSScore;
decoders.doorScoreCS = doorScoreCS;
decoders.doorScoreNoCS = doorScoreNoCS;
decoders.sources = sources;
decoders.targetNeuronIdx = targetNeuronIdx;

save(cfg.files.pidOut, 'decoders', 'pid', 'cfg');
fprintf('Saved decoder/PID outputs to %s\n', cfg.files.pidOut);

figure('Name', 'Decoder scores'); hold on;
plot(csScore, 'DisplayName', 'CS score');
plot(csOnlyCSScore, 'DisplayName', 'CS-only score');
plot(doorScoreCS, 'DisplayName', 'Door score, CS trials');
plot(doorScoreNoCS, 'DisplayName', 'Door score, no-CS trials');
xlabel('Time bin'); ylabel('Decoder-derived source strength');
legend('Location', 'best'); box off;
