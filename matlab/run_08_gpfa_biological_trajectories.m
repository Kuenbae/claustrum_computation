%% GPFA trajectory visualization for biological claustral recordings
% Uses the standardized biological GPFA input file and NeuralTraj dependency.

clear; clc; close all;
addpath(genpath(fileparts(mfilename('fullpath'))));
cfg = config_analysis();

if ~exist(cfg.files.publicGpfaInput, 'file')
    error('Missing %s. Download the standardized data bundle.', cfg.files.publicGpfaInput);
end
if exist('neuralTraj', 'file') ~= 2
    error('neuralTraj was not found. Add the NeuralTraj package to the MATLAB path before running this script.');
end

load(cfg.files.publicGpfaInput);
figDir = fullfile(cfg.figureDir, 'gpfa');
if ~isfolder(figDir), mkdir(figDir); end

clusters = {target1, target2, target3};
names = {'c1', 'c2', 'c3'};
grpList = {'nonexp', 'nonexp', 'exp', 'exp'};
subList = {'preCS', 'preNeutral', 'preCS', 'preNeutral'};

for ic = 1:numel(clusters)
    t = clusters{ic};
    nm = names{ic};
    nonexpIdx = setdiff(t, exploratory_list);
    expIdx = setdiff(t, nonexp_list);
    nonexpCS = nonexpIdx(nonexpIdx <= 101);
    nonexpNeutral = setdiff(nonexpIdx, nonexpCS);
    expCS = expIdx(expIdx <= 101);
    expNeutral = setdiff(expIdx, expCS);
    lists = {nonexpCS, nonexpNeutral, expCS, expNeutral};

    for li = 1:4
        grp = grpList{li};
        sub = subList{li};
        idxList = lists{li};
        cues = arrayfun(@(nid) data_compiled{1, nid}.session{1, 1}.events{1, 1}.timestamps, idxList);
        sessionStops = arrayfun(@(nid) data_compiled{1, nid}.session{1, 1}.events{1, 1}.session_stop, idxList);
        crossings = arrayfun(@(nid) data_compiled{1, nid}.session{1, 1}.events{1, 1}.Cross_timepoint, idxList);

        m.(nm).(grp).cs_z = Zscore_timenorm(:, nonexpCS);
        m.(nm).(grp).neutral_z = Zscore_timenorm(:, nonexpNeutral);
        m.(nm).(grp).([sub '_cues']) = cues;
        m.(nm).(grp).([sub '_sessionStops']) = sessionStops;
        m.(nm).(grp).([sub '_crossings']) = crossings;
    end
end

% The manuscript figure uses cluster 3 biological units for the trajectory comparison.
nm = 'c3';
xDim = 3;
runSpecs = {
    0, 'nonexp', 'cs_z', 'cs';
    1, 'nonexp', 'neutral_z', 'neutral'
};

gpfaResults = struct();
for i = 1:size(runSpecs, 1)
    runIdx = runSpecs{i, 1};
    grp = runSpecs{i, 2};
    fieldName = runSpecs{i, 3};
    label = runSpecs{i, 4};
    dataMat = m.(nm).(grp).(fieldName);
    gpfaResults.(label) = run_gpfa_on_mat(dataMat, runIdx, xDim, label);
end

% Event indices after discarding the first 19 bins for visualization.
tStart = 0; tCSon = 29; tCSoff = 49; tOpen = 54; tCross = 74;
idxStart = floor(tStart / binsize) + 1 + 19;
idxCSon = floor(tCSon / binsize) + 1;
idxCSoff = floor(tCSoff / binsize) + 1;
idxOpen = floor(tOpen / binsize) + 1;
idxCross = floor(tCross / binsize) + 1;
edges = [idxStart, idxCSon, idxCSoff, idxOpen, idxCross] - 19;
idxSeg = arrayfun(@(k) edges(k):edges(k+1), 1:(numel(edges)-1), 'UniformOutput', false);

plot_gpfa_trajectory(gpfaResults.cs, 'cs', true, figDir, idxSeg, edges, [-151, 41], [-12, -24, 20], [1.5, -0.75, 0.25]);
plot_gpfa_trajectory(gpfaResults.neutral, 'neutral', true, figDir, idxSeg, edges, [-31, 41], [-12, -24, 20], [1.5, 0.75, 0.25]);

% Split-neuron validation for CS units.
[split1, split2] = split_neurons_by_fr(m.(nm).nonexp.cs_z);
gpfaResults.cs_split1 = run_gpfa_on_mat(split1, 7, xDim, 'cs_split1');
gpfaResults.cs_split2 = run_gpfa_on_mat(split2, 8, xDim, 'cs_split2');

save(cfg.files.gpfaOut, 'gpfaResults', 'm', 'cfg');
fprintf('Saved biological GPFA output to %s\n', cfg.files.gpfaOut);
