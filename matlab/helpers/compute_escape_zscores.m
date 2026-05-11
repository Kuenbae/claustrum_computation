function out = compute_escape_zscores(S, cfg, varargin)
%COMPUTE_ESCAPE_ZSCORES Compute trial-wise and epoch-wise z-scored RNN activity.
%
% out = compute_escape_zscores(S, cfg) separates CS-present and CS-absent
% delayed-escape trials, detects crossing latency from model output, and
% computes z-scored activity across six task epochs:
% baseline, cue, interval, early open, late open, and after cross.
%
% Optional name/value arguments:
%   'DoorDelayBins'     cue-onset to door-opening interval, in bins
%   'FixedCrossAbs'     fixed absolute cross index for no-open simulations
%   'FixedCrossLatency' latency assigned to fixed-cross trials
%   'TotalSize'         number of time bins to analyze

p = inputParser;
addParameter(p, 'DoorDelayBins', cfg.timing.doorDelayBins, @isscalar);
addParameter(p, 'FixedCrossAbs', [], @(x) isempty(x) || isscalar(x));
addParameter(p, 'FixedCrossLatency', cfg.pid.meanCrossBins, @isscalar);
addParameter(p, 'TotalSize', cfg.timing.totalSize, @isscalar);
parse(p, varargin{:});
opts = p.Results;

hidden = S.hidden_activity_escape;
output = S.model_output_escape;
trialParams = S.trial_params_escape;
[nTrials, nTime, nNeurons] = size(hidden);
totalSize = min([opts.TotalSize, nTime, size(output, 2)]);

csTrial = false(nTrials, 1);
for i = 1:nTrials
    csTrial(i) = get_trial_field(trialParams, i, 'CS_present') == 1;
end
csIdx = find(csTrial);
neuIdx = find(~csTrial);

crossAbs = nan(nTrials, 1);
crossLatency = nan(nTrials, 1);
for i = 1:nTrials
    if isempty(opts.FixedCrossAbs)
        searchStart = cfg.timing.cueStart + opts.DoorDelayBins;
        y = squeeze(output(i, :, 1));
        idx = find(y(searchStart:end) >= cfg.timing.crossThreshold, 1, 'first');
        if ~isempty(idx)
            crossLatency(i) = idx;
            crossAbs(i) = searchStart + idx - 1;
        end
    else
        crossAbs(i) = min(opts.FixedCrossAbs, totalSize);
        crossLatency(i) = opts.FixedCrossLatency;
    end
end

[out.z_score_CS, out.z_score_trial_CS, out.CS_trial, out.cross_cs, out.online_cs] = ...
    compute_group(hidden, csIdx, crossAbs, crossLatency, cfg, opts.DoorDelayBins, totalSize);
[out.z_score_neu, out.z_score_trial_neu, out.Neu_trial, out.cross_Neu, out.online_neu] = ...
    compute_group(hidden, neuIdx, crossAbs, crossLatency, cfg, opts.DoorDelayBins, totalSize);

out.cs_present_idx = csIdx;
out.cs_absent_idx = neuIdx;
out.cross_abs = crossAbs;
out.cross_latency = crossLatency;
out.network_size = nNeurons;
out.total_size = totalSize;
out.cs_trial = csTrial;

end

function [zAvg, zTrial, validTrials, validLatency, onlineActivity] = compute_group(hidden, trialIdx, crossAbs, crossLatency, cfg, doorDelayBins, totalSize)
nNeurons = size(hidden, 3);
validMask = ~isnan(crossAbs(trialIdx));
validTrials = trialIdx(validMask);
validLatency = crossLatency(validTrials);
zTrial = nan(length(validTrials), totalSize, nNeurons);
epochZ = nan(6, nNeurons, length(validTrials));
onlineActivity = nan(length(validTrials), nNeurons);

for k = 1:length(validTrials)
    tr = validTrials(k);
    activity = reshape(hidden(tr, 1:totalSize, :), totalSize, nNeurons);
    crossHere = min(max(round(crossAbs(tr)), 1), totalSize);
    doorOpen = cfg.timing.cueStart + doorDelayBins;
    halfCross = round((crossHere - doorOpen) / 2) + doorOpen;
    halfCross = min(max(halfCross, doorOpen), crossHere);

    baselineIdx = safe_range(cfg.timing.cueStart - cfg.timing.baselineLength + 1, cfg.timing.cueStart - 1, totalSize);
    cueIdx = safe_range(cfg.timing.cueStart, cfg.timing.cueStart + cfg.timing.csDuration - 1, totalSize);
    intervalIdx = safe_range(cfg.timing.cueStart + cfg.timing.csDuration, doorOpen - 1, totalSize);
    earlyOpenIdx = safe_range(doorOpen, halfCross, totalSize);
    lateOpenIdx = safe_range(halfCross + 1, crossHere, totalSize);
    afterCrossIdx = safe_range(crossHere + 1, min(crossHere + cfg.timing.afterCross, totalSize), totalSize);

    baseline = mean(activity(baselineIdx, :), 1, 'omitnan');
    sigma = std(activity(baselineIdx, :), 0, 1, 'omitnan');
    sigma(sigma == 0 | isnan(sigma)) = NaN;

    epochActivity = [
        mean(activity(baselineIdx, :), 1, 'omitnan');
        mean(activity(cueIdx, :), 1, 'omitnan');
        mean(activity(intervalIdx, :), 1, 'omitnan');
        mean(activity(earlyOpenIdx, :), 1, 'omitnan');
        mean(activity(lateOpenIdx, :), 1, 'omitnan');
        mean(activity(afterCrossIdx, :), 1, 'omitnan')];

    z = (epochActivity - baseline) ./ sigma;
    z(~isfinite(z)) = 0;
    epochZ(:, :, k) = z;

    zFull = (activity - baseline) ./ sigma;
    zFull(~isfinite(zFull)) = 0;
    zTrial(k, :, :) = zFull;

    onlineIdx = safe_range(cfg.timing.cueStart, crossHere, totalSize);
    onlineActivity(k, :) = mean(zFull(onlineIdx, :), 1, 'omitnan');
end

if isempty(validTrials)
    zAvg = nan(6, nNeurons);
else
    zAvg = mean(epochZ, 3, 'omitnan');
end
end

function idx = safe_range(firstIdx, lastIdx, n)
firstIdx = max(1, round(firstIdx));
lastIdx = min(n, round(lastIdx));
if lastIdx < firstIdx
    idx = firstIdx;
else
    idx = firstIdx:lastIdx;
end
end

function value = get_trial_field(trialParams, idx, fieldName)
% Support both MATLAB cell arrays of structs and struct arrays loaded from scipy.savemat.
if iscell(trialParams)
    entry = trialParams{idx};
else
    entry = trialParams(idx);
end
if isstruct(entry)
    raw = entry.(fieldName);
else
    raw = entry{1}.(fieldName);
end
value = double(raw);
if numel(value) > 1
    value = value(1);
end
end
