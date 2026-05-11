function normActivity = normalize_open_to_cross(activity, crossLatency, fixedPreOpenBins, postOpenBins, varargin)
%NORMALIZE_OPEN_TO_CROSS Time-normalize trial activity around door opening.
%
% Inputs
%   activity         trials x time x neurons matrix
%   crossLatency    trials x 1 latency from door opening to crossing
%   fixedPreOpenBins number of bins retained before the variable interval
%   postOpenBins     number of bins used after interpolation
%
% Output
%   normActivity    trials x (fixedPreOpenBins + postOpenBins) x neurons

p = inputParser;
addParameter(p, 'Method', 'pchip', @(x) ischar(x) || isstring(x));
addParameter(p, 'Smooth', false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
method = char(p.Results.Method);
doSmooth = logical(p.Results.Smooth);

if ndims(activity) == 2
    activity = reshape(activity, size(activity, 1), size(activity, 2), 1);
end

[nTrials, nTime, nNeurons] = size(activity);
crossLatency = round(crossLatency(:));
if numel(crossLatency) ~= nTrials
    error('crossLatency must have one value per trial.');
end

normLen = fixedPreOpenBins + postOpenBins;
normActivity = nan(nTrials, normLen, nNeurons);

for tr = 1:nTrials
    latency = max(1, crossLatency(tr));
    fixedEnd = min(fixedPreOpenBins, nTime);
    fixedPart = squeeze(activity(tr, 1:fixedEnd, :));
    if isvector(fixedPart)
        fixedPart = reshape(fixedPart, fixedEnd, []);
    end

    varStart = fixedPreOpenBins + 1;
    varEnd = min(fixedPreOpenBins + latency, nTime);
    if varStart > nTime
        segment = fixedPart(end, :);
    else
        segment = squeeze(activity(tr, varStart:varEnd, :));
    end
    if isvector(segment)
        segment = reshape(segment, size(segment, 1), []);
    end
    if isempty(segment)
        segment = fixedPart(end, :);
    end

    if doSmooth && size(segment, 1) > 2
        segment = smoothdata(segment, 1);
    end

    if size(segment, 1) == 1
        interpData = repmat(segment, postOpenBins, 1);
    else
        oldTime = 1:size(segment, 1);
        newTime = linspace(1, size(segment, 1), postOpenBins);
        interpData = interp1(oldTime, segment, newTime, method, 'extrap');
    end

    if fixedEnd < fixedPreOpenBins
        padRows = repmat(fixedPart(end, :), fixedPreOpenBins - fixedEnd, 1);
        fixedPart = [fixedPart; padRows]; %#ok<AGROW>
    end

    normActivity(tr, :, :) = [fixedPart; interpData];
end
end
