function activityTN = time_normalize_activity(zTrial, crossLatency, targetNeurons, fixedPartEnd, postBins)
%TIME_NORMALIZE_ACTIVITY Interpolate variable post-open intervals to fixed length.
%
% zTrial:       trials x time x neurons z-scored activity
% crossLatency: trials x 1 latency from door opening to crossing
% targetNeurons: logical or numeric neuron selector
% fixedPartEnd: last bin retained before the variable interval
% postBins:      number of bins after interpolation
%
% activityTN: (fixedPartEnd + postBins) x trials x selectedNeurons

if islogical(targetNeurons)
    neuronIdx = find(targetNeurons);
else
    neuronIdx = targetNeurons(:)';
end

[nTrials, nTime, ~] = size(zTrial);
activityTN = nan(fixedPartEnd + postBins, nTrials, length(neuronIdx));

for tr = 1:nTrials
    latency = max(1, round(crossLatency(tr)));
    fixedEnd = min(fixedPartEnd, nTime);
    firstPart = squeeze(zTrial(tr, 1:fixedEnd, neuronIdx));
    if isvector(firstPart)
        firstPart = reshape(firstPart, fixedEnd, []);
    end

    varStart = fixedPartEnd + 1;
    varEnd = min(fixedPartEnd + latency, nTime);
    segment = squeeze(zTrial(tr, varStart:varEnd, neuronIdx));
    if isempty(segment)
        segment = squeeze(zTrial(tr, fixedEnd, neuronIdx));
    end
    if isvector(segment)
        segment = reshape(segment, size(segment, 1), []);
    end

    if size(segment, 1) == 1
        interpData = repmat(segment, postBins, 1);
    else
        oldTime = linspace(0, 1, size(segment, 1));
        newTime = linspace(0, 1, postBins);
        interpData = interp1(oldTime, segment, newTime, 'linear', 'extrap');
    end

    activityTN(:, tr, :) = [firstPart; interpData];
end
end
