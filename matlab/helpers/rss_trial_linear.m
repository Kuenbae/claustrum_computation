function rssByBin = rss_trial_linear(targetTraj, openOnlyMean, csOnlyMean)
%RSS_TRIAL_LINEAR Bin-wise RSS for a linear reconstruction of one trajectory.

[nBins, nDim] = size(targetTraj);
rssByBin = zeros(1, nBins);
for d = 1:nDim
    X = [openOnlyMean(:, d), csOnlyMean(:, d)];
    beta = X \ targetTraj(:, d);
    prediction = X * beta;
    rssByBin = rssByBin + (targetTraj(:, d) - prediction)'.^2;
end
end
