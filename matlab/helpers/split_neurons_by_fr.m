function [group1, group2] = split_neurons_by_fr(activity)
%SPLIT_NEURONS_BY_FR Split neurons into two firing-rate-matched groups.

frMean = mean(activity, 1, 'omitnan');
[~, sortedIdx] = sort(frMean, 'descend');
group1 = activity(:, sortedIdx(1:2:end));
group2 = activity(:, sortedIdx(2:2:end));
end
