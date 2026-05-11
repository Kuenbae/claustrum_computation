function accuracy = decode_by_time(activityTN, labels)
%DECODE_BY_TIME Five-fold cross-validated linear discriminant decoding.
%
% activityTN: time x trials x neurons
% labels:     trials x 1 class labels

nTime = size(activityTN, 1);
accuracy = nan(nTime, 1);
labels = labels(:);

for t = 1:nTime
    X = squeeze(activityTN(t, :, :));
    if isvector(X)
        X = X(:);
    end
    valid = all(isfinite(X), 2) & isfinite(labels);
    if numel(unique(labels(valid))) < 2 || sum(valid) < 5
        continue;
    end
    model = fitcdiscr(X(valid, :), labels(valid), 'DiscrimType', 'linear');
    cvModel = crossval(model, 'KFold', 5);
    accuracy(t) = 1 - kfoldLoss(cvModel);
end
end
