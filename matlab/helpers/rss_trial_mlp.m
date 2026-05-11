function rssByBin = rss_trial_mlp(targetTraj, openOnlyMean, csOnlyMean, hiddenSize, regularization)
%RSS_TRIAL_MLP Bin-wise RSS for a small MLP reconstruction of one trajectory.
%
% Requires MATLAB Neural Network Toolbox / Deep Learning Toolbox support for
% fitnet and mapminmax.

[nBins, ~] = size(targetTraj);
X = [openOnlyMean, csOnlyMean]';
Y = targetTraj';
X = mapminmax(X, -1, 1);

net = fitnet(hiddenSize, 'trainlm');
net.performParam.regularization = regularization;
net.trainParam.showWindow = false;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net = train(net, X, Y);
Yhat = net(X)';
rssByBin = sum((targetTraj - Yhat).^2, 2)';
end
