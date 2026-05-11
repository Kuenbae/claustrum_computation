function [trainedClassifier, validationAccuracy] = train_linear_discriminant(trainingData, responseData, varargin)
%TRAIN_LINEAR_DISCRIMINANT Train a 5-fold cross-validated linear discriminant.
%
% This small wrapper replaces classifier-app-generated functions with a
% generic implementation that works for any number of neurons/features.

p = inputParser;
addParameter(p, 'KFold', 5, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});

trainingData = double(trainingData);
responseData = responseData(:);
if size(trainingData, 1) ~= numel(responseData)
    error('trainingData rows must match responseData length.');
end
if numel(unique(responseData)) < 2
    error('responseData must contain at least two classes.');
end

nFeatures = size(trainingData, 2);
predictorNames = cellstr(compose('x%d', 1:nFeatures));
inputTable = array2table(trainingData, 'VariableNames', predictorNames);

classificationDiscriminant = fitcdiscr( ...
    inputTable, responseData, ...
    'DiscrimType', 'linear', ...
    'Gamma', 0, ...
    'FillCoeffs', 'on', ...
    'ClassNames', unique(responseData));

predictorExtractionFcn = @(x) array2table(double(x), 'VariableNames', predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;

partitionedModel = crossval(classificationDiscriminant, 'KFold', p.Results.KFold);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end
