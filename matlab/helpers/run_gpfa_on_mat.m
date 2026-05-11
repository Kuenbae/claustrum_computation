function result = run_gpfa_on_mat(dataMat, runIdx, xDim, labelText)
%RUN_GPFA_ON_MAT Run GPFA on a time x neuron data matrix.
%
% Requires the NeuralTraj code package on the MATLAB path.

if nargin < 4
    labelText = 'gpfa';
end
if exist('neuralTraj', 'file') ~= 2
    error('neuralTraj was not found on the MATLAB path. Install/add NeuralTraj before running GPFA.');
end

T = size(dataMat, 1);
dat = struct('seq', []);
dat.seq(1).trialId = 1;
dat.seq(1).T = T;
dat.seq(1).y = dataMat';

result = neuralTraj(runIdx, dat, ...
    'datFormat', 'seq', ...
    'method', 'gpfa', ...
    'xDims', xDim, ...
    'kernSDList', 20, ...
    'parallelize', true);

fprintf('GPFA complete: %s\n', labelText);
end
