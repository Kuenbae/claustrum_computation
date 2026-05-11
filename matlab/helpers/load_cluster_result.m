function clusterResult = load_cluster_result(cfg, clusterFile)
%LOAD_CLUSTER_RESULT Load the fixed t-SNE clustering result.
%
% The repository stores only the final clustering used in the manuscript
% (perplexity = 24, exaggeration = 48). Legacy exploratory result cells are
% intentionally not used by the analysis scripts.

if nargin < 2 || isempty(clusterFile)
    clusterFile = cfg.files.clusteringOut;
end

if ~exist(clusterFile, 'file')
    error('Cluster result file not found: %s. Run run_01_population_clustering.m or download the standardized data bundle.', clusterFile);
end

S = load(clusterFile);
if ~isfield(S, 'clusterResult')
    error('Expected variable clusterResult in %s.', clusterFile);
end

clusterResult = S.clusterResult;
requiredFields = {'Y', 'idx', 'cluster_num', 'perplexity', 'exaggeration'};
for k = 1:numel(requiredFields)
    if ~isfield(clusterResult, requiredFields{k})
        error('clusterResult is missing required field: %s', requiredFields{k});
    end
end
clusterResult.idx = clusterResult.idx(:);
end
