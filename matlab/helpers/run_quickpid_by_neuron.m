function pid = run_quickpid_by_neuron(Y, X1, X2, targetNeuronIdx, fixLen)
%RUN_QUICKPID_BY_NEURON Run quickPID separately for each selected neuron.
%
% Y is a cell array in which Y{n} is fixLen x trials for the n-th selected
% neuron. X1 and X2 are fixLen x trials source matrices.

nNeurons = numel(targetNeuronIdx);
PILVals = cell(1, nNeurons);
pVals = cell(1, nNeurons);

if isempty(gcp('nocreate'))
    parpool;
end

parfor n = 1:nNeurons
    [tempPIL, tempP] = quickPID(Y{n}, X1, X2, 'nBins', fixLen);
    PILVals{n} = tempPIL;
    pVals{n} = tempP;
end

pid.PILVals = PILVals;
pid.pVals = pVals;
pid.PILValsMat = cat(3, PILVals{:});
pid.pMat = cat(3, pVals{:});
pid.PILValsAvg = mean(pid.PILValsMat, 3, 'omitnan');
pid.PILValsSEM = std(pid.PILValsMat, 0, 3, 'omitnan') ./ sqrt(nNeurons);
pid.pAvg = mean(pid.pMat, 3, 'omitnan');
pid.targetNeuronIdx = targetNeuronIdx(:);
end
