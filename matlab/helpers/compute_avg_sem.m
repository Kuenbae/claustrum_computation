function [avgTraj, semTraj] = compute_avg_sem(traj)
%COMPUTE_AVG_SEM Mean and SEM across trials for trajectory data.
%
% traj is expected to be trials x time x dimensions.

avgTraj = squeeze(mean(traj, 1, 'omitnan'));
semTraj = squeeze(std(traj, 0, 1, 'omitnan')) ./ sqrt(size(traj, 1));
end
