function fig = plot_gpfa_trajectory(result, conditionName, saveFigures, savePath, idxSeg, edges, viewAngle, camPos, camTgt)
%PLOT_GPFA_TRAJECTORY Plot a GPFA latent trajectory.

if nargin < 3 || isempty(saveFigures), saveFigures = false; end
if nargin < 4 || isempty(savePath), savePath = pwd; end
if nargin < 7 || isempty(viewAngle), viewAngle = [0 90]; end
if nargin < 8, camPos = []; end
if nargin < 9, camTgt = []; end

isOpenOnly = contains(lower(conditionName), 'openonly') || contains(lower(conditionName), 'neutral');
if isOpenOnly
    segmentColors = get_group_colors('openonly', 'seg');
    markerColors = get_group_colors('openonly', 'marker');
else
    segmentColors = get_group_colors('cs', 'seg');
    markerColors = get_group_colors('cs', 'marker');
end

traj = result.seqTrain.xsm(1:3, 20:end)';
traj = smoothdata(traj, 'gaussian', 10);
if ~isOpenOnly
    traj(:, 3) = -traj(:, 3);
end

fig = figure('Position', get(0, 'ScreenSize')); hold on;
plot_traj_segments(traj, idxSeg, edges, segmentColors, markerColors);
grid on;
ax = gca;
if ~isempty(camPos), ax.CameraPosition = camPos; end
if ~isempty(camTgt), ax.CameraTarget = camTgt; end
view(ax, viewAngle);
set(ax, 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], ...
    'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);

if saveFigures
    if ~isfolder(savePath), mkdir(savePath); end
    saveas(fig, fullfile(savePath, sprintf('GPFA_%s.jpg', conditionName)));
end
hold off;
end
