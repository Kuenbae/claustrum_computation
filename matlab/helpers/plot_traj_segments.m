function plot_traj_segments(traj, idxSeg, edges, segmentColors, markerColors)
%PLOT_TRAJ_SEGMENTS Plot a 3-D trajectory with epoch colors and event markers.

for k = 1:numel(idxSeg)
    h = plot3(traj(idxSeg{k}, 1), traj(idxSeg{k}, 2), traj(idxSeg{k}, 3), 'LineWidth', 4);
    h.Color = [segmentColors{k}, 1];
end
for k = 1:min(numel(edges), numel(markerColors))
    scatter3(traj(edges(k), 1), traj(edges(k), 2), traj(edges(k), 3), 500, 'filled', 'MarkerFaceColor', markerColors{k});
end
end
