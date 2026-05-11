function [viewAngle, camPos, camTgt, figPos, axesPosition] = get_view_settings(publishedCluster)
%GET_VIEW_SETTINGS Camera presets used for PCA trajectory figures.

figPos = get(0, 'ScreenSize');
switch publishedCluster
    case 3
        viewAngle = [45, 21];
        camPos = [176, -103, 24];
        camTgt = [10, 1, -1];
        axesPosition = [0.02, 0.02, 0.96, 0.96];
    case 2
        viewAngle = [-10, 67];
        camPos = [-10, -89, 62];
        camTgt = [11, 0.5, 0.1];
        axesPosition = [0.2, 0, 0.7, 0.7];
    otherwise
        viewAngle = [-10, 67];
        camPos = [-10, -89, 62];
        camTgt = [11, 0.5, 0.1];
        axesPosition = [0.1, 0.1, 0.9, 0.9];
end
end
