function colors = get_group_colors(groupName, typeName)
%GET_GROUP_COLORS Return colors for trajectory segments or event markers.

groupName = lower(groupName);
typeName = lower(typeName);

colorMap.cs.seg = {[0.5 0.5 0.5], [0 0.6 0], [0.2 0.2 0.2], [0.80 0.35 0.17]};
colorMap.cs.marker = {[0 0 0], [0 0.3 0], [0 0.3 0], [0.45 0.25 0.00], [0.65 0.1 0.13]};

colorMap.openonly.seg = {[0.5 0.5 0.5], [0.5 0.5 0.5], [0.5 0.5 0.5], [0.80 0.35 0.17]};
colorMap.openonly.marker = {[0 0 0], [0.2 0.3 0.2], [0.2 0.3 0.2], [0.45 0.25 0.00], [0.65 0.1 0.13]};

colorMap.csonly.seg = {[0.5 0.5 0.5], [0 0.6 0], [0.2 0.2 0.2], [0.2 0.2 0.2]};
colorMap.csonly.marker = {[0 0 0], [0 0.3 0], [0 0.3 0], [0.6 0.6 0.6], [0 0 0]};

colorMap.inhibition.seg = {[0.3 0.3 0.3], [0 0.6 0], [0.5 0 0.5], [0.9 0.8 0.3]};
colorMap.inhibition.marker = {[0 0 0], [0 0.3 0], [0 0.3 0], [0.9 0.8 0.3], [0.9 0.8 0.3]};

if ~isfield(colorMap, groupName)
    error('Unknown group name: %s', groupName);
end
colors = colorMap.(groupName).(typeName);
end
