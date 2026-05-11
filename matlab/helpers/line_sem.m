function hLine = line_sem(ax, x, meanValue, semValue, colorValue)
%LINE_SEM Plot a mean line with a shaded SEM band.

hold(ax, 'on');
fill(ax, [x fliplr(x)], [meanValue - semValue, fliplr(meanValue + semValue)], ...
    colorValue, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
hLine = plot(ax, x, meanValue, 'Color', colorValue, 'LineWidth', 1.5, 'HandleVisibility', 'on');
end
