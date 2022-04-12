function [ax, pl] = plot_pr(obj, ax)

if ~exist('ax', 'var') || ~isvalid(ax)
    h = figure(); ax = gca();
end

hold(ax, 'on');
pl = plot3(ax, obj.recall, obj.precision, obj.thresh, '-', 'LineWidth', 2);
xlabel(ax,'Recall');
ylabel(ax,'Precision');
zlabel(ax,'Threshold');
grid(ax, 'on');
hold(ax, 'off');

obj.axpr = ax;