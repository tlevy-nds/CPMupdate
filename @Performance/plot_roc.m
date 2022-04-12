function [ax, pl] = plot_roc(obj, ax)

if ~exist('ax', 'var') || ~isvalid(ax)
    h = figure(); ax = gca();
end

hold(ax, 'on');
if ~isempty(obj.P)
    thr = obj.Pthresh;
else
    thr = obj.thresh;
end

if length(thr) ~= length(obj.fpr) && (thr(1) == thr(2) || isnan(thr(1)) && isnan(thr(2)))
    thr = thr(2:end);
end

if length(thr) == length(obj.fpr)    
    pl = plot3(ax, obj.fpr, obj.tpr, thr, '-', 'LineWidth', 2, 'ButtonDownFcn', @(src, evt) obj.window_button_down_fcn(src, evt));
else
    pl = plot(ax, obj.fpr, obj.tpr, '-', 'LineWidth', 2, 'ButtonDownFcn', @(src, evt) obj.window_button_down_fcn(src, evt));
end
xlabel(ax, 'FPR');
ylabel(ax, 'TPR');
if length(thr) == length(obj.fpr) 
    zlabel(ax, 'Threshold');
end
grid(ax, 'on');
hold(ax, 'off');

obj.axroc = ax;