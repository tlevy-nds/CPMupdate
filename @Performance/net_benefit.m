function nb = net_benefit(obj, hfig, dispName, myCol, plotAllNone)

% if isempty(obj.fpr) || isempty(obj.tpr) || isempty(obj.thresh)
    % [fpr, tpr, thresh] = perfcurve(obj.y, obj.P, max(obj.y), 'XCrit', 'fpr', 'YCrit', 'tpr');
    [fpr, tpr, thresh] = perfcurve(~obj.y, 1-obj.P, max(~obj.y), 'XCrit', 'fpr', 'YCrit', 'tpr');
% else
%     [fpr, tpr, thresh] = deal(obj.fpr, obj.tpr, obj.thresh);
% end

w = thresh ./ (1 - thresh);

% if w == 0, then NB == TP/n and 

eventRate = nnz(~obj.y == 1) / numel(~obj.y);
nb = tpr * eventRate - w .* fpr * (1-eventRate);
nbNone = 0 - w .* 0;
nbAll = eventRate - w .* (1 - eventRate);

if ~exist('hfig', 'var') || isempty(hfig)
    hfig = figure();
end
    
figure(hfig);ax = gca();hold(ax, 'on');
if plotAllNone
    plot(ax, thresh, nbNone, 'Color', [0 0 0], 'LineWidth', 2, 'LineStyle', '-', 'DisplayName', 'None');
    plot(ax, thresh, nbAll, 'Color', [.5 .5 .5], 'LineWidth', 2, 'LineStyle', '-', 'DisplayName', 'All');
end
plot(ax, thresh, nb, 'Color', myCol, 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', dispName);
hold(ax, 'off');
ylim(ax, [-.1, eventRate + .1]);
grid(ax, 'on');
xlabel(ax, 'Threshold Probability');
ylabel(ax, 'Net Benefit');
% legend(ax, {'None' 'All' 'Model'});
