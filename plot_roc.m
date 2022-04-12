function plot_roc(ax, P, y, col, updateMethod)

[allFpr, allTpr, allThresh, allAuc] = perfcurve(y, P, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
[~, aucErr_] = roc_ci(allAuc, nnz(y), nnz(~y), 0.05);
hold(ax, 'on');
plot(ax, allFpr, allTpr, 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s AUC = %1.3f [%1.3f, %1.3f]', strrep(updateMethod, '_', ' '), ...
    allAuc, allAuc - aucErr_, allAuc + aucErr_));
hold(ax, 'off');
