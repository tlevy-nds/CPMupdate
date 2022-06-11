function meanVarN = plot_roc(ax, P, y, col, updateMethod)

[allFpr, allTpr, allThresh, allAuc] = perfcurve(y, P, 1, 'XCrit', 'fpr', 'YCrit', 'tpr');
alpha_ = 0.05;
[~, aucErr_] = roc_ci(allAuc, nnz(y), nnz(~y), alpha_);
hold(ax, 'on');
plot(ax, allFpr, allTpr, 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s AUC = %1.3f [%1.3f, %1.3f]', strrep(updateMethod, '_', ' '), ...
    allAuc, allAuc - aucErr_, allAuc + aucErr_));
hold(ax, 'off');

% I assume that n should be length(y) and not some effective n like 2 * sqrt(nnz(y) * nnz(~y))?
meanVarN = [allAuc, (aucErr_ / norminv(1 - alpha_/2, 0, 1)) ^ 2, length(y)];