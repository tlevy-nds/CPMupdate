function [auc, fpr, tpr, thresh] = auc_metric(p, y)
% reshape p to 2 dimensions
szp = size(p);
p = reshape(p, szp(1), []);
% loop over second dimension
auc = zeros(1, size(p, 2));
for ii = 1:size(p, 2)
    [fpr, tpr, thresh, auc(ii)] = perfcurve(y, p(:, ii), max(y), 'XCrit', 'fpr', 'YCrit', 'tpr');
end
% reshape back to the original shape
auc = reshape(auc, [1 szp(2:end)]);
auc = max(auc, 1-auc);