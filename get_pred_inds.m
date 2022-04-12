function predInds = get_pred_inds(allPredictors, fixedPredictors)
[~, temp] = ismember(allPredictors, fixedPredictors);
[~, si] = sort(temp);
predInds = si(end - nnz(temp ~= 0) + 1:end);