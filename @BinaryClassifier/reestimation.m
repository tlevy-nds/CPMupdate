function reestimation(obj, X, y, varargin)

predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);

obj.train(X(:, predInds), y, varargin{:});
obj.alphaNew = 0;
obj.betaOverall = 1;
