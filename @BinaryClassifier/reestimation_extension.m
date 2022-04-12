function reestimation_extension(obj, X, y, varargin)

ind = find(cellfun(@(x) isequal(x, 'Unbiased'), varargin));
if length(ind) == 1
    unbiased = varargin{ind + 1};
    varargin(ind:ind+1) = [];
else
    unbiased = false;
end

obj.train(X, y, varargin{:});
if isempty(obj.predictorNames)
    disp('no predictors');
end

obj.alphaNew = 0;
obj.betaOverall = 1;

if unbiased
    predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);
    obj.train(X(:, predInds), y, varargin{:});
    obj.alphaNew = 0;
    obj.betaOverall = 1;
end
