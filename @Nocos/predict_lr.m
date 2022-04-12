function [PsBiased, PsUnbiased, keepInds] = predict_lr(obj, X, Y, predictorNames, imputeMethod, plotflag)

if ~exist('plotflag', 'var') || isempty(plotflag)
    plotflag = false;
end

assert(isequal(reshape(obj.predictorNames, 1, []), reshape(predictorNames, 1, [])));

if isempty(X)
    [PsBiased, PsUnbiased, keepInds] = deal([]);    
    return
end

% standardize
zs = obj.standardize(X, false);

% impute missing data
[zs, keepInds] = obj.impute_data(zs, Y, imputeMethod);
Y = Y(keepInds);

PsBiased = glmval(obj.B1, zs, 'logit');  % biased model
PsUnbiased = obj.unbiasedModel.predict(zs);

% if ~isempty(obj.calCorrection) && ~isempty(Ps)
%     Ps = min(1, max(0, obj.calCorrection(1) * Ps + obj.calCorrection(2)));    
% end

if exist('Y', 'var') && ~isempty(Y) && plotflag
    % verify that labels are 0 (negative outcome) and 1 (positive outcome)    
    if all(ismember(Y, [-1 1]))
        Y(Y == -1) = 0;
    end
    assert(all(ismember(Y, [0 1])));
    
    biasedPerf = Performance(PsBiased, Y, PsBiased);
    biasedPerf.plot_roc();title(sprintf('AUC = %1.3f', biasedPerf.auc));
    biasedPerf.p_vs_p(0.1, false, 'Biased', 0);    
    
    unbiasedPerf = Performance(PsUnbiased, Y, PsUnbiased);
    unbiasedPerf.plot_roc();title(sprintf('AUC = %1.3f', unbiasedPerf.auc));
    unbiasedPerf.p_vs_p(0.1, false, 'Unbiased', 0);    
end
