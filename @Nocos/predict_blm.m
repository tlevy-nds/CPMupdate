function [Ps, keepInds] = predict_blm(obj, X, Y, predictorNames, imputeMethod, plotflag)

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


% Do I need to do cross-validation in the training to determine the
% distributions?
% https://math.stackexchange.com/questions/1246358/the-product-of-multiple-univariate-gaussians
% sigma2 = 1/sum(sigma^-2)
% mu = sigma2 * sum(sigma^-2 * mu)
% S = (2*pi)^(1-n)/2 * sigma/prod(sigma) * exp(0.5*sigma^-2*mu^2 - 0.5*sum(sigma^-2*mu^2))
% This forces the distributions to be Gaussian. I think the covariance was
% diagonal too. 

% zs N x p
% B2 p x 2
mu_ = zs * obj.B2(:, 1);
sigma2 = zs.^2 * obj.B2(:, 2);

% To determine the posterior, I would integrate over the product of the
% trained distributions and the distributions from the testing samples.
% It will take narrow overlapping functions to get large values. Is there a
% normalization factor I have to incorporate? I think applying Bayes
% Theorem should take care of it.

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
