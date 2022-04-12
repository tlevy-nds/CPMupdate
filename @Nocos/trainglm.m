function [keepInds] = trainglm(obj, X, Y, imputeMethod, resamplingMethod, lambdaCriterion, predictorNames, numLambda, K, plotflag, controlRandomNumberGeneration)
if ~exist('plotflag', 'var') || isempty(plotflag)
    plotflag = true;
end
if ~exist('controlRandomNumberGeneration', 'var') || isempty(controlRandomNumberGeneration)
    controlRandomNumberGeneration = false;
end

% make the results reproducible, but will also prevent estimating the variance over multiple runs
if controlRandomNumberGeneration
    rng(0, 'twister');
end

assert(size(X, 2) == length(predictorNames));

% verify that labels are 0 (negative outcome) and 1 (positive outcome)
negOutcome = 0;
posOutcome = 1;
if all(ismember(Y, [-1 1]))
    Y(Y == -1) = negOutcome;
end
assert(all(ismember(Y, [negOutcome posOutcome])));

% standardize
zs = obj.standardize(X, true);

% impute missing data
[zs, keepInds] = obj.impute_data(zs, Y, imputeMethod);
Y_ = Y(keepInds);

nneg = nnz(Y == negOutcome);
npos = nnz(Y == posOutcome);
[~, mi] = min([nneg npos]);

switch resamplingMethod
    case 'undersampleMajorityClass'        
         if mi == 1
             % nneg < npos
             inds = sort([find(Y == negOutcome); randsample(find(Y == posOutcome), nneg, false)], 'ascend');             
         else
             % npos < nneg
             inds = sort([find(Y == posOutcome); randsample(find(Y == negOutcome), npos, false)], 'ascend');            
         end
         zs = zs(inds, :);
         Y_ = Y_(inds);         
    case 'oversampleMinorityClass'
        if mi == 1
             % nneg < npos
             inds = sort([find(Y == posOutcome); randsample(find(Y == negOutcome), npos, true)], 'ascend');             
         else
             % npos < nneg
             inds = sort([find(Y == negOutcome); randsample(find(Y == posOutcome), nneg, true)], 'ascend');
         end
         zs = zs(inds, :);
         Y_ = Y_(inds);    
    case 'none'
        
    case 'generateSyntheticSamples'
        error('%s not yet implemented\n', resamplingMethod);
    otherwise
        error('%s is not a valid resampling method\n', resamplingMethod);
end

% train a Lasso logistic regression model
[B, FitInfo] = lassoglm(zs, Y_, 'binomial', 'NumLambda', numLambda, 'CV', K);
if plotflag
    lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best') % show legend
    lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
end

% choose lambda based on the deviance
switch lambdaCriterion
    case 'mindev'
        B0 = B(:, FitInfo.IndexMinDeviance);        
        obj.B1 = [FitInfo.Intercept(FitInfo.IndexMinDeviance); nonzeros(B0)];
        obj.lambda = FitInfo.LambdaMinDeviance;
    case '1se'
        B0 = B(:, FitInfo.Index1SE);
        obj.B1 = [FitInfo.Intercept(FitInfo.Index1SE); nonzeros(B0)];
        obj.lambda = FitInfo.Lambda1SE;    
    otherwise
        [~, mi] = min(abs(FitInfo.DF - lambdaCriterion));
        B0 = B(:, mi);
        obj.B1 = [FitInfo.Intercept(mi); nonzeros(B0)];
        obj.lambda = FitInfo.Lambda(mi);                 
end
obj.nfeatures = nnz(B0);
inds = B0 ~= 0;
obj.predictorNames = predictorNames(inds)';

% create an unbiased model using the Lasso predictors
obj.unbiasedModel = fitglm(zs(:, inds), Y_, 'linear', 'Distribution', 'binomial', 'PredictorVars', obj.predictorNames);

% preds = glmval(B1, X(:, inds), 'logit');  % biased model
% preds = unbiasedModel.predict(X(:, inds))

obj.mus = obj.mus(inds)';
obj.sigs = obj.sigs(inds)';
