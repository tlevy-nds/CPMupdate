function [keepInds] = trainblm(obj, X, Y, imputeMethod, resamplingMethod, predictorNames, lambda, alpha_, N, cv, plotflag, controlRandomNumberGeneration)
% allow K-fold cross-validation or leave-one-hospital-out cross-validation

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
negOutcome = -1;
posOutcome = 1;
if all(ismember(Y, [0 1]))
    Y(Y == 0) = negOutcome;
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

if isa(cv, 'cvpartition')
    % pass
elseif isstruct(cv)    
    assert(isequal(resamplingMethod, 'none'), 'the resampling must be built into the cross-validation struct');    
    assert(isfield(cv, 'NumObservations') && isscalar(cv.NumObservations));
    assert(isfield(cv, 'NumTestSets') && isscalar(cv.NumTestSets));
    assert(isfield(cv, 'TrainSize') && length(cv.TrainSize) == cv.NumTestSets);
    assert(isfield(cv, 'TestSize') && length(cv.TrainSize) == cv.NumTestSets);
    assert(isfield(cv, 'training') && isa(cv.training, 'containers.Map'));
    assert(isfield(cv, 'test') && isa(cv.test, 'containers.Map'));
elseif isscalar(cv)
    cv = cvpartition(Y_, 'KFold', K, 'Stratify', true);
end

posInds = Y == posOutcome;
negInds = Y == negOutcome;

% train a Lasso logistic regression model
FitInfo = struct('DF', zeros(1, length(lambda)), 'Lambda', lambda, 'Intercept', zeros(1, length(lambda)));
[B2, muVals, sigma2Vals] = deal(cell(1, length(lambda)));
for ilambda = 1:length(lambda)
    lambda_ = lambda(ilambda);
    
    muVals{ilambda} = NaN(length(Y), 1);
    sigma2Vals{ilambda} = NaN(length(Y), 1);
    for k = 1:cv.NumTestSets
        trainingInds = cv.training(k);
        testInds = cv.test(k);
        
        B2{ilambda} = bayesian_lasso_generic(zs(trainingInds, :), Y_(trainingInds), lambda_);        
        inds = arrayfun(@(ii) sum(sign(quantile(B2{ilambda}(:, ii), [alpha_ / 2, 1 - alpha_ / 2]))) ~= 0, 1:size(B2{ilambda}, 2));

        muVals{ilambda}(posInds & testInds) = zs(posInds(testInds), inds) * mean(B2{ilambda}(:, inds), 1)';
        sigma2Vals{ilambda}(posInds & testInds) = zs(posInds(testInds), inds).^2 * var(B2{ilambda}(:, inds), 0, 1)';
        muVals{ilambda}(negInds & testInds) = zs(negInds(testInds), inds) * mean(B2{ilambda}(:, inds), 1)';
        sigma2Vals{ilambda}(negInds & testInds) = zs(negInds(testInds), inds).^2 * var(B2{ilambda}(:, inds), 0, 1)';
        
        fprintf(1, '%d of %d\n', k, cv.NumTestSets);
    end
    % get muS, sigma2S, muD, sigma2D
    % These can be my kernels, I can visualize them by summing all of the
    % RVs. I can randomly sample each RV N times and then call fitdist.
    if plotflag
        % x = linspace(min(muVals{ilambda} - 3*sqrt(sigma2Vals{ilambda})), max(muVals{ilambda} + 3*sqrt(sigma2Vals{ilambda})), 1001)';
        x = linspace(-4, 4, 1001)';
        [yS, yD] = deal(zeros(size(x)));
        for ii = 1:length(Y)
            if posInds(ii)
                yS = yS + normpdf(x, muVals{ilambda}(ii), sqrt(sigma2Vals{ilambda}(ii)));
            elseif negInds(ii)
                yD = yD + normpdf(x, muVals{ilambda}(ii), sqrt(sigma2Vals{ilambda}(ii)));
            end
        end
        figure;plot(x, [yS yD]/length(Y), '-');legend({'Survived' 'Died'});
    end
    % There was no separation with lambda = 1.5
    % With lambda = 300 I got {'Age', 'Albumin_Serum', 'Blood_Urea_Nitrogen_S…', 'Platelet_Count__Autom…', 'Red_Cell_Distrib_Width', 'CRP', 'eGFR'}
    % for the first couple k at least, but still no separation. Try
    % changing cv by downsampling the majority class. Still no separation
    % Maybe I should try lassoblm from the econometrics toolbox?    
    
    mySamples = zeros(length(Y), N);
    for ii = 1:length(Y)        
        mySamples(ii, :) = normrnd(muVals{ilambda}(ii), sqrt(sigma2Vals{ilambda}(ii)), [1 N]);
    end
    % TODO see if I need to use Pareto tails or which distribution works the best
    obj.pdNeg{1} = fitdist(reshape(mySamples(negInds, :), [], 1), 'normal');
    obj.pdPos{1} = fitdist(reshape(mySamples(posInds, :), [], 1), 'normal');
    
    inds = arrayfun(@(ii) sum(sign(quantile(B2(:, ii), [0.025 0.975]))) ~= 0, 1:size(B2, 2));
    obj.B2 = [reshape(mean(B2{ilambda}(:, inds), 1), [], 1) reshape(var(B2{ilambda}(:, inds), 0, 1), [], 1)];
    obj.nfeatures = nnz(inds);
    obj.predictorNames = predictorNames(inds)';
    
    if plotflag        
        figure(); hold on;
        for ii = 1:size(B2, 2)
            if inds(ii)
                [N, edges] = histcounts(B2(:, ii));
                plot((edges(1:end-1) + edges(2:end))/2, N, '-');
            end
        end
        legend(predictorNames(inds));
        title(sprintf('lambda = %f', lambda_));
    end
    FitInfo.DF(ilambda) = nnz(inds);
end

obj.mus = obj.mus(inds)';
obj.sigs = obj.sigs(inds)';
