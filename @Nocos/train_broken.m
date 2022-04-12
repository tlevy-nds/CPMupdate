function [vals, keepInds, Bs] = train(obj, X, Y, los, predictorNames, imputeMethod, resamplingMethod, nfeatures, lambda, useCV, likelihoodDist, controlRandomNumberGeneration, coefficientAnalysis)
% set lambda to 0 when using fixed predictors
if ~exist('coefficientAnalysis', 'var') || isempty(coefficientAnalysis)
    coefficientAnalysis = false;
end

if ~exist('controlRandomNumberGeneration', 'var') || isempty(controlRandomNumberGeneration)
    controlRandomNumberGeneration = true;
end

% make the results reproducible, but will also prevent estimating the variance over multiple runs
if controlRandomNumberGeneration
    rng(0, 'twister');
end

assert(size(X, 2) == length(predictorNames), 'size(X, 2) ~= length(predictorNames)');

% verify that labels are -1 (negative outcome) and +1 (positive outcome)
negOutcome = -1;
posOutcome = 1;
if all(ismember(Y, [0 1]))
    Y(Y == 0) = negOutcome;
end
assert(all(ismember(Y, [negOutcome posOutcome])), 'Y can only have two distinct outcomes');

assert(nnz(Y == negOutcome) < nnz(Y == posOutcome), 'expected more positive outcomes than negative outcomes');

% standardize
zs = obj.standardize(X, true);

% impute missing data
[zs, keepInds] = obj.impute_data(zs, Y, imputeMethod);
Y_ = Y(keepInds);
los = los(keepInds);

useNew = false; matchFrequency = true;

% use the cross-validation loop to estimate the likelihood functions
% this is not necessary for generating an ROC curve
if isa(useCV, 'logical') && useCV || ~isempty(useCV) && isnumeric(useCV) || isstruct(useCV) || isa(useCV, 'cvpartition')
    % cross-validation loop
    [npatients, npredictors] = size(zs);
    vals = zeros(npatients, 1);
    % Bs = zeros(npredictors, npatients);    
    lambdas = zeros(1, npatients);
    
    if isnumeric(useCV)
        K = useCV;
        cv = cvpartition(Y_, 'KFold', K, 'Stratify', true);
    elseif islogical(useCV)
        cv = cvpartition(length(Y_), 'LeaveOut');
        % cv = struct('NumObservations', length(Y_), 'NumTestSets', length(Y_), ...
        %     'TrainSize', repmat(length(Y_) - 1, [1 length(Y_)]), ...
        %     'TestSize', ones(1, length(Y_)), ...
        %     'training', @(x) 1:length(Y_) ~= x, ...
        %     'test', @(x) 1:length(Y_) == x);
    else
        cv = useCV;
    end
    Bs = zeros(npredictors, cv.NumTestSets);
    
    hmsg = MessageUpdater();
    for k = 1:cv.NumTestSets       
        % looInds = setdiff(1:npatients, loo);
        
        % downsample the survival to match class frequencies in the training set
        trainingInds = find(cv.training(k));
        
        if length(unique(Y_(trainingInds))) == 1
            continue
        end
        
        if useNew
        Yti = Y_(trainingInds);
        nneg = nnz(Yti == negOutcome);
        npos = nnz(Yti == posOutcome);
        [~, mi] = min([nneg npos]);
        switch resamplingMethod
            case 'undersampleMajorityClass'
                if mi == 1
                    % nneg < npos
                    inds = sort([find(Yti == negOutcome); randsample(find(Yti == posOutcome), nneg, false)], 'ascend');
                else
                    % npos < nneg
                    inds = sort([find(Yti == posOutcome); randsample(find(Yti == negOutcome), npos, false)], 'ascend');
                end
                % zs = zs(inds, :);
                % Y_ = Y_(inds);
                trainingInds = trainingInds(inds);
            case 'oversampleMinorityClass'
                if mi == 1
                    % nneg < npos
                    inds = sort([find(Yti == posOutcome); randsample(find(Yti == negOutcome), npos, true)], 'ascend');
                else
                    % npos < nneg
                    inds = sort([find(Yti == negOutcome); randsample(find(Yti == posOutcome), nneg, true)], 'ascend');
                end
                % zs = zs(inds, :);
                % Y_ = Y_(inds);
                trainingInds = trainingInds(inds);
            case 'none'
                
            case 'generateSyntheticSamples'
                error('%s not yet implemented\n', resamplingMethod);
            otherwise
                error('%s is not a valid resampling method\n', resamplingMethod);
        end
        else
        if matchFrequency
            if nnz(Y_(trainingInds) == -1) < nnz(Y_(trainingInds) == 1)
                output1 = -1;
                output2 = 1;
            else
                output1 = 1;
                output2 = -1;
            end
            N = round(nnz(Y_(trainingInds) == output1)/1);
            trainingInds2 = trainingInds(Y_(trainingInds) == output2);
            
            if controlRandomNumberGeneration
                trainingInds = sort([trainingInds(Y_(trainingInds) == output1); trainingInds2(1:N)], 'ascend');
            else
                trainingInds = sort([trainingInds(Y_(trainingInds) == output1); randsample(trainingInds2, N)], 'ascend');
            end
        end
        end
        testInds = find(cv.test(k));
        
        alpha_ = 1;  % 0.1:.1:1
        
        [B, fitinfo] = lasso(zs(trainingInds, :), Y_(trainingInds), 'Alpha', alpha_(1), ...
            'Standardize', false, 'Lambda', lambda, 'PredictorNames', predictorNames);
        
%         inds = find(fitinfo.DF == 19, 1, 'first'):find(fitinfo.DF == 6, 1, 'last');        
%         inds2 = find(B(:, inds(1)));
%         [~, si] = sort(abs(B(inds2, inds(1))), 'descend');
%         figure;plot(log(fitinfo.Lambda(inds)), B(inds2(si), inds)'); legend(predictorNames(inds2(si)));
%         title('7-Day Horizon');
        
        for ialpha = 2:length(alpha_)
            [B_, fitinfo_] = lasso(zs(trainingInds, :), Y_(trainingInds), 'Alpha', alpha_(ialpha), ...
                'Standardize', false, 'Lambda', lambda, 'PredictorNames', predictorNames);
            B = [B B_];
            fitinfo.Intercept = [fitinfo.Intercept fitinfo_.Intercept];
            fitinfo.Lambda = [fitinfo.Lambda fitinfo_.Lambda];
            fitinfo.DF = [fitinfo.DF fitinfo_.DF];
            fitinfo.MSE = [fitinfo.MSE fitinfo_.MSE];            
        end
        % figure;plot(reshape(fitinfo.DF, [100 10]), reshape(fitinfo.MSE, [100 10]), '.-')
        
        % choose lambda
        ind = obj.choose_lambda(B, fitinfo, zs(trainingInds, :), Y_(trainingInds), nfeatures);
        lambdas(testInds) = fitinfo.Lambda(ind);
        
        % vals(loo) = X(loo, :) * B + fitinfo.Intercept;
        
        vals(testInds) = lasso_reconstruct(zs(testInds, :), B(:, ind), fitinfo.Intercept(ind), nfeatures);
        
        % Bs(:, testInds) = repmat(B(:, ind), [1 length(testInds)]);
        Bs(:, k) = B(:, ind);
        hmsg.update_message(sprintf('%d of %d\n', testInds, npatients))
    end
    delete(hmsg);
    
    compareCoeffs = false;
    if compareCoeffs
        % inds = any(Bs ~= 0, 2);
        inds = 1:size(Bs, 1);
        figure;boxplot(Bs(inds, :)');set(gca, 'XTickLabel', predictorNames(inds), 'XTickLabelRotation', 60);grid on;
        % save('BsRetroResample.mat', 'Bs');
    end
    
    obj.lambda = median(lambdas);    
else
    obj.lambda = lambda;
    vals = [];
end

if coefficientAnalysis
    return
end

% rerun lasso with the optimal lambda on all of the training data to obtain the final set of coefficients
trainingInds = (1:length(Y_))';
if useNew
nneg = nnz(Y_ == negOutcome);
npos = nnz(Y_ == posOutcome);
[~, mi] = min([nneg npos]);
switch resamplingMethod
    case 'undersampleMajorityClass'
        if mi == 1
            % nneg < npos
            inds = sort([find(Y_ == negOutcome); randsample(find(Y_ == posOutcome), nneg, false)], 'ascend');
        else
            % npos < nneg
            inds = sort([find(Y_ == posOutcome); randsample(find(Y_ == negOutcome), npos, false)], 'ascend');
        end
        % zs = zs(inds, :);
        % Y_ = Y_(inds);
        trainingInds = trainingInds(inds);
    case 'oversampleMinorityClass'
        if mi == 1
            % nneg < npos
            inds = sort([find(Y_ == posOutcome); randsample(find(Y_ == negOutcome), npos, true)], 'ascend');
        else
            % npos < nneg
            inds = sort([find(Y_ == negOutcome); randsample(find(Y_ == posOutcome), nneg, true)], 'ascend');
        end
        % zs = zs(inds, :);
        % Y_ = Y_(inds);
        trainingInds = trainingInds(inds);
    case 'none'
        
    case 'generateSyntheticSamples'
        error('%s not yet implemented\n', resamplingMethod);
    otherwise
        error('%s is not a valid resampling method\n', resamplingMethod);
end
else
if matchFrequency
    if nnz(Y_(trainingInds) == -1) < nnz(Y_(trainingInds) == 1)
        output1 = -1;
        output2 = 1;
    else
        output1 = 1;
        output2 = -1;
    end
    N = round(nnz(Y_(trainingInds) == output1)/1);
    trainingInds2 = trainingInds(Y_(trainingInds) == output2);
    trainingInds = sort([trainingInds(Y_(trainingInds) == output1); randsample(trainingInds2, N)], 'ascend');
end
end
[B, fitinfo] = lasso(zs(trainingInds, :), Y_(trainingInds), 'Alpha', 1, 'Standardize', false, ...
    'Lambda', obj.lambda, 'PredictorNames', predictorNames);

% betas = fminunc(@cross_entropy_L1_reg, [randn(size(B)); 0; 0; 3; 0; 3], [], zs, double(Y_==1), lambda, nnz(Y_==1)/length(Y_), nnz(Y_==-1)/length(Y_));
% B = betas(1:size(zs, 2));
% fitinfo.Intercept = betas(size(zs, 2)+1);
% obj.muPos = betas(size(zs, 2)+2);
% obj.stdPos = betas(size(zs, 2)+3);
% obj.muNeg = betas(size(zs, 2)+4);
% obj.stdNeg = betas(size(zs, 2)+5);

% choose lambda
ind = obj.choose_lambda(B, fitinfo, zs, Y_, nfeatures);
obj.lambda = fitinfo.Lambda(ind);

[~, si] = sort(abs(B), 'descend');
% obj.nfeatures is an upper bound
nfeatures = min(nfeatures, nnz(B));
inds = si(1:nfeatures);

obj.predictorNames = predictorNames(inds)';
obj.mus = obj.mus(inds)';
obj.sigs = obj.sigs(inds)';
obj.lassoCoeffs = B(inds);
obj.biasTerm = fitinfo.Intercept(ind);

if isempty(vals)
    vals = lasso_reconstruct(zs, obj.lassoCoeffs, obj.biasTerm, nfeatures);
else
    % plot the cross-validation result
    if true
        obj.estimate_distributions(vals, Y_, los, likelihoodDist);
        P = (obj.priorPos * obj.pdPos{1}.pdf(vals) ./ (obj.priorPos * obj.pdPos{1}.pdf(vals) + obj.priorNeg * obj.pdNeg{1}.pdf(vals)))';
        
        xvalRoc = Performance(vals, Y_, P);
        xvalRoc.plot_roc(); legend(sprintf('10-fold cross-validation (AUC = %1.3f)', xvalRoc.auc), 'Location', 'southeast');
        fig_lasso_coeffs(obj.lassoCoeffs, obj.predictorNames);
        
        xvalRoc.p_vs_p(0.1, false, '10-Fold Cross-Validation', 7);
    end
end

% need to get sign correct when mapping to P
% I can compare the means of the distributions
% xs = sum(zs .* obj.lassoCoeffs) + obj.biasTerm;

%% Posterior Probability
% I have enough data now that cross-validation isn't necessary
if (isa(useCV, 'logical') && useCV || ~isempty(useCV) && isnumeric(useCV) || isstruct(useCV) || isa(useCV, 'cvpartition')) || ...
        exist('likelihoodDist', 'var') && ~isempty(likelihoodDist) % && isequal(likelihoodDist, 'gmm')
    obj.estimate_distributions(vals, Y_, los, likelihoodDist);
    
%     p = zeros(length(vals), 2);
%     p(Y_==-1, 1) = exp(vals(Y_==-1)) ./ (1 + exp(vals(Y_==-1)));
%     p(Y_==1, 2) = exp(vals(Y_==1)) ./ (1 + exp(vals(Y_==1))) + .01;
%     P = obj.priorPos * obj.pdPos{1}.pdf(vals) ./ (obj.priorPos * obj.pdPos{1}.pdf(vals) + obj.priorNeg * obj.pdNeg{1}.pdf(vals));
%     figure;plot(vals, [p P], '.');
%     
%     mybins = quantile(vals, 0:.02:1);
%     P2 = arrayfun(@(ii1, ii2) nnz(Y_(vals > ii1 & vals <= ii2) == posOutcome) / length(Y_(vals > ii1 & vals <= ii2)), mybins(1:end-1), mybins(2:end));
%     figure;bar(mean([mybins(1:end-1); mybins(2:end)], 1), P2);grid on;
%     
%     % https://www.mathworks.com/help/stats/regularize-logistic-regression.html
%     [B,FitInfo] = lassoglm(X, (Y + 1)/2, 'binomial', 'NumLambda', 25, 'CV', 10);
%     figure; lassoPlot(B,FitInfo,'PlotType','CV'); legend('show','Location','best');
%     figure; lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
else
    % [obj.kPos, obj.kNeg, obj.muPos, obj.muNeg, obj.stdPos, obj.stdNeg, obj.priorPos, obj.priorNeg] = deal([]);
    [obj.pdPos, obj.pdNeg, obj.priorPos, obj.priorNeg] = deal([]);
end

% If the function succeeded set obj.nfeatures
obj.nfeatures = nfeatures;

