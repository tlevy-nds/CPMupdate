function [xs, Ps, rocStruct, keepInds] = predict(obj, X, Psurvive, Y, los, predictorNames, imputeMethod, plotflag, axroc)

if ~exist('plotflag', 'var') || isempty(plotflag)
    plotflag = false;
end

if ~exist('Psurvive', 'var') || isempty(Psurvive)
    Psurvive = obj.priorPos;    
end
Pexpire = 1 - Psurvive;
if ~exist('imputeMethod', 'var') || isempty(imputeMethod)
    imputeMethod = 'mean';
end

if exist('predictorNames', 'var') && ~isempty(predictorNames)    
    assert(isequal(reshape(obj.predictorNames, 1, []), reshape(predictorNames, 1, [])), 'predictor names don''t match');
end

if isempty(X)
    [xs, Ps, keepInds] = deal([]);
    rocStruct = struct('auc', 0);
    if plotflag
        figure(); rocStruct.ax = gca();
    else
        rocStruct.ax = [];
    end
    return
end

% standardize
zs = obj.standardize(X, false);

% impute missing data
if exist('Y', 'var') && ~isempty(Y)
    [zs, keepInds] = obj.impute_data(zs, Y, imputeMethod);
    Y = Y(keepInds);
end
% los = los(keepInds);

xs = lasso_reconstruct(zs, obj.lassoCoeffs, obj.biasTerm, obj.nfeatures);
if iscell(obj.pdNeg)
    assert(length(obj.pdNeg) == 1);
    xs = min(xs, min(obj.pdNeg{1}.icdf(0.999), obj.pdPos{1}.icdf(0.999)));  % Maimonides pediatrics caused underflow
end

% if isempty(obj.likelihoodDistPos)
if ~isempty(obj.pdPos) && ~isempty(obj.pdNeg) && ~isa(obj.pdPos, 'gmdistribution') && ~isa(obj.pdNeg, 'gmdistribution')
    % TODO likelihoodDistPos and likelihoodDistNeg
    % switch obj.likelihoodDistPos
    if length(obj.pdPos) == 1 && ~iscell(obj.pdPos)
        f1 = obj.pdPos.pdf(xs) .* Psurvive;
        f2 = obj.pdNeg.pdf(xs) .* Pexpire;
        Ps = f1 ./ (f1 + f2);
    else
        % Try separate distributions based on length of stay
        Ps = NaN(size(xs));
        for ilos = 1:size(obj.losBins, 1)
            % inds = los > obj.losBins(ilos, 1) & los <= obj.losBins(ilos, 2);
            inds = true(size(xs));
            
            f1 = obj.pdPos{ilos}.pdf(xs(inds)) .* Psurvive(:, ilos);
            f2 = obj.pdNeg{ilos}.pdf(xs(inds)) .* Pexpire(:, ilos);
            Ps(inds) =  f1 ./ (f1 + f2);            
        end
        % Ps(f1 < 1e-4)
    end
elseif ~isempty(obj.fPos) && ~isempty(obj.xPos) && ~isempty(obj.fNeg) && ~isempty(obj.xNeg) && ...
        ~isempty(obj.priorPos) && ~isempty(obj.priorNeg)
    f1 = interp1(obj.xPos, obj.fPos, xs, 'linear') .* Psurvive;
    f2 = interp1(obj.xNeg, obj.fNeg, xs, 'linear') .* Pexpire;
    Ps = f1 ./ (f1 + f2);
else
    Ps = [];
end

if ~isempty(obj.calCorrection) && ~isempty(Ps)
    Ps = min(1, max(0, obj.calCorrection(1) * Ps + obj.calCorrection(2)));    
end

% logistic recalibration
lp = log(Ps ./ (1 - Ps));
lp2 = obj.alphaNew + obj.betaOverall * lp;
Ps = 1 ./ (1 + exp(-lp2));

if exist('Y', 'var') && ~isempty(Y)
    % verify that labels are 0 (negative outcome) and 1 (positive outcome)    
    if all(ismember(Y, [-1 1]))
        Y(Y == -1) = 0;
    end
    assert(all(ismember(Y, [0 1])));
    
    rocMap = containers.Map([-1 0 1], [0 0 1]);
    changeSign = mean(xs(Y == 1)) < mean(xs(Y == 0));  % ~isempty(obj.pdPos) && ~isempty(obj.pdNeg) && obj.pdPos.mu < obj.pdNeg.mu || ...
    if (~exist('axroc', 'var') || isempty(axroc)) && plotflag
        figure();axroc = gca();
    elseif (~exist('axroc', 'var') || isempty(axroc)) && ~plotflag
        axroc = [];
    end    
    
    if isa(obj.pdPos, 'gmdistribution') && isa(obj.pdNeg, 'gmdistribution')    
        [ax, auc, tpr, fpr, thresholds, pl] = lasso_roc(Ps, Y, changeSign, rocMap, obj.nfeatures, plotflag, axroc, '', '');
    else
        [ax, auc, tpr, fpr, thresholds, pl] = lasso_roc(xs, Y, changeSign, rocMap, obj.nfeatures, plotflag, axroc, '', '');
    end
    
    if auc < 0.5
        changeSign = ~changeSign;
        delete(pl);
        set(axroc, 'ColorOrderIndex', max(1, get(axroc, 'ColorOrderIndex') - 1));
        if isa(obj.pdPos, 'gmdistribution') && isa(obj.pdNeg, 'gmdistribution') 
            rocVar = Ps;
        else
            if isscalar(Psurvive)
                rocVar = xs;
            else
                rocVar = Ps;
            end            
        end
        [ax, auc, tpr, fpr, thresholds] = lasso_roc(rocVar, Y, changeSign, rocMap, obj.nfeatures, plotflag, axroc, '', '');
    end
    
    rocStruct = struct('ax', ax, 'auc', auc, 'tpr', tpr, 'fpr', fpr, 'thresholds', thresholds);
else
    rocStruct = [];
end
