function train(obj, X, y, varargin)

p = inputParser();
addRequired(p, 'X', @(x) istable(x));
addRequired(p, 'y', @(x) isvector(x));
addParameter(p, 'numFeatures', 6, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'Lambda', [], @(x) isvector(x) || isempty(x));
% addParameter(p, 'setStandardization', false, @(x) isscalar(x) && islogical(x));
parse(p, X, y, varargin{:});

numFeatures = p.Results.numFeatures;
lambda_ = p.Results.Lambda;
% setStandardization = p.Results.setStandardization;

mymean = mean(X{:, :}, 1, 'omitnan');
mystd = std(X{:, :}, 0, 1, 'omitnan');

temp = X{:, :};
% temp(~isfinite(temp)) = 0;
[B, FitInfo] = lassoglm(temp, y,  'binomial', 'Link', 'logit', 'Lambda', lambda_);  % Standardize is true by default
coefs = obj.select_lambda(B, FitInfo, numFeatures);
inds = [true; coefs(2:end) ~= 0];

if nnz(inds(2:end)) < numFeatures - 1
    [B, FitInfo] = lassoglm(temp, y,  'binomial', 'Link', 'logit', 'Lambda', []);
    coefs = obj.select_lambda(B, FitInfo, numFeatures);
    inds = [true; coefs(2:end) ~= 0];
end
    
numFeatures = nnz(inds(2:end));

% train is biased, functions that call train can choose to retrain the unbiased result
% temp = glmfit((X{inds, outfit.locus1(find(coefs(2:end) ~= 0))}), y, 'binomial', 'Link', 'logit');

% pad dimensions
% temp = [coefs(1); nonzeros(coefs(2:end))];
% outfitInds{ii} = [find(ismember(outfit.locus1, find(coefs(2:end) ~= 0))); ...
%     ones(size(thetahat{ii}, 1) - length(temp), 1)];

obj.mdl.thetahat = {coefs(inds)};
obj.mdl.Sigmahat = {diag([0.05; ones(numFeatures, 1)])};

obj.mdl.mean = mymean(inds(2:end));
obj.mdl.std = mystd(inds(2:end));

obj.predictorNames = X.Properties.VariableNames(inds(2:end));
