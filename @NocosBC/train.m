function train(obj, X, y, varargin)

p = inputParser();
addRequired(p, 'X', @(x) istable(x));
addRequired(p, 'y', @(x) isvector(x));
addParameter(p, 'numFeatures', 6, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'Lambda', [], @(x) isscalar(x) || isempty(x));
parse(p, X, y, varargin{:});

numFeatures = p.Results.numFeatures;
lambda_ = p.Results.Lambda;

obj.mdl.train(X{:, :}, y, [], X.Properties.VariableNames, 'mean', 'oversampleMinorityClass', numFeatures, lambda_, [], 'best');
obj.predictorNames = obj.mdl.predictorNames;
