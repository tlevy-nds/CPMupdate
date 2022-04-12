function train(obj, X, y, varargin)

p = inputParser();
addRequired(p, 'X', @(x) istable(x));
addRequired(p, 'y', @(x) isvector(x));
addParameter(p, 'maxNumSplits', 3, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'minLeafSize', 3, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'minParentSize', 3, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'method', 'LogitBoost', @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'numBins', 20, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'numLearningCycles', 10, @(x) isscalar(x) && mod(x, 1) == 0);
addParameter(p, 'numFeatures', 6, @(x) isscalar(x) && mod(x, 1) == 0);
parse(p, X, y, varargin{:});

maxNumSplits = p.Results.maxNumSplits;
minLeafSize = p.Results.minLeafSize;
minParentSize = p.Results.minParentSize;
method = p.Results.method;
numBins = p.Results.numBins;
numLearningCycles = p.Results.numLearningCycles;
numFeatures = p.Results.numFeatures;

% form the cost matrix
predictedSurvivedActuallyDied = nnz(y) / length(y);  % I want to penalize this more because it has the larger prior
predictedDiedActuallySurvived = 1 - predictedSurvivedActuallyDied;
myCost = [0 predictedSurvivedActuallyDied; ...
    predictedDiedActuallySurvived 0];

% use the templateTree as a learner
tFinal = templateTree('MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize, 'MinParentSize', minParentSize);

obj.mdl = fitcensemble(X, y, 'Method', method, 'NumBins', numBins, 'ScoreTransform', 'logit', ...
            'Cost', myCost, 'NumLearningCycles', numLearningCycles, 'Learners', tFinal);
    
[sv, si] = sort(obj.mdl.predictorImportance, 'descend');
numFeatures = min(numFeatures, length(nonzeros(sv)));
if numFeatures < size(X, 2)    
    obj.mdl = fitcensemble(X(:, si(1:numFeatures)), y, 'Method', method, 'NumBins', numBins, 'ScoreTransform', 'logit', ...
            'Cost', myCost, 'NumLearningCycles', numLearningCycles, 'Learners', tFinal);
end
        
obj.predictorNames = X.Properties.VariableNames(si(1:numFeatures));
