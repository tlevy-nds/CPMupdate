function load(obj, filename)
load(filename, 'lambda', 'nfeatures', 'mus', 'sigs', ...
                'lassoCoeffs', 'biasTerm', 'pdPos', 'pdNeg', 'priorPos', 'priorNeg', 'predictorNames', 'trainingCvguids', 'calCorrection');

if ~exist('pdPos', 'var')
    [pdPos, pdNeg, priorPos, priorNeg] = deal([]);
end
if ~exist('trainingCvguids', 'var')
    trainingCvguids = [];
end
if ~exist('calCorrection', 'var')
    calCorrection = [];
end

[obj.lambda, obj.nfeatures, obj.mus, obj.sigs, ...
                obj.lassoCoeffs, obj.biasTerm, obj.pdPos, obj.pdNeg, obj.priorPos, obj.priorNeg, ...
                obj.predictorNames, obj.trainingCvguids, obj.calCorrection] = deal(lambda, nfeatures, mus, ...
                sigs, lassoCoeffs, biasTerm, pdPos, pdNeg, priorPos, priorNeg, predictorNames, trainingCvguids, calCorrection);
