function [zs, keepinds] = impute_data(zs, Y, imputeMethod)

if ~exist('imputeMethod', 'var')
    imputeMethod = 'mean';
end

% Impute
% keepinds = true(length(Y), 1);
keepinds = true(size(zs, 1), 1);
switch imputeMethod
    case 'mean'
        zs(~isfinite(zs)) = 0;
    case 'exclude'     
        keepinds = ~any(~isfinite(zs), 2);
        zs = zs(keepinds, :);
        % Y = Y(keepinds);
    case 'knn'
        missing1 = nnz(~isfinite(zs));        
        zs = knnimpute(zs', 5)';
        missing2 = nnz(~isfinite(zs));
        fprintf(1, 'missing: from %d to %d\n', missing1, missing2);
        zs(~isfinite(zs)) = 0;
    case 'random'
        % [X3_, Y_, myPredictors_] = impute_random(zs, Y, myPredictors, N1, N2)
    case 'mice'        
        missingMask = [];        
        [zs, Bs, missingMask] = mice_impute(zs, 'NumCycles', 10, 'Mask', missingMask);
    otherwise
        error('no impute mode specified');
end
